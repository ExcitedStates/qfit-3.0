'''
Excited States software: qFit 3.0

Contributors: Saulo H. P. de Oliveira, Gydo van Zundert, and Henry van den Bedem.
Contact: vdbedem@stanford.edu

Copyright (C) 2009-2019 Stanford University
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

This entire text, including the above copyright notice and this permission notice
shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
'''

import argparse
import sys
import numpy as np
import random
import copy
from .vdw_radii import vdwRadiiTable, EpsilonTable
from .structure import Structure


def cartesian_product(*arrays):
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=int)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def update_progress(progress):
    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rProgress: [{0}] {1:2.0f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


class RelabellerOptions:
    def __init__(self, nSims=10000, nChains=10):
        self.nSims = nSims
        self.nChains = nChains

    def apply_command_args(self, args):

        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class Relabeller:
    "Relabel alternate conformers using SA"

    def __init__(self, structure, options):
        self.structure = structure
        self.nSims  = options.nSims
        self.nChains= options.nChains

        self.nodes    = []
        self.permutation = []
        self.initNodes()

        self.metric = np.full((len(self.nodes),len(self.nodes)),0.0)
        self.initMetric()

    def initNodes(self):
        node = 0
        segment = []
        for chain in self.structure:
            for residue in chain:
                resInd=[]
                for altloc in list(set(residue.altloc)):
                    if altloc != '':
                        self.nodes.append(residue.extract('altloc', altloc))
                        resInd.append(node)
                        node+=1
                self.permutation.append(resInd)

    def initMetric(self):
        # print("Calculating all possible Van der Waals interactions:")
        for i in range(len(self.nodes)):
            for j in range(i+1,len(self.nodes)):
                if self.nodes[i].resi[0]!=self.nodes[j].resi[0] or self.nodes[i].chain[0]!=self.nodes[j].chain[0]:
                    self.metric[i][j] = self.calc_energy(self.nodes[i],self.nodes[j])
                    self.metric[j][i] = self.metric[i][j]
            # update_progress(i/len(self.nodes))


    def vdw_energy(self,atom1, atom2, coor1, coor2):
        e = EpsilonTable[atom1][atom2]
        s = (vdwRadiiTable[atom1]+vdwRadiiTable[atom2]) / 1.122
        r = np.linalg.norm(coor1 - coor2)
        return 4 * e * (np.power(s/r, 12) - np.power(s/r, 6))

    def calc_energy(self, node1, node2):
        energy = 0.0
        if np.linalg.norm(node1.coor[0]-node2.coor[0]) < 16.0:
            for name1,ele1,coor1 in zip(node1.name,node1.e,node1.coor):
                for name2,ele2,coor2 in zip(node2.name,node2.e,node2.coor):
                    if name1 not in ["N","CA","C","O","H","HA"] or name2 not in ["N","CA","C","O","H","HA"] or np.abs(node1.resi[0] - node2.resi[0]) != 1:
                        energy += self.vdw_energy(ele1,ele2,coor1,coor2)
        return energy

    def SimulatedAnnealing(self,permutation):
        energyList = []
        NumOfClusters = len(max(permutation,key=len))
        energies = np.zeros(NumOfClusters)
        clusters = [[] for x in range(NumOfClusters)]

        # Use the permutation to identify the clusters:
        for i,elem  in enumerate(permutation):
            for j in range(len(elem)):
                clusters[j].append(elem[j])

        # Calculate the energy of each cluster:
        for i,cluster in enumerate(clusters):
            b=cartesian_product(cluster,cluster)
            energies[i] = np.sum(self.metric[b[:,0],b[:,1]])/2

        # Sum the energy of each cluster:
        energyList.append(np.sum(energies))
        # print(f"Starting energy: {energyList[-1]}")

        for i in range(self.nSims):
            # update_progress(i/self.nSims)
            temperature = 273*(1-i/self.nSims)
            # Perturb the current solution:
            tmpPerm = copy.deepcopy(permutation)
            # Choose five residues to swap:
            for x in np.random.choice(range(len(tmpPerm)),5):
                # If the residue has a single conformer:
                if not tmpPerm[x]:
                    continue
                # Identify the indexes of the segment the residue belongs to:
                l_index = x - 1
                while l_index >= 0:
                    if not tmpPerm[l_index]:
                        break
                    if len(tmpPerm[l_index]) != len(tmpPerm[x]):
                        break
                    occ1 = np.min(self.nodes[tmpPerm[l_index][0]].q )
                    occ2 = np.min(self.nodes[tmpPerm[x][0]].q )
                    if occ1 != occ2 or occ1 > 0.95:
                        break
                    l_index -= 1
                l_index += 1
                u_index = x + 1
                while u_index < len(tmpPerm):
                    if not tmpPerm[u_index]:
                        break
                    if len(tmpPerm[u_index]) != len(tmpPerm[x]):
                        break
                    occ1 = np.min(self.nodes[tmpPerm[u_index][-1]].q )
                    occ2 = np.min(self.nodes[tmpPerm[x][-1]].q )
                    if occ1 != occ2 or occ1 > 0.95:
                        break
                    u_index += 1
                u_index -= 1
                ordering = list(range(len(tmpPerm[x])))
                np.random.shuffle(ordering)
                for counter in range(l_index,u_index+1):
                    tmpPerm[counter] = [tmpPerm[counter][li] for li in ordering]
                # np.random.shuffle(tmpPerm[x])

            # Calculate the new clusters:
            tmpCluster  = [[] for x in clusters]
            for ii,value in enumerate(tmpPerm):
                for j in range(len(value)):
                    tmpCluster[ j ].append(value[j])

            # Calculate the energy across all clusters
            tmpEnergies = np.zeros_like(energies)
            for i,cluster in enumerate(tmpCluster):
                b=cartesian_product(cluster,cluster)
                tmpEnergies[i] = np.sum(self.metric[b[:,0],b[:,1]])/2

            #print(f" {np.sum(tmpEnergies)}")
            # If the new energy is better than the old one:
            if  np.sum(energies) >= np.sum(tmpEnergies) or np.exp(-(np.sum(tmpEnergies)-np.sum(energies))/temperature) >= np.random.uniform() :
                energies = tmpEnergies[:]
                clusters = tmpCluster[:]
                permutation = copy.deepcopy(tmpPerm)
                energyList.append(np.sum(energies))

        # print(f"Locally optimal energy: {energyList[-1]}")
        return energyList[-1], permutation

    def run(self):
        perm = []
        energyList = []

        for i in range(self.nChains):
            # print(f"\nRunning iteration {i+1} of Simulated Annealing")
            energy, permutation = self.SimulatedAnnealing(self.permutation)
            energyList.append(energy)
            perm.append(permutation)

        minIdx = energyList.index(min(energyList))
        # Relabel the residues:
        Altlocs = ['A','B','C','D','E']
        node = 0
        res = 0
        for chain in self.structure:
            for residue in chain:
                tmpAltlocs = copy.deepcopy(residue.altloc)
                for altloc in list(set(residue.altloc)):
                    if altloc == '':
                        continue
                    else:
                        Idx = perm[minIdx][res].index(node)
                        new_altloc = Altlocs[Idx]
                        mask = (residue.altloc == altloc)
                        tmpAltlocs[mask] = new_altloc
                    node+=1
                residue.altloc=copy.deepcopy(tmpAltlocs)
                res+=1
        self.structure.reorder()
        # self.structure.tofile("Test_relabel.pdb")
        return self.structure

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str,
            help="PDB-file containing structure.")

    # MCMC options
    p.add_argument("-S", "--number-of-sims", type=int, dest="nSims", default=10000, metavar="<int>",
            help="Number of simulations for MCMC.")
    p.add_argument("-NC", "--number-of-chains", type=int,dest="nChains", default=10, metavar="<int>",
            help="Number of chains for MCMC.")

    args = p.parse_args()
    return args

def main():
    args = parse_args()
    random.seed(17)
    # Extract residue and prepare it
    structure = Structure.fromfile(args.structure).reorder()

    options = RelabellerOptions()
    options.apply_command_args(args)

    relabeller = Relabeller(structure,options)
    relabeller.run()

if __name__== "__main__":
    main()
