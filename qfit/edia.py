"""Use EDIA to assess quality of model fitness to electron density"""
import numpy as np
from . import Structure, XMap, ElectronDensityRadiusTable
from . import ResolutionBins, BondLengthTable
import argparse
import logging
import os
import time
logger = logging.getLogger(__name__)


class ediaOptions:
    def __init__(self):
        # General options
        self.directory = '.'
        self.debug = False

        # Density creation options
        self.map_type = None
        self.resolution = None
        self.resolution_min = None
        self.scattering = 'xray'

    def apply_command_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class Weight():
    def __init__(self, radius):
        # Per definition:
        self.b1 = 1.0 #(maximum of first parabola)
        self.b2 = -0.4 #(minimum of second parabola)
        self.b3 = 0.0 #(maximum of third parabola)
        self.c1 = 0.0 # (we want the first parabola to have its maximum at x=0)
        self.m1 = -1.0/(radius**2) # This ensures that the density becomes superfluous if p is in d(a)
        self.c3 = 2 * radius       # (we want the third parabola to have its maximum at x=2*r)
        self.r0 = 1.0822*radius    # The point where P1 and P2 intersect (pre-defined)
        self.r2 = 2*radius         # The point where P3 becomes 0.

        # Calculate unknowns:
        # Unknowns: r1,m2,m3,c2
        self.c2 = -(self.b1-self.b2)/(self.m1*self.r0)
        self.m2 = (self.r0**2) * (self.m1**2) / ((self.r0**2)*self.m1 - self.b2 + self.b1)
        self.r1 = (self.m2*self.c2*self.c3 - self.m2*self.c2*self.c2 -self.b2)/ (self.m2*self.c3 - self.m2*self.c2)
        self.m3 = self.m2*(self.r1-self.c2) / (self.r1 - self.c3)

        self.P  = lambda x,m,c,b: m*(x-c)**2 + b

    def __call__(self, dist):
        # Calculate the weight:
        if(dist<self.r0):
            return self.P(dist,self.m1,self.c1,self.b1)
        elif(dist<self.r1):
            return self.P(dist,self.m2,self.c2,self.b2)
        elif(dist<self.r2):
            return self.P(dist,self.m3,self.c3,self.b3)
        else:
            return 0.0

class Point():
    def __init__(self,coor):
        self.coor = coor
        self.S=[]
        self.D=[]

    def set_Point(self, new_point):
        self.coor = new_point.coor
        self.S=new_point.S
        self.D=new_point.D

class _BaseEDIA():
    def __init__(self, conformer, structure, xmap, options):
        self.structure = structure
        self.conformer = conformer
        self.residue = conformer
        self.xmap = xmap
        self.options = options
        self._coor_set = [self.conformer.coor]
        self._voxel_volume = self.xmap.unit_cell.calc_volume() / self.xmap.array.size
        self.weighter = Weight(1.0)
        # Calculate space diagonal and the partitioning factor p
        self.d = np.linalg.norm(xmap.voxelspacing)
        self.p = np.ceil(self.d/0.7)
        abc = np.asarray([self.xmap.unit_cell.a, self.xmap.unit_cell.b, self.xmap.unit_cell.c])
        self.grid_to_cartesian = np.transpose( ( self.xmap.unit_cell.frac_to_orth / abc ) * self.xmap.voxelspacing )
        self.cartesian_to_grid = np.linalg.inv(self.grid_to_cartesian)
        self.Grid = np.zeros_like(xmap.array, dtype=object)
        self.mean = xmap.array.mean()
        self.sigma = xmap.array.std()
        self.Populate_Grid(self.residue)

    def Populate_Grid(self,target_residue=None):
      for chain in self.structure:
        for residue in chain:
             for ind in range(len(residue.name)):
                  atom,element,charge,coor,icode,record,occ,resi = residue.name[ind],residue.e[ind],residue.charge[ind],residue.coor[ind],residue.icode[ind],residue.record[ind],residue.q[ind],residue.resi[ind]

                  if target_residue!=None:
                      flag=0
                      for idx in range(len(target_residue.name)):
                          if np.linalg.norm(coor-target_residue.coor[idx])<2.16*2+0.2:
                              flag=1
                              break
                  if flag == 0:
                      continue

                  grid = np.dot(coor,self.cartesian_to_grid).astype(int) - np.asarray(self.xmap.offset) # (i,j,k)
                  if element == "H":
                      continue
                  if charge == '':
                       ed_radius = self.calculate_density_radius(element, self.options.resolution,int(residue.b[ind]))
                  else:
                       ed_radius = self.calculate_density_radius(element, self.options.resolution,int(residue.b[ind]),charge)
                  box = (np.ceil(ed_radius*2/self.xmap.voxelspacing)).astype(int)

                  # Iterate over all grid points in the box and calculate their ownership
                  for i in range(grid[2]-box[2],grid[2]+box[2]):
                      for j in range(grid[1]-box[1],grid[1]+box[1]):
                          for k in range(grid[0]-box[0],grid[0]+box[0]):
                              try:
                                  dist = np.linalg.norm(coor-self.Grid[i][j][k].coor)
                              except:
                                  self.Grid[i][j][k]=Point(np.dot(np.asarray([k,j,i])+np.asarray(self.xmap.offset),self.grid_to_cartesian))
                                  dist = np.linalg.norm(coor-self.Grid[i][j][k].coor)
                              if(dist<ed_radius):
                                  self.Grid[i][j][k].S.append([coor,atom,element,occ,resi])
                              elif(dist<ed_radius*2):
                                  self.Grid[i][j][k].D.append([coor,atom,element,occ,resi])


    # Calculates the atomic radius based on the table
    def calculate_density_radius(self,atom, resolution,bfactor,charge="0"):
        a = int(np.floor(resolution/0.5)-1)
        b = int(np.ceil(resolution/0.5)-1)

        if atom not in ElectronDensityRadiusTable.keys():
            atom = atom[0]+atom[1:].lower()
        if charge not in ElectronDensityRadiusTable[atom].keys():
            charge = charge[::-1]
        if a == b:
            radius = ElectronDensityRadiusTable[atom][charge][a]
        else:
            radius = ElectronDensityRadiusTable[atom][charge][a]+(ElectronDensityRadiusTable[atom][charge][b]-ElectronDensityRadiusTable[atom][charge][a])*(resolution - ResolutionBins[a])/(ResolutionBins[b] - ResolutionBins[a])
        return np.asarray(radius)

    def ownership(self, p, dist, ed_radius,S,D,I):
        if (dist/ed_radius >= 2.0): # Grid point p is too far from the atom...
            o=0.0
        elif (dist/ed_radius >= 1.0): # Grid point p is in d(atom)
           if len(S)> 0: # Another atom owns the grid point
               o=0.0
           else:
               if len(D)==1: # No other atom owns the grid point, target atom is the only atom in D.
                   o=1.0
               else:   # Ownership of the atom is adjusted by the contribution of all atoms in D.
                   o = 1 - dist/sum([ np.linalg.norm(p-atom[0]) for atom in D ])
        else:
             if len(I)==1: # Target atom is the only atom that owns the grid point.
                 o=1.0
             else: # Ownership of the atom is adjusted by the contribution of other atoms that own the point.
                 o = 1 - dist/sum([ np.linalg.norm(p-atom[0]) for atom in I ])
        return o

    def print_density(self,contour=1.0):
        for i in range(0,len(self.xmap.array)):
            for j in range(0,len(self.xmap.array[i])):
                for k in range(0,len(self.xmap.array[i][j])):
                    if(self.xmap.array[i][j][k] - self.mean >contour*self.sigma):
                        coor = np.dot(np.asarray([k,j,i])+np.asarray(self.xmap.offset),self.grid_to_cartesian)
                        print("HETATM {0:4d}  H   HOH A {0:3d}    {1:8.3f}{2:8.3f}{3:8.3f}  1.00 37.00           H".format(1,coor[0],coor[1],coor[2]))

    def print_stats(self):
        # Note that values of the offset are based on C,R,S - these are not always ordered like x,y,z
        offset=self.xmap.offset
        voxelspacing=self.xmap.voxelspacing # These ARE ordered (x,y,z)
        print("Unit cell shape:", self.xmap.unit_cell.shape) # These are ordered (z,y,x)
        print("Unit cell a,b,c: {0:.2f} {1:.2f} {2:.2f}".format(self.xmap.unit_cell.a, self.xmap.unit_cell.b, self.xmap.unit_cell.c)) # These ARE ordered (x,y,z)
        print("Unit cell alpha,beta,gamma: {0:.2f} {1:.2f} {2:.2f}".format(self.xmap.unit_cell.alpha,self.xmap.unit_cell.beta,self.xmap.unit_cell.gamma))
        print("XMap array dimentions: ", [len(self.xmap.array),len(self.xmap.array[0]),len(self.xmap.array[0][0])]) # These are ordered (z,y,x)
        abc = np.asarray([self.xmap.unit_cell.a, self.xmap.unit_cell.b, self.xmap.unit_cell.c])
        print("abc/voxelspacing:",abc/self.xmap.voxelspacing)
        print("Offset: ",offset)

    # Returns 1 if atom_a is covalently bonded to atom_b, 0 otherwise.
    def covalently_bonded(self,atom_a,atom_b):
        error = 2*0.06 # two standard deviations from the largest observed standard deviation for protein bond lengths
        try:
            if np.linalg.norm(np.asarray(atom_a[0])-np.asarray(atom_b[0])) < float(BondLengthTable[atom_a[2]][atom_b[2]]) + error:
                return 1
        except:
            return 0
        return 0

    # Find all atoms in 'set' that are not covalently bonded to 'atom_a'
    def calculate_non_bonded(self,atom_a,set):
        I = []
        for atom_b in set:
            if atom_a[4] == atom_b[4] and atom_a[3]!=atom_b[3] and atom_a[3]<1.0:
                continue
            if not self.covalently_bonded(atom_a,atom_b) or np.linalg.norm(np.asarray(atom_a[0])-np.asarray(atom_b[0])) <0.01:
                I.append(atom_b)
        return I

    def calc_edia(self,atom,element,charge,coor,occ,resi,bfactor):
        # Identify the closest grid point to the cartesian coordinates of the atom
        grid = np.dot(coor,self.cartesian_to_grid).astype(int) - np.asarray(self.xmap.offset) # (x,y,z)
        # Look up the electron density radius on the lookup table:
        if charge == '':
             ed_radius = self.calculate_density_radius(element, self.options.resolution,bfactor)
        else:
             ed_radius = self.calculate_density_radius(element, self.options.resolution,bfactor,charge)
        # Update the parabolas used for Weighing
        self.weighter = Weight(ed_radius)
        # Define a box of grid points that inscribes the sphere of interest
        box = (np.ceil(ed_radius*2/self.xmap.voxelspacing)).astype(int) # (x,y,z)
        sum_pos_weights = sum_neg_weights = sum_product = sum_pos_product = sum_neg_product = 0.0
        # Iterate over all grid points in the box and calculate their contribution to the EDIA score.

        for i in range(grid[2]-box[2],grid[2]+box[2]): # z
            for j in range(grid[1]-box[1],grid[1]+box[1]): # y
                for k in range(grid[0]-box[0],grid[0]+box[0]): # x
                    # Identify the coordinates of grid point (k,j,i) of density self.xmap.array[i][j][k]
                    p = self.Grid[i][j][k].coor
                    #if(self.xmap.array[i][j][k] - self.mean > 1.2*self.sigma):
                    #    print("HETATM {0:4d}  H   HOH A {0:3d}    {1:8.3f}{2:8.3f}{3:8.3f}  1.00 37.00           H".format(1,p[0],p[1],p[2]))
                    #continue
                    dist = np.linalg.norm(coor-p)
                    # Calculate the distance-dependent weighting factor w
                    weight = self.weighter(dist)
                    # Calculate the ownership value o
                    I = self.calculate_non_bonded([coor,atom,element,occ,resi],self.Grid[i][j][k].S)
                    o = self.ownership(p, dist, ed_radius,self.Grid[i][j][k].S,self.Grid[i][j][k].D,I)
                    # Calculate the density score z(p) truncated at 1.2σs
                    z=min(max((self.xmap.array[i][j][k]-self.mean)/self.sigma,0.0),1.2)
                    #print(atom,dist,weight,o,z)
                    # Calculate the sums for EDIA
                    if weight > 0.0:
                        sum_pos_weights += weight
                        sum_pos_product += weight*o*z
                    else:
                        sum_neg_weights += weight
                        sum_neg_product += weight*o*z
                    sum_product += weight*o*z

        return sum_pos_product/sum_pos_weights,sum_neg_product/sum_neg_weights,sum_product/sum_pos_weights

    def calc_edia_residue(self,residue):
         length={}
         ediasum={}
         occupancy={}
         # Create arrays to store the EDIA components of the
         edia = np.zeros(len(residue.name))
         edia_plus = np.zeros(len(residue.name))
         edia_minus = np.zeros(len(residue.name))
         prev_altloc=residue.altloc[0]
         # For each atom in the residue:
         for ind in range(len(residue.name)):
             atom,element,charge,coor,icode,record,occ = residue.name[ind],residue.e[ind],residue.charge[ind],residue.coor[ind],residue.icode[ind],residue.record[ind],residue.q[ind]
             # By default, Hydrogens are excluded from the calculation!
             if element == "H":
                 continue
             # Store the values of the negative, positive, and full component in the atomic arrays:
             edia_plus[ind],edia_minus[ind],edia[ind] = self.calc_edia(atom,element,charge,coor,occ,residue.resi[ind],residue.b[ind])

             # Currently, we are truncating the negative values of EDIA at 0.
             if edia[ind] < 0.0:
                 edia[ind] = 0.0

             if residue.altloc[ind] not in ediasum:
                 ediasum[residue.altloc[ind]]=0.0
                 length[residue.altloc[ind]]=0.0
                 occupancy[residue.altloc[ind]]=residue.q[ind]
             ediasum[residue.altloc[ind]]+=(edia[ind]+0.1)**(-2)
             length[residue.altloc[ind]]+=1

         EDIAm_Comb=0.0
         for key in ediasum:
             if length[key] > 0:
                 if key != "" and "" in ediasum:
                     flag=1
                     ediasum[key] +=  ediasum[""]
                     length[key] +=  length[""]
                 EDIAm = ( ediasum[key] / length[key] ) ** (-0.5) - 0.1
                 OPIA = self.calc_opia_residue(residue,edia,key)
                 if key != "":
                     EDIAm_Comb+=occupancy[key]*EDIAm
                     print("{0} {1} {2:.2f} {3:.2f} {4:.2f}".format(residue.resi[0],key,occupancy[key],EDIAm,OPIA))
         try:
             print("{0} Comb {1:.2f} {2:.2f} {3:.2f}".format(residue.resi[0],sum(occupancy.values())-occupancy[""],EDIAm_Comb,OPIA))
         except:
             print("{0} Comb {1:.2f} {2:.2f} {3:.2f}".format(residue.resi[0],sum(occupancy.values()),EDIAm_Comb,OPIA))
         if ""  in ediasum and len(list(set(ediasum.keys()))) == 1:
           if length[""] > 0:
             key=""
             EDIAm = ( ediasum[key] / length[key] ) ** (-0.5) - 0.1
             OPIA = self.calc_opia_residue(residue,edia,"")
             print("{0} A 1.0 {2:.2f} {3:.2f}".format(residue.resi[0],EDIAm,OPIA))

         return EDIAm,OPIA

    def calc_opia_residue(self,residue,edia,key):
        altloc = [ x for i,x in enumerate(residue.altloc) if x==key or x==""]
        index_altloc = [ i for i,x in enumerate(residue.altloc) if x==key or x==""]

        self.adj_matrix = np.zeros( ( len(altloc),len(altloc) ),dtype=int)
        # Calculate adjacency matrix
        for x,i in enumerate(index_altloc):
            atom_a = [residue.coor[i],residue.name[i],residue.e[i]]
            if edia[i] >= 0.8:
                for y,j in enumerate(index_altloc):
                    atom_b = [residue.coor[j],residue.name[j],residue.e[j]]
                    if self.covalently_bonded(atom_a,atom_b):
                        self.adj_matrix[x][y]=1
                        self.adj_matrix[y][x]=1

        # Initialize all vertices as not visited
        self.visited = np.zeros(len(altloc),dtype=int)

        # Perform DFS search to identify the connected components
        label = 1
        for i in range(len(altloc)):
            if self.visited[i]==0:
                self.DFS(i,label)
            label+=1

        # Calculate OPIA
        coverage=0
        for i in range(len(np.bincount(self.visited))):
            if np.bincount(self.visited)[i] >= 2:
                coverage += np.bincount(self.visited)[i]
        return coverage/len(altloc)

    def DFS(self,residue,label):
        if self.visited[residue] != 0:
            return
        else:
            self.visited[residue] = label
            for i in range(len(self.adj_matrix)):
                if self.adj_matrix[residue][i]:
                    self.DFS(i,label)

class ediaResidue(_BaseEDIA):
    def __init__(self, residue, structure, xmap, options):
        super().__init__(residue, structure, xmap, options)

    def __call__(self):
         #self.print_stats()
         #self.print_density(2.5)
         EDIAm, OPIA = self.calc_edia_residue(self.residue)

class ediaProtein(_BaseEDIA):
    def __init__(self, structure, xmap, options):
        super().__init__(structure, structure, xmap, options)
        self.EDIAm = np.zeros(len(list(self.structure.residues)))
        self.OPIA = np.zeros(len(list(self.structure.residues)))

    def __call__(self):
      #self.print_stats()
      #self.print_density(3.0)
      for chain in self.structure:
         idx=0
         for residue in chain:
             self.EDIAm[idx],self.OPIA[idx] = self.calc_edia_residue(residue)
             # Calculate the values of EDIAm for the residue:
             print("{0} {1:.2f} {2:.2f}".format(residue.id[0],self.EDIAm[idx],self.OPIA[idx]))
             idx+=1

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xmap", type=str,
            help="X-ray density map in CCP4 format.")
    p.add_argument("resolution", type=float,
            help="Map resolution in angstrom.")
    p.add_argument("structure", type=str,
            help="PDB-file containing structure.")

    p.add_argument('--selection', default=None, type=str,
            help="Chain, residue id, and optionally insertion code for residue in structure, e.g. A,105, or A,105:A.")
    p.add_argument("-d", "--directory", type=os.path.abspath, default='.', metavar="<dir>",
            help="Directory to store results.")
    p.add_argument("-v", "--verbose", action="store_true",
            help="Be verbose.")

    return p.parse_args()

""" Main function """
def main():

    args = parse_args()
    """ Create the output directory provided by the user: """
    try:
        os.makedirs(args.directory)
    except OSError: # If directory already exists...
        pass

    time0 = time.time() # Useful variable for profiling run times.

    """ Processing input structure and map """
    # Read structure in:
    structure = Structure.fromfile(args.structure)
    # This line would ensure that we only select the '' altlocs or the 'A' altlocs.
    structure = structure.extract('altloc', ('', 'A','B','C','D','E'))

    if args.selection is not None:
        chainid, resi = args.selection.split(',')
        # Select all residue conformers
        chain = structure[chainid]
        for res in chain:
            if res.resi[0] == int(resi):
                residue = res
                break
    # Prepare X-ray map
    xmap = XMap.fromfile(args.xmap)

    options = ediaOptions()
    options.apply_command_args(args)

    if args.selection is None:
    	edia = ediaProtein(structure, xmap, options)
    else:
        edia = ediaResidue(residue,structure,xmap,options)
    edia()

    """ Profiling run time: """
    passed = time.time() - time0
#    print(f"Time passed: {passed}s")
