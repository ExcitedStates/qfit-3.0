import numpy as np

from .base_structure import _BaseStructure
from .math import dihedral_angle, Rz
from .rotamers import ROTAMERS


_PROTEIN_BACKBONE_ATOMS = ['N', 'CA', 'C']
_NUCLEOTIDE_BACKBONE_ATOMS = ['P', "O5'", "C5'", "C4'", "C3'", "O3"]
_SOLVENTS = ['HOH']


def residue_type(residue):
    """Check residue type.

    The following types are recognized:
        standard-residue
        aa-residue
            Amino acid residue
        residue
            DNA / RNA residue
        solvent
            HOH
        ligand
            Other
    """

    # RNA and DNA backbone atoms
    if residue.resn[0] in ROTAMERS:
        return 'rotamer-residue'
    bb_atoms = _PROTEIN_BACKBONE_ATOMS
    if residue.select('name', bb_atoms).size == 3:
        return 'aa-residue'
    xna_atoms = _NUCLEOTIDE_BACKBONE_ATOMS
    if residue.select('name', xna_atoms).size == 6:
        return 'residue'
    if residue.resn[0] in _SOLVENTS:
        return 'solvent'
    return 'ligand'


class _BaseResidue(_BaseStructure):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = (kwargs['resi'], kwargs['icode'])
        self.type = kwargs['type']

    def __repr__(self):
        resi, icode = self.id
        string = "Residue: {}".format(resi)
        if self.id[1] != "":
            string += ":{}".format(icode)
        return string


class _Residue(_BaseResidue):
    pass


class _RotamerResidue(_BaseResidue):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        resname = self.resn[0]
        self._rotamers = ROTAMERS[resname]
        self.nchi = self._rotamers['nchi']
        self.nrotamers = len(self._rotamers['rotamers'])
        self.rotamers = self._rotamers['rotamers']
        self._init_clash_detection()

    def _init_clash_detection(self):
        # Setup the condensed distance based arrays for clash detection and fill them
        self._ndistances = self.natoms * (self.natoms - 1) // 2
        self._clash_mask = np.ones(self._ndistances, bool)
        self._clash_radius2 = np.zeros(self._ndistances, float)
        radii = self.covalent_radius
        bonds = self._rotamers['bonds']
        offset = self.natoms * (self.natoms - 1) // 2
        for i in range(self.natoms - 1):
            #starting_index = int(offset - sp_comb(self.natoms - i, 2)) - i - 1
            natoms_left = self.natoms - i
            starting_index = offset - natoms_left * (natoms_left - 1) // 2 - i - 1
            name1 = self.name[i]
            covrad1 = radii[i]
            for j in range(i + 1, self.natoms):
                bond1 = [name1, self.name[j]]
                bond2 = bond1[::-1]
                covrad2 = radii[j]
                index = starting_index + j
                self._clash_radius2[index] = covrad1 + covrad2 + 0.4
                if bond1 in bonds or bond2 in bonds:
                    self._clash_mask[index] = False
        self._clash_radius2 *= self._clash_radius2
        self._clashing = np.zeros(self._ndistances, bool)
        self._dist2_matrix = np.empty(self._ndistances, float)

        # All atoms are active from the start
        self.active = np.ones(self.natoms, bool)
        self._active_mask = np.ones(self._ndistances, bool)

    def update_clash_mask(self):
        offset = self.natoms * (self.natoms - 1) // 2 - 1
        for i, active in enumerate(self.active[:-1]):
            natoms_left = self.natoms - i
            starting_index = offset - natoms_left * (natoms_left - 1) // 2 - i
            end = starting_index + self.natoms - i + 1
            self._active_mask[starting_index: end] = active

    def clashes(self):

        """Checks if there are any internal clashes.
        Deactivated atoms are not taken into account.
        """

        dm = self._dist2_matrix
        coor = self.coor
        dot = np.dot
        k = 0
        for i in range(self.natoms - 1):
            u = coor[i]
            for j in range(i + 1, self.natoms):
                u_v = u - coor[j]
                dm[k] = dot(u_v, u_v)
                k += 1
        np.less_equal(dm, self._clash_radius2, self._clashing)
        self._clashing &= self._clash_mask
        self._clashing &= self._active_mask
        nclashes = self._clashing.sum()
        return nclashes

    def get_chi(self, chi_index):
        atoms = self._rotamers['chi'][chi_index]
        selection = self.select('name', atoms)
        ordered_sel = []
        for atom in atoms:
            for sel in selection:
                if atom == self._name[sel]:
                    ordered_sel.append(sel)
                    break
        coor = self._coor[ordered_sel]
        angle = dihedral_angle(coor)
        return angle

    def set_chi(self, chi_index, value):
        atoms = self._rotamers['chi'][chi_index]
        selection = self.select('name', atoms)
        coor = self._coor[selection]
        origin = coor[1].copy()
        coor -= origin
        zaxis = coor[2]
        zaxis /= np.linalg.norm(zaxis)
        yaxis = coor[0] - np.inner(coor[0], zaxis) * zaxis
        yaxis /= np.linalg.norm(yaxis)
        xaxis = np.cross(yaxis, zaxis)
        backward = np.asmatrix(np.zeros((3, 3), float))
        backward[0] = xaxis
        backward[1] = yaxis
        backward[2] = zaxis
        forward = backward.T

        atoms_to_rotate = self._rotamers['chi-rotate'][chi_index]
        selection = self.select('name', atoms_to_rotate)
        coor_to_rotate = np.dot(self._coor[selection] - origin, backward.T)
        rotation = Rz(np.deg2rad(value - self.get_chi(chi_index)))

        R = forward * rotation
        self._coor[selection] = coor_to_rotate.dot(R.T) + origin


    def print_residue(self):
        for atom,coor,element,b,q in zip(self.name,self.coor,self.e,self.b,self.q):
            print("{} {} {} {} {}".format(atom,coor,element,b,q))

    def complete_residue(self):
        if residue_type(self) != "rotamer-residue":
            msg = "Error! Cannot complete non-aminoacid residue. Please, complete the missing atoms of the residue for qFiting!"
            raise(msg)

        first_atom=1
        atoms = self.name
        for atom, position in zip(self._rotamers['atoms'],self._rotamers['positions']):
            if atom in atoms:
                idx=np.argwhere(atoms == atom)[0]
                # Perform a superposition between the ideal residue coordinates in self._rotamers['positions']
                # and the ones observed on the PDB. This superposition is used to complete the missing atomsself.
                # TO DO: adapt this code to use a more robust superposition algorithm!
                if first_atom==1:
                    trans = np.array(position)
                    origin = self.coor[idx]
                    first_atom=0
                    trans_next = np.array(position)
                    origin_next = self.coor[idx]
                else:
                    trans = trans_next
                    origin = origin_next
                    # Calculate the rotation matrix using the Rodrigues Formula:
                    A = np.array(position) - trans # Vector of the ideal position
                    B = self.coor[idx] - origin # Vector of the observed position
                    n = np.cross(A,B) / np.linalg.norm(np.cross(A,B)) # The unit-orthogonal vector
                    nx = np.array(   [ [   0   ,-n[0,2], n[0,1]],   # Skew matrix based on the unit-orthogonal vector
                                       [ n[0,2],  0    ,-n[0,0]],
                                       [-n[0,1],n[0,0] ,  0   ]])
                    nx2 = np.matmul(nx,nx)
                    cos_theta = A.dot(B[0,:]) / ( np.linalg.norm(A) * np.linalg.norm(B) )
                    sin_theta = np.linalg.norm( np.cross(A,B)) / ( np.linalg.norm(A) * np.linalg.norm(B) )
                    rot = np.identity(3) + nx*sin_theta + (1-cos_theta)*nx2
                    trans_next = np.array(position)
                    origin_next = self.coor[idx]
            else:
                self.add_atom(atom,atom[0],origin+np.matmul(rot,(np.array(position)-trans).T))
                idx+=1

    def add_atom(self,name,element,coor):
        index = np.ndarray((1,),dtype='int')
        index[0,]=np.array(self.__dict__['_selection'][-1])
        for attr in ["record", "name", "b", "q", "coor", "resn", "resi","icode", "e", "charge", "chain", "altloc"]:
            if attr != "coor":
                self.__dict__['_'+attr]=np.insert(self.__dict__['_'+attr],index+1 , self.__dict__['_'+attr][index])
            else:
                self.__dict__['_'+attr]=np.reshape(np.insert(self.__dict__['_'+attr],3*index+3 , self.__dict__['_'+attr][index][0]),[-1,3])
        self.__dict__['_selection']=np.append(self.__dict__['_selection'],index+1)
        self.__dict__['_name'][index+1]=name
        self.__dict__['_coor'][index+1]=coor
        self.__dict__['_e'][index+1]=element
        self.__dict__['natoms']+=1
