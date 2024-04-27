import itertools as itl
import logging

from scitbx.array_family import flex
import iotbx.pdb.hierarchy
import numpy as np

from .base_structure import BaseMonomer
from .math import *
from .rotamers import ROTAMERS


logger = logging.getLogger(__name__)


_PROTEIN_BACKBONE_ATOMS = ["N", "CA", "C"]
_NUCLEOTIDE_BACKBONE_ATOMS = ["P", "O5'", "C5'", "C4'", "C3'", "O3"]
_SOLVENTS = ["HOH"]
_COVALENT_BOND_LENGTH = 1.5


def residue_type(residue) -> str:
    """Check residue type.

    The following types are recognized:
        rotamer-residue
            Amino acid for which rotamer information is available
        aa-residue
            Amino acid residue
        residue
            DNA / RNA residue
        solvent
            HOH
        ligand
            Other
    """
    # FIXME this is being called with objects that may represent multimers
    # RNA and DNA backbone atoms
    if residue.resn[0] in ROTAMERS:
        return "rotamer-residue"
    bb_atoms = _PROTEIN_BACKBONE_ATOMS
    if residue.select("name", bb_atoms).size == 3:
        return "aa-residue"
    xna_atoms = _NUCLEOTIDE_BACKBONE_ATOMS
    if residue.select("name", xna_atoms).size == 6:
        return "residue"
    if residue.resn[0] in _SOLVENTS:
        return "solvent"
    return "ligand"


class Residue(BaseMonomer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = (kwargs["resi"], kwargs["icode"])

    def __repr__(self):
        resi, icode = self.id
        string = "Residue: {}".format(resi)
        if self.id[1] != "":
            string += ":{}".format(icode)
        return string

    @property
    def identifier_tuple(self) -> tuple:
        """Returns (chain, resi, icode) to identify this residue."""
        chainid = self.chain[0]
        resi, icode = self.id
        return (chainid, resi, icode)

    @property
    def shortcode(self) -> str:
        (chainid, resi, icode) = self.identifier_tuple
        shortcode = f"{chainid}_{resi}"
        if icode:
            shortcode += f"_{icode}"
        return shortcode

    def reinitialize_object(self):
        """
        Return a fresh object of the same class with the same data array,
        after unpickling in a multiprocessing call.
        """
        return self.__class__(
            self._pdb_hierarchy,
            resi=self.id[0],
            icode=self.id[1],
            monomer_type=self.type,
            selection=self._selection,
            parent=self.parent,
        )

    def get_named_atom_selection(self, atom_names):
        """
        Given a list of atom names, return the corresponding selection
        as a flex.size_t array
        """
        atom_i_seqs = {self._atoms[i].name.strip():i for i in self._selection}
        sel = []
        for atom_name in atom_names:
            if atom_name in atom_i_seqs:
                sel.append(atom_i_seqs[atom_name])
            else:
                raise KeyError(f"Can't find atom named '{atom_name}' in residue {self}")
        return flex.size_t(sel)

    def _is_next_polymer_residue(self, other):
        bond_length = np.finfo(float).max
        if self.type in ("rotamer-residue", "aa-residue"):
            # Check for nearness
            sel = self.select("name", "C")
            C = self.get_xyz(sel)
            sel = other.select("name", "N")
            N = other.get_xyz(sel)
            bond_length = np.linalg.norm(N - C)
        elif self.type == "residue":
            # Check if RNA / DNA segment
            O3 = self.extract("name O3'")
            P = other.extract("name P")
            bond_length = np.linalg.norm(O3.coor[0] - P.coor[0])
        return bond_length < _COVALENT_BOND_LENGTH


class RotamerResidue(Residue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._residue_info = ROTAMERS[self.resname]
        self.nchi = self._residue_info["nchi"]
        self.rotamers = self._residue_info["rotamers"]
        self.nrotamers = len(self.rotamers)
        self._init_clash_detection()
        #self.i = 0

    def get_residue_info(self, key):
        return self._residue_info[key]

    # FIXME deprecated
    def get_rotamers(self, key):
        return self._residue_info[key]

    def get_missing_atoms(self):
        """Return a list of missing non-hydrogen atom names"""
        expected_atoms = np.array(self.get_residue_info("atoms"))
        missing_sel = np.isin(
            expected_atoms, test_elements=self.name, invert=True
        )
        return expected_atoms[missing_sel]

    def _init_clash_detection(self, scaling_factor=0.75, covalent_bonds=None):
        # Setup the condensed distance based arrays for clash detection and fill them
        self._ndistances = self.natoms * (self.natoms - 1) // 2
        self._clash_mask = np.ones(self._ndistances, bool)
        self._clash_radius2 = np.zeros(self._ndistances, float)
        radii = self.covalent_radius
        bonds = self._residue_info["bonds"]
        if covalent_bonds:
            for bond in covalent_bonds:
                bonds.append(bond)
        offset = self.natoms * (self.natoms - 1) // 2
        for i in range(self.natoms - 1):
            # starting_index = int(offset - sp_comb(self.natoms - i, 2)) - i - 1
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
        self._clash_radius2 *= scaling_factor
        self._clashing = np.zeros(self._ndistances, bool)
        self._dist2_matrix = np.empty(self._ndistances, float)

        # All atoms are active from the start
        self.set_active()
        self._active_mask = np.ones(self._ndistances, bool)

    def update_clash_mask(self):
        offset = self.natoms * (self.natoms - 1) // 2 - 1
        for i, active in enumerate(self.active[:-1]):
            natoms_left = self.natoms - i
            starting_index = offset - natoms_left * (natoms_left - 1) // 2 - i
            end = starting_index + self.natoms - i + 1
            self._active_mask[starting_index:end] = active

    def clashes(self) -> int:
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

    def get_chi(self, chi_index) -> float:
        atoms = self._residue_info["chi"][chi_index]
        selection = self.select("name", atoms)
        ordered_sel = []
        for atom in atoms:
            for i_seq in selection:
                if atom == self._atoms[int(i_seq)].name.strip():
                    ordered_sel.append(i_seq)
                    break
        coor = self._atoms.extract_xyz().as_numpy_array()[ordered_sel]
        angle = dihedral_angle(coor)
        return angle

    def set_chi(self, chi_index, value, covalent=None, length=None):
        atoms = self._residue_info["chi"][chi_index]
        selection = self.select("name", atoms)

        # Translate coordinates to center on coor[1]
        coor = self._atoms.extract_xyz().as_numpy_array()[selection]
        origin = coor[1].copy()
        coor -= origin

        # Make an orthogonal axis system based on 3 atoms
        backward = gram_schmidt_orthonormal_zx(coor)
        forward = backward.T

        # Complete selection to be rotated
        atoms_to_rotate = self._residue_info["chi-rotate"][chi_index]
        selection = self.select("name", atoms_to_rotate)
        if covalent in atoms_to_rotate:
            # If we are rotating the atom that is covalently bonded
            # to the ligand, we should also rotate the ligand.
            atoms_to_rotate2 = self.name[length:]
            selection2 = self.select("name", atoms_to_rotate2)
            selection = np.array(list(selection) + list(selection2), dtype=int)

        # Create transformation matrix
        angle = np.deg2rad(value - self.get_chi(chi_index))
        rotation = Rz(angle)
        R = forward @ rotation @ backward

        # Apply transformation
        coor_to_rotate = self._atoms.extract_xyz().as_numpy_array()[selection]
        coor_to_rotate -= origin
        coor_to_rotate = np.dot(coor_to_rotate, R.T)
        coor_to_rotate += origin
        self.set_xyz(coor_to_rotate, selection)

    def complete_residue(self):
        if residue_type(self) != "rotamer-residue":
            msg = (
                f"Cannot complete non-aminoacid residue {self}. "
                f"Please complete the missing atoms of the residue before "
                f"running qFit again!"
            )
            raise RuntimeError(msg)

        for atom in self._residue_info["atoms"]:
            # Found a missing atom!
            if atom not in self.name:
                self._complete_residue_recursive(atom)

    def _complete_residue_recursive(self, next_atom_name):
        if next_atom_name in ["N", "C", "CA", "O"]:
            msg = (
                f"{self} is missing backbone atom {next_atom_name}. "
                f"qFit cannot complete missing backbone atoms. "
                f"Please complete the missing backbone atoms of "
                f"the residue before running qFit again!"
            )
            raise RuntimeError(msg)

        ref_atom = self._residue_info["connectivity"][next_atom_name][0]
        if ref_atom not in self.name:
            print('ref atom:')
            print(ref_atom)
            self.complete_residue_recursive(ref_atom)
        print(self.name)
        idx = np.argwhere(self.name == ref_atom)[0]
        ref_coor = self.coor[idx]
        bond_length, bond_length_sd = self._residue_info["bond_dist"][ref_atom][next_atom_name]
        # Identify a suitable atom for the bond angle:
        for angle in self._residue_info["bond_angle"]:
            if angle[0][1] == ref_atom and angle[0][2] == next_atom_name:
                if angle[0][0][0] == "H":
                    continue
                bond_angle_atom = angle[0][0]
                bond_angle, bond_angle_sd = angle[1]
                if bond_angle_atom not in self.name:
                    self._complete_residue_recursive(bond_angle_atom)
                bond_angle_coor = self.coor[
                    np.argwhere(self.name == bond_angle_atom)[0]
                ]
                dihedral_atom = None
                # If the atom's position is dependent on a rotamer,
                # identify the fourth dihedral angle atom:
                for i, chi in self._residue_info["chi"].items():
                    if (
                        chi[1] == bond_angle_atom
                        and chi[2] == ref_atom
                        and chi[3] == next_atom_name
                    ):
                        dihedral_atom = chi[0]
                        dihed_angle = self._residue_info["rotamers"][0][i - 1]

                # If the atom's position is not dependent on a rotamer,
                # identify the fourth dihedral angle atom:
                if dihedral_atom is None:
                    for dihedral in self._residue_info["dihedral"]:
                        if (
                            dihedral[0][1] == bond_angle_atom
                            and dihedral[0][2] == ref_atom
                            and dihedral[0][3] == next_atom_name
                        ):
                            dihedral_atom = dihedral[0][0]
                            if dihedral[1][0] in self._residue_info["atoms"]:
                                other_dihedral_atom = dihedral[1][0]
                                if dihedral_atom not in self.name:
                                    self._complete_residue_recursive(dihedral_atom)
                                dihedral_atom_coor = self.coor[
                                    np.argwhere(self.name == dihedral_atom)[0]
                                ]
                                if other_dihedral_atom not in self.name:
                                    self._complete_residue_recursive(other_dihedral_atom)
                                other_dihedral_atom_coor = self.coor[
                                    np.argwhere(self.name == other_dihedral_atom)[0]
                                ]
                                try:
                                    dihed_angle = dihedral[1][1]
                                    dihed_angle += dihedral_angle(
                                        [
                                            dihedral_atom_coor[0],
                                            bond_angle_coor[0],
                                            ref_coor[0],
                                            other_dihedral_atom_coor[0],
                                        ]
                                    )
                                    if dihed_angle > 180:
                                        dihed_angle -= 360  # wrap to (-180, 180]
                                except:
                                    dihed_angle = 180
                                    dihed_angle += dihedral_angle(
                                        [
                                            dihedral_atom_coor[0],
                                            bond_angle_coor[0],
                                            ref_coor[0],
                                            other_dihedral_atom_coor[0],
                                        ]
                                    )
                                    if dihed_angle > 180:
                                        dihed_angle -= 360  # wrap to (-180, 180]
                            else:
                                dihed_angle = dihedral[1][0]
                            break
                if dihedral_atom is not None:
                    if dihedral_atom not in self.name:
                        self._complete_residue_recursive(dihedral_atom)
                    dihedral_atom_coor = self.coor[
                        np.argwhere(self.name == dihedral_atom)[0]
                    ]
                    break

        logger.debug(
            f"Rebuilding {next_atom_name}:\n"
            f"  {dihedral_atom}@{dihedral_atom_coor.flatten()}\n"
            f"  {bond_angle_atom}@{bond_angle_coor.flatten()}\n"
            f"  {ref_atom}@{ref_coor.flatten()}\n"
            f"  length:{bond_length}±{bond_length_sd}\n"
            f"  angle:{bond_angle}±{bond_angle_sd}\n"
            f"  dihedral_angle:{dihed_angle}"
        )
        try:
            new_coor = self.calc_coordinates(
                dihedral_atom_coor.flatten(),
                bond_angle_coor.flatten(),
                ref_coor.flatten(),
                bond_length,
                bond_length_sd,
                np.deg2rad(bond_angle),
                np.deg2rad(bond_angle_sd),
                np.deg2rad(dihed_angle),
            )
            new_coor = [round(x, 3) for x in new_coor]
        except RuntimeError as e:
            raise RuntimeError(f"Unable to rebuild atom {next_atom_name}.") from e
        else:
            logger.info(f"Rebuilt {next_atom_name} at {new_coor}")
        self._add_atom(next_atom_name, next_atom_name[0], new_coor)

    def _add_atom(self, name, element, coor):
        """
        Add a new atom to the residue; internally, this means appending
        to the parent iotbx.pdb.hierarchy.atom_group object and updating
        the internal arrays.
        """
        last_atom = self.atoms[-1]
        atom = iotbx.pdb.hierarchy.atom()
        atom.name = name if len(name) == 4 else f" {name}"
        atom.element = element
        atom.xyz = tuple(coor)
        atom.occ = last_atom.occ
        atom.b = last_atom.b
        last_atom.parent().append_atom(atom)
        # XXX what about the parent structure?  does it matter?
        self._atoms = self._pdb_hierarchy.atoms()
        self._atoms.reset_i_seq()
        self._atoms.reset_serial()
        self._selection.append(atom.i_seq)
        self.natoms += 1
        self.total_length += 1
        # XXX not sure about this...
        self._active_flag = np.concatenate((self._active_flag, [False]))
        self.active = True

    def get_rebuilt_structure(self):
        return self.parent.parent.parent.__class__(
            self._pdb_hierarchy,
            **self._kwargs)

    @staticmethod
    def calc_coordinates(i, j, k, L, sig_L, theta, sig_theta, chi):
        """Calculate coords of an atom from three atomic positions and bond parms.

        Will permit deviations in bond length or theta to position an atom.

        TODO: Use a solver to minimise error in both L and theta.

        Args:
            i (np.ndarray[float, shape=(3,)]): coords of atom 3-bonds away
            j (np.ndarray[float, shape=(3,)]): coords of atom 2-bonds away
            k (np.ndarray[float, shape=(3,)]): coords of neighbouring atom
            L (float): bond length
            sig_L (float): standard deviation of bond length
            theta (float): bond angle (in radians)
            sig_theta (float): standard deviation of bond angle (in radians)
            chi (float): dihedral angle (in radians)

        Returns:
            np.ndarray[float, shape=(3,): coords of atom
        """
        print(i, j, k, L, sig_L, theta, sig_theta, chi)
        # We will try these parameters
        step_size = sig_theta / 5
        theta_options_larger = np.linspace(theta, theta + sig_theta, 5, endpoint=False) + step_size
        theta_options_smaller = np.linspace(theta, theta - sig_theta, 5, endpoint=False) - step_size
        theta_options = [
            theta,
            *np.ravel(list(zip(theta_options_larger, theta_options_smaller))),
        ]

        L_step_size = sig_L / 5
        L_options_larger = np.linspace(L, L + sig_L, 5, endpoint=False) + L_step_size
        L_options_smaller = np.linspace(L, L - sig_L, 5, endpoint=False) - L_step_size
        L_options = [L, *np.ravel(list(zip(L_options_larger, L_options_smaller)))]

        tries = itl.product(theta_options, L_options)

        # Loop over parameters, and return the first success.
        for try_theta, try_L in tries:
            try:
                coordinates = RotamerResidue.position_from_bond_parms(
                    i, j, k, try_L, try_theta, chi
                )
            except ValueError:
                # If these parameters didn't work, try the next set.
                continue
            else:
                return coordinates

        # If we get here, we can't rebuild the atom according to our parameters.
        raise RuntimeError(
            f"Could not determine position. "
            f"Exhausted L ∈ [{L - sig_L:.2f}, {L + sig_L:.2f}] and "
            f"theta ∈ [{theta - sig_theta:.2f}, {theta + sig_theta:.2f}]"
        )

    @staticmethod
    def position_from_bond_parms(i, j, k, L, theta, chi):
        """Calculate coords of a 4th atom from 3 atomic coords and bond parms.

        Args:
            i (np.ndarray[float, shape=(3,)]): coords of atom 3-bonds away
            j (np.ndarray[float, shape=(3,)]): coords of atom 2-bonds away
            k (np.ndarray[float, shape=(3,)]): coords of neighbouring atom
            L (float): bond length
            theta (float): bond angle (in radians)
            chi (float): dihedral angle (in radians)

        Returns:
            np.ndarray[float, shape=(3,): coords of atom
        """

        # First, calculate distance vectors u = vec(ji); v = vec(kj)
        u = i - j
        v = j - k

        # Our task here is to find x, the distance vector < a_x, b_x, c_x >
        #     from k to our new atom.

        # Equation 1: plane (derived from θ)
        # ==================================
        #   x . v
        # ---------   = cosθ
        #  ‖x‖ ‖v‖
        #       x . v = ‖v‖ L cosθ
        #
        #     Let  C1 = ‖v‖ L cosθ
        #
        #       x . v = C1

        # If v is normalized
        norm_v = v / np.linalg.norm(v)
        norm_C1 = L * np.cos(theta)

        # Equation 2: plane (derived from χ)
        # ==================================
        # https://en.wikipedia.org/wiki/Dihedral_angle#Mathematical_background
        #        v . (( x × v ) × ( u × v ))
        #       ----------------------------- = sinχ
        #         ‖ v ‖ ‖ u × v ‖ ‖ x × v ‖
        #
        #     But, ‖ x × v ‖ = ‖x‖ ‖v‖ sinθ = ‖v‖ L sinθ
        #         v . (( x × v ) × ( u × v )) = ‖ u × v ‖ ‖v‖^2 L sinθ sinχ
        #     Flipping the cross products
        #         v . (( v × x ) × ( v × u )) = ‖ u × v ‖ ‖v‖^2 L sinθ sinχ
        #     Factoring a scalar triple product out of the vector product
        #              v . (( v . ( x × u ))v = ‖ u × v ‖ ‖v‖^2 L sinθ sinχ
        #     Noting that this is simply ( v . f v ), which is f * ‖v‖^2
        #           (( v . ( x × u )) * ‖v‖^2 = ‖ u × v ‖ ‖v‖^2 L sinθ sinχ
        #     Rotating the scalar triple product, cancelling ‖v‖^2
        #                       x . ( u × v ) = ‖ u × v ‖ L sinθ sinχ
        #
        #     Let                           w = u × v
        #                                  C2 = ‖ u × v ‖ L sinθ sinχ
        #
        #                               x . w = C2

        w = np.cross(u, v)

        # Normalizing this vector
        norm_w = w / np.linalg.norm(w)
        norm_C2 = L * np.sin(theta) * np.sin(chi)

        # Intersection of planes to construct a line
        # ==========================================
        # https://en.wikipedia.org/wiki/Plane_(geometry)#Line_of_intersection_between_two_planes

        # The cross product of the normals to the two planes will be a line
        #     direction, co-linear to the line of intersection.

        # The line of intersection between two planes
        #     Π_1: n1 . r = h1
        #     Π_2: n2 . r = h2
        #     where n_i are normalized can be written as:
        #
        #     r = k(n1 × n2) + r0
        #
        #     If n1 and n2 are orthonormal, r0 is
        #           r0 = n1 h1 + n2 h2

        # Since w = u × v, v ⟂ w, and
        r0 = norm_v * norm_C1 + norm_w * norm_C2

        # Calculate a unit vector along the line from v × w
        unit_vw = np.cross(v, w) / np.linalg.norm(np.cross(v, w))

        # Equation 3: sphere (derived from L)
        # ==================================
        # a_x^2 + b_x^2 + c_x^2 - L^2 = 0

        # Intersection of sphere with line to determine points
        # ====================================================
        # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection

        # Here, we have our sphere:
        #     ‖x‖^2 = L^2
        # where x are the points on the sphere.

        # We also have a line:
        #     x = k l + r0
        # where x are points on the line,
        #       k is a distance along the line
        #       l is a unit vector along the line
        #       r0 is the origin of the line.

        # Combining these two equations
        #     L^2 = ‖ r0 + k l ‖^2
        #         = (r0 + k l) . (r0 + k l)
        #       0 = k^2 (l . l)  +  k (2 r0 . l)  +  (r0 . r0 - L^2)
        # which is a quadratic in k.
        # l is a unit vector, so:
        #       0 = k^2 (1)  +  k (2 r0 . l)  +  (r0 . r0 - L^2)

        b = 2 * np.dot(unit_vw, r0)
        c = np.dot(r0, r0) - L**2

        discriminant = b**2 - 4 * c
        if discriminant < 0:
            logger.debug(
                f"Bond parameters too restrictive!\n"
                f"  u, v: {u, v}\n"
                f"  r0: {r0}\n"
                f"  unit_vw: {unit_vw}\n"
                f"  b, c: {b, c}\n"
                f"  discriminant: {discriminant:.2e}"
            )
            raise ValueError(
                f"Could not determine position to rebuild atom. "
                f"Discriminant: {discriminant:.2e}"
            )

        # Solve the quadratic, and determine 2 potential atom positions.
        # Resolve ambiguity of chi with 2-arg arctan.
        # Translate correct coordinate (it was calc'd relative to k).
        positions = r0 + unit_vw * np.roots([1.0, b, c])[:, np.newaxis]
        calc_chis = [
            np.arctan2(
                np.dot(v, np.cross(np.cross(pos, v), np.cross(u, v))),
                np.linalg.norm(v) * np.dot(np.cross(u, v), np.cross(pos, v)),
            )
            for pos in positions
        ]
        correct_chi = np.isclose(calc_chis, chi)

        # We have calculated atom positions. If there are no correct_chi, this
        # seems to occur from rounding errors at the edges of (-π, π].
        if correct_chi.sum() == 0:
            logger.debug(
                f"No valid chi results:\n"
                f"  chi / calc_chis: {chi}/({calc_chis})\n"
                f"  correct_chi: {correct_chi}"
            )
            raise ValueError(f"Couldn't determine a matching chi.")

        # if discriminant~0, this has identical sols. Take the first.
        x = positions[correct_chi][0]
        x += k
        return x
