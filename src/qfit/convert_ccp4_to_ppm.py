import argparse
import logging
import os
import sys
import time
from string import ascii_uppercase

logger = logging.getLogger(__name__)
from math import sqrt
import numpy as np

from . import (
    Structure,
    XMap,
    Transformer,
    ElectronDensityRadiusTable,
    ResolutionBins,
    BondLengthTable,
)


class converterOptions:
    def __init__(self):
        # General options
        self.directory = "."
        self.debug = False

        # Density creation options
        self.map_type = None
        self.resolution = None
        self.resolution_min = None
        self.scattering = "xray"

    def apply_command_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class Weight:
    def __init__(self, radius):
        # Per definition:
        self.b1 = 1.0  # (maximum of first parabola)
        self.b2 = -0.4  # (minimum of second parabola)
        self.b3 = 0.0  # (maximum of third parabola)
        self.c1 = 0.0  # (we want the first parabola to have its maximum at x=0)
        self.m1 = -1.0 / (
            radius**2
        )  # This ensures that the density becomes superfluous if p is in d(a)
        self.c3 = (
            2 * radius
        )  # (we want the third parabola to have its maximum at x=2*r)
        self.r0 = 1.0822 * radius  # The point where P1 and P2 intersect (pre-defined)
        self.r2 = 2 * radius  # The point where P3 becomes 0.

        # Calculate unknowns:
        # Unknowns: r1,m2,m3,c2
        self.c2 = -(self.b1 - self.b2) / (self.m1 * self.r0)
        self.m2 = (
            (self.r0**2)
            * (self.m1**2)
            / ((self.r0**2) * self.m1 - self.b2 + self.b1)
        )
        self.r1 = (
            self.m2 * self.c2 * self.c3 - self.m2 * self.c2 * self.c2 - self.b2
        ) / (self.m2 * self.c3 - self.m2 * self.c2)
        self.m3 = self.m2 * (self.r1 - self.c2) / (self.r1 - self.c3)

        self.P = lambda x, m, c, b: m * (x - c) ** 2 + b

    def __call__(self, dist):
        # Calculate the weight:
        if dist < self.r0:
            return self.P(dist, self.m1, self.c1, self.b1)
        elif dist < self.r1:
            return self.P(dist, self.m2, self.c2, self.b2)
        elif dist < self.r2:
            return self.P(dist, self.m3, self.c3, self.b3)
        else:
            return 0.0


class Point:
    def __init__(self, coor):
        self.coor = coor
        self.Green = 0.0
        self.Blue = 0.0
        self.visited = 0

    def set_Point(self, new_point):
        self.coor = new_point.coor
        self.Green = new_point.Green
        self.Blue = new_point.Blue
        self.visited = new_point.visited


class _BaseConvertMap:
    def __init__(self, structure, xmap, options):
        self.structure = structure
        self.xmap = xmap
        self.options = options
        self._coor_set = [self.structure.coor]
        self._voxel_volume = self.xmap.unit_cell.calc_volume() / self.xmap.array.size
        # Calculate space diagonal and the partitioning factor p
        abc = np.asarray(
            [self.xmap.unit_cell.a, self.xmap.unit_cell.b, self.xmap.unit_cell.c]
        )
        self.grid_to_cartesian = np.transpose(
            (self.xmap.unit_cell.frac_to_orth / abc) * self.xmap.voxelspacing
        )
        self.cartesian_to_grid = np.linalg.inv(self.grid_to_cartesian)
        self.Grid = np.zeros_like(xmap.array, dtype=object)
        self.mean = xmap.array.mean()
        self.sigma = xmap.array.std()
        # self.Populate_Grid(self.residue)

    def Populate_Grid(self):
        for chain in self.structure:
            for residue in chain:
                for ind in range(len(residue.name)):
                    atom, element, charge, coor, icode, record, occ, resi = (
                        residue.name[ind],
                        residue.e[ind],
                        residue.charge[ind],
                        residue.coor[ind],
                        residue.icode[ind],
                        residue.record[ind],
                        residue.q[ind],
                        residue.resi[ind],
                    )

                    grid = np.dot(coor, self.cartesian_to_grid).astype(
                        int
                    ) - np.asarray(
                        self.xmap.offset
                    )  # (i,j,k)
                    if element == "H":
                        continue
                    if charge == "":
                        ed_radius = self.calculate_density_radius(
                            element, self.options.resolution, int(residue.b[ind])
                        )
                    else:
                        ed_radius = self.calculate_density_radius(
                            element,
                            self.options.resolution,
                            int(residue.b[ind]),
                            charge,
                        )
                    box = (np.ceil(ed_radius * 2 / self.xmap.voxelspacing)).astype(int)

                    # Iterate over all grid points in the box and calculate their ownership
                    for i in range(grid[2] - box[2], grid[2] + box[2]):
                        for j in range(grid[1] - box[1], grid[1] + box[1]):
                            for k in range(grid[0] - box[0], grid[0] + box[0]):
                                try:
                                    dist = np.linalg.norm(
                                        coor - self.Grid[i][j][k].coor
                                    )
                                except:
                                    self.Grid[i][j][k] = Point(
                                        np.dot(
                                            np.asarray([k, j, i])
                                            + np.asarray(self.xmap.offset),
                                            self.grid_to_cartesian,
                                        )
                                    )
                                    dist = np.linalg.norm(
                                        coor - self.Grid[i][j][k].coor
                                    )
                                if dist < ed_radius:
                                    self.Grid[i][j][k].Green = dist
                                    self.Grid[i][j][k].Blue = residue.b[ind]
                                    self.Grid[i][j][k].visited = 1
                                else:
                                    if self.Grid[i][j][k].visited == 0:
                                        self.Grid[i][j][k] = 3.0

    # Calculates the atomic radius based on the table
    def calculate_density_radius(self, atom, resolution, bfactor, charge="0"):
        a = int(np.floor(resolution / 0.5) - 1)
        b = int(np.ceil(resolution / 0.5) - 1)

        if atom not in ElectronDensityRadiusTable.keys():
            atom = atom[0] + atom[1:].lower()
        if charge not in ElectronDensityRadiusTable[atom].keys():
            charge = charge[::-1]
        if a == b:
            radius = ElectronDensityRadiusTable[atom][charge][a]
        else:
            radius = ElectronDensityRadiusTable[atom][charge][a] + (
                ElectronDensityRadiusTable[atom][charge][b]
                - ElectronDensityRadiusTable[atom][charge][a]
            ) * (resolution - ResolutionBins[a]) / (
                ResolutionBins[b] - ResolutionBins[a]
            )
        return np.asarray(radius)

    def print_ppm(self):
        self.Populate_Grid()
        print("P3")
        dim = int(sqrt(len(self.xmap.array)))
        print(
            "{} {}".format(
                len(self.xmap.array[0][0]) * dim, len(self.xmap.array[0]) * dim
            )
        )
        image = np.zeros(
            (len(self.xmap.array[0]) * dim, len(self.xmap.array[0][0]) * dim)
        )
        green = np.zeros(
            (len(self.xmap.array[0]) * dim, len(self.xmap.array[0][0]) * dim)
        )
        blue = np.zeros(
            (len(self.xmap.array[0]) * dim, len(self.xmap.array[0][0]) * dim)
        )
        print("255")
        column = 0
        row = 0
        for i in range(0, len(self.xmap.array)):
            for j in range(0, len(self.xmap.array[i])):
                for k in range(0, len(self.xmap.array[i][j])):
                    red = max(self.xmap.array[i][j][k], 0.0)
                    red = int((red * 255) / (3 * self.sigma))
                    red = min(red, 255)
                    image[row + j][column + k] = 255 - red
                    try:
                        green[row + j][column + k] = min(
                            255, 255 * self.Grid[i][j][k].Green / 3.0
                        )
                    except:
                        green[row + j][column + k] = 255
                    try:
                        blue[row + j][column + k] = 255 - min(
                            255, self.Grid[i][j][k].Blue * 255 / 100
                        )
                    except:
                        blue[row + j][column + k] = 255
            column += len(self.xmap.array[0][0])
            if column >= dim * len(self.xmap.array[0][0]):
                column = 0
                row += len(self.xmap.array[0])
                if row >= dim * len(self.xmap.array[0]):
                    break
        for i in range(len(image)):
            for j in range(len(image[i])):
                print(
                    "{0:d} {1:d} {2:d} ".format(
                        int(image[i][j]), int(green[i][j]), int(blue[i][j])
                    ),
                    end="",
                )
            print()
            # if(self.xmap.array[i][j][k] - self.mean >contour*self.sigma):
            #    coor = np.dot(np.asarray([k,j,i])+np.asarray(self.xmap.offset),self.grid_to_cartesian)
            #    print("HETATM {0:4d}  H   HOH A {0:3d}    {1:8.3f}{2:8.3f}{3:8.3f}  1.00 37.00           H".format(1,coor[0],coor[1],coor[2]))

    def print_stats(self):
        # Note that values of the offset are based on C,R,S - these are not always ordered like x,y,z
        offset = self.xmap.offset
        voxelspacing = self.xmap.voxelspacing  # These ARE ordered (x,y,z)
        print(
            "Unit cell shape:", self.xmap.unit_cell.shape
        )  # These are ordered (z,y,x)
        print(
            "Unit cell a,b,c: {0:.2f} {1:.2f} {2:.2f}".format(
                self.xmap.unit_cell.a, self.xmap.unit_cell.b, self.xmap.unit_cell.c
            )
        )  # These ARE ordered (x,y,z)
        print(
            "Unit cell alpha,beta,gamma: {0:.2f} {1:.2f} {2:.2f}".format(
                self.xmap.unit_cell.alpha,
                self.xmap.unit_cell.beta,
                self.xmap.unit_cell.gamma,
            )
        )
        print(
            "XMap array dimentions: ",
            [len(self.xmap.array), len(self.xmap.array[0]), len(self.xmap.array[0][0])],
        )  # These are ordered (z,y,x)
        abc = np.asarray(
            [self.xmap.unit_cell.a, self.xmap.unit_cell.b, self.xmap.unit_cell.c]
        )
        print("abc/voxelspacing:", abc / self.xmap.voxelspacing)
        print("Offset: ", offset)


class ConvertMap(_BaseConvertMap):
    def __init__(self, structure, xmap, options):
        super().__init__(structure, xmap, options)

    def __call__(self):
        self.print_ppm()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xmap", type=str, help="X-ray density map in CCP4 format.")
    p.add_argument("structure", type=str, help="PDB-file containing structure.")
    p.add_argument("resolution", type=float, help="Map resolution in angstrom.")
    p.add_argument(
        "-d",
        "--directory",
        type=os.path.abspath,
        default=".",
        metavar="<dir>",
        help="Directory to store results.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Be verbose.")
    return p.parse_args()


""" Main function """


def main():
    args = parse_args()
    """ Create the output directory provided by the user: """
    os.makedirs(args.directory, exist_ok=True)

    time0 = time.time()  # Useful variable for profiling run times.
    """ Processing input structure and map """
    # Read structure in:
    structure = Structure.fromfile(args.structure)
    # This line would ensure that we only select the '' altlocs or the 'A' altlocs.
    structure = structure.extract("altloc", ("", "A", "B", "C", "D", "E"))
    # Prepare X-ray map
    xmap = XMap.fromfile(args.xmap)

    options = converterOptions()
    options.apply_command_args(args)

    converter = ConvertMap(structure, xmap, options)
    converter()
    """ Profiling run time: """
    passed = time.time() - time0


#    print(f"Time passed: {passed}s")
