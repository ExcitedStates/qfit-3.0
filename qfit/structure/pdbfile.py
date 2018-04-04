import gzip
from collections import defaultdict

import numpy as np

class PDBFile:

    @classmethod
    def read(cls, fname):
        cls.coor = defaultdict(list)
        cls.resolution = None
        if fname.endswith('.gz'):
            fopen = gzip.open
            mode = 'rt'
        else:
            fopen = open
            mode = 'r'

        with fopen(fname, mode) as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    values = CoorRecord.parse_line(line)
                    for field in CoorRecord.fields:
                        cls.coor[field].append(values[field])
                elif line.startswith('MODEL'):
                    raise NotImplementedError("MODEL record is not implemented.")
                elif line.startswith('REMARK   2 RESOLUTION'):
                    cls.resolution = float(line.split()[-2])
        return cls

    @staticmethod
    def write(fname, structure):
        with open(fname, 'w') as f:
            atomid = 1
            fields = list(CoorRecord.fields)
            del fields[1]
            #for fields in zip(*[getattr(structure, x) for x in CoorRecord.fields]):
            for field in zip(*[getattr(structure, x) for x in fields]):
                field = list(field)
                field.insert(1, atomid)
                if len(field[-2]) == 2 or len(field[2]) == 4:
                    f.write(CoorRecord.line2.format(*field))
                else:
                    f.write(CoorRecord.line1.format(*field))
                atomid += 1
            f.write(EndRecord.line)


class Record:

    @classmethod
    def parse_line(cls, line):
        values = {}
        for field, column, dtype in zip(cls.fields, cls.columns, cls.dtypes):
            values[field] = dtype(line[slice(*column)].strip())
        return values


class ModelRecord(Record):
    fields = 'record modelid'
    columns = [(0, 6), (11, 15)]
    dtypes = (str, int)
    line = '{:6s}' + ' ' * 5 + '{:6d}\n'


class CoorRecord(Record):
    fields = 'record atomid name altloc resn chain resi icode x y z q b e charge'.split()
    columns = [(0, 6), (6, 11), (12, 16), (16, 17), (17, 20), (21, 22),
               (22, 26), (26, 27), (30, 38), (38, 46), (46, 54), (54, 60),
               (60, 66), (76, 78), (78, 80),
               ]
    dtypes = (str, int, str, str, str, str, int, str, float, float, float,
              float, float, str, str)
    line1 = ('{:6s}{:5d}  {:3s}{:1s}{:3s} {:1s}{:4d}{:1s}   '
             '{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}' + ' ' * 10 + '{:>2s}{:>2s}\n')
    line2 = ('{:6s}{:5d} {:<4s}{:1s}{:3s} {:1s}{:4d}{:1s}   '
             '{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}' + ' ' * 10 + '{:>2s}{:2s}\n')


class AnisouRecord(Record):
    fields = 'record atomid atomname altloc resn chain resi icode u00 u11 u22 u01 u02 u12 e charge'.split()
    columns = [
        (0, 6), (6, 11), (12, 16), (16, 17), (17, 20), (21, 22), (22, 26),
        (26, 27), (28, 35), (35, 42), (42, 49), (49, 56), (56, 63), (63, 70), (76, 78), (78, 80),
    ]
    dtypes = (str, int, str, str, str, str, int, str, float, float, float,
              float, float, float, str, str)
    line1 = ('{:6s}{:5d}  {:3s}{:1s}{:3s} {:1s}{:4d}{:1s}   '
             '{:7d}' * 6 + ' ' * 7 + '{:>2s}{:>2s}\n')
    line2 = ('{:6s}{:5d} {:<4s}{:1s}{:3s} {:1s}{:4d}{:1s}   '
             '{:7d}' * 6 + ' ' * 7 + '{:>2s}{:>2s}\n')


class ExpdtaRecord(Record):
    fields = 'record cont technique'.split()
    columns = [(0,6), (8, 10), (10, 79)]
    dtypes= (str, str, str)


class RemarkRecord(Record):
    fields = 'record remarkid text'.split()
    columns = [(0, 6), (7, 10), (11, 79)]
    dtypes = (str, int, str)


class Remark2DiffractionRecord(Record):
    # For diffraction experiments
    fields = 'record remarkid RESOLUTION resolution ANGSTROM'.split()
    columns = [(0, 6), (9, 10), (11, 22), (23, 30), (31, 41)]
    dtypes = (str, str, str, float, str)


class Remark2NonDiffractionRecord(Record):
    # For diffraction experiments
    fields = 'record remarkid NOTAPPLICABLE'.split()
    columns = [(0, 6), (9, 10), (11, 38)]
    dtypes = (str, str, str)


class EndRecord(Record):
    fields = ['record']
    columns = [(0,6)]
    dtypes = (str,)
    line = 'END   ' + ' ' * 74 +'\n'
