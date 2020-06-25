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

import gzip
from collections import defaultdict
import sys
import numpy as np

class PDBFile:

    @classmethod
    def read(cls, fname):
        cls.coor = defaultdict(list)
        cls.anisou = defaultdict(list)
        cls.link = defaultdict(list)
        cls.cryst1 = {}
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
                elif line.startswith('ANISOU'):
                    values = AnisouRecord.parse_line(line)
                    for field in AnisouRecord.fields:
                        cls.anisou[field].append(values[field])
                elif line.startswith('MODEL'):
                    raise NotImplementedError("MODEL record is not implemented.")
                elif line.startswith('REMARK   2 RESOLUTION'):
                    try:
                        values = Remark2DiffractionRecord.parse_line(line)
                        cls.resolution = values['resolution']
                    except:
                        pass
                elif line.startswith('LINK '):
                    try:
                        values = LinkRecord.parse_line(line)
                        for field in LinkRecord.fields:
                            cls.link[field].append(values[field])
                    except:
                        sys.stderr.write("Error parsing LINK data.\n")
                        pass
                elif line.startswith('CRYST1'):
                    cls.cryst1 = Cryst1Record.parse_line(line)
        return cls

    @staticmethod
    def write(fname, structure):
        with open(fname, 'w') as f:
            atomid = 1
            if structure.link_data:
                fields = list(LinkRecord.fields)
                for field in zip(*[structure.link_data[x] for x in fields]):
                    field = list(field)
                    if not field[-1]:
                        field[-1] = ''
                        f.write(LinkRecord.line2.format(*field))
                    else:
                        f.write(LinkRecord.line.format(*field))
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
            try:
                values[field] = dtype(line[slice(*column)].strip())
            except ValueError:
                values[field] = None
        return values


class ModelRecord(Record):
    # http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#MODEL
    fields  = ("record", "modelid")
    columns = [(0, 6),   (10, 14)]
    dtypes  = (str,      int)
    fmtstr  = '{:<6s}' + ' ' * 4 + '{:>4d}' + '\n'


class LinkRecord(Record):
    fields = ('record name1 altloc1 resn1 chain1 resi1 icode1 name2 '
              'altloc2 resn2 chain2 resi2 icode2 sym1 sym2 length').split()
    columns = [(0, 6), (12, 16), (16, 17), (17, 20), (21, 22),
               (22, 26), (26, 27), (42, 46), (46, 47), (47, 50),
               (51, 52), (52, 56), (56, 57), (59, 65), (66, 72),
               (73, 78),
               ]
    dtypes = (str, str, str, str, str, int, str,
                   str, str, str, str, int, str,
                   str, str, float)
    line = ('{:6s}' + ' ' * 7 + '{:3s}{:1s}{:3s} '
            '{:1s}{:4d}{:1s}' + ' ' * 16 + '{:3s}{:1s}{:3s} '
            '{:1s}{:4d}{:1s} {:6s} {:6s} {:5.2f}\n')
    line2 = ('{:6s}' + ' ' * 7 + '{:3s}{:1s}{:3s} '
            '{:1s}{:4d}{:1s}' + ' ' * 16 + '{:3s}{:1s}{:3s} '
            '{:1s}{:4d}{:1s} {:6s} {:6s} {:5s}\n')

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

class Cryst1Record(Record):
    fields = 'record a b c alpha beta gamma spg'.split()
    columns = [(0,6), (6, 15), (15, 24), (24, 33), (33, 40), (40, 47), (47, 54), (55, 66), (66, 70)]
    dtypes = (str, float, float, float, float, float, float, str, int)

class EndRecord(Record):
    fields = ['record']
    columns = [(0,6)]
    dtypes = (str,)
    line = 'END   ' + ' ' * 74 +'\n'
