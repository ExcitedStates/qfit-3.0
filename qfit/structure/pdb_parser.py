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
from itertools import izip
from collections import OrderedDict

class PDBFile(object):

    """
    PDB file parser.  It considers the PDB file as a tree:
        structure -> model -> chain -> resi -> atom
    """

    TITLE_SECTION = tuple((
        'HEADER OBSLTE TITLE SPLT CAVEAT COMPND SOURCE KEYWDS EXPDTA NUMMDL MDLTYP '
        'AUTHOR REVDAT SPRSDE JRNL REMARK'
    ).split())
    PRIMARY_STRUCTURE_SECTION = ('DBREF ', 'SEQADV', 'DBREF1', 'DBREF2', 'SEQRES')
    HETEROGEN_SECTION = ('HET   ', 'HETNAM', 'FORMUL')
    SECONDARY_STRUCTURE_SECTION = ('HELIX ', 'SHEET ')
    CONNECTIVITY_ANNOTATION_SECTION = ('SSBOND', 'LINK  ')
    MISCELLANEOUS_FEATURES_SECTION = ('SITE  ')
    CRYSTALLOGRAPHIC_SECTION = tuple('CRYST1 ORIGX1 ORIGX2 ORIGX3 SCALE1 SCALE2 SCALE3 MTRIX1 MTRIX2 MTRIX3'.split())
    COORDINATE_SECTION = ('MODEL ', 'ATOM  ', 'ANISOU', 'TER   ', 'HETATM', 'ENDMDL')
    CONNECTIVITY_SECTION = ('CONECT')
    BOOKKEEPING_SECTION = ('MASTER', 'END   ')

    def __init__(self, fname):
        self.fname = fname
        self._open = False

    def open(self):
        if self.fname[-3:] == '.gz':
            fopen = gzip.open
            mode = 'rb'
        else:
            fopen = open
            mode = 'r'
        self._f = fopen(self.fname, mode)
        self._open = True

    def close(self):
        if self._open:
            self._f.close()

    def read(self):

        pdb = {}
        f = self._f
        line = f.readline()

        remarkid = None
        while line.startswith(self.TITLE_SECTION):
            if line.startswith('EXPDTA'):
                values = ExpdtaRecord.parse_line(line)
                if values['cont']:
                    pdb['expdata'] += values['technique']
                else:
                    pdb['expdata'] = values['technique']
            elif line.startswith('REMARK'):
                values = RemarkRecord.parse_line(line)
                remarkid = values['remarkid']
                if values['text'].isspace():
                    continue
                if remarkid == 2:
                    try:
                        values = Remark2DiffractionRecord.parse_line(line)
                        pdb['resolution'] = values['resolution']
                    except ValueError:
                        values = Remark2NonDiffractionRecord.parse_line(line)
                        pdb['resolution'] = float('nan')
            line = f.readline()

        while not line.startswith(self.COORDINATE_SECTION):
            line = f.readline()

        pdb['structure'] = OrderedDict()
        structure = pdb['structure']
        modelid = 1
        structure[modelid] = OrderedDict()
        model = structure[modelid]
        atom_fields = "record atomname altloc icode x y z q b e charge".split()
        anisou_fields = "u00 u11 u22 u01 u02 u12".split()
        while line.startswith(self.COORDINATE_SECTION):
            if line.startswith(('ATOM  ', 'HETATM')):
                values = CoorRecord.parse_line(line)
                chainid = values['chain']
                if chainid not in model:
                    model[chainid] = OrderedDict()
                chain = model[chainid]
                resid = "{}-{}".format(values['resi'], values['resn'])
                if resid not in chain:
                    chain[resid] = OrderedDict()
                residue = chain[resid]
                atomid = values['atomid']
                residue[atomid] = {}
                atom = residue[atomid]
                for field in atom_fields:
                    atom[field] = values[field]
            elif line.startswith('ANISOU'):
                values = AnisouRecord.parse_line(line)
                atom = structure[modelid][chain][resid][atomid]
                for field in anisou_fields:
                    atom[field] = values[field]
            elif line.startswith('MODEL'):
                values = ModelRecord.parse_line(line)
                modelid = values['modelid']
                structure[modelid] = OrderedDict()
                model = structure[modelid]
            line = f.readline()
        return pdb

    def write(structure):
        with open(self.fname, 'w') as f:
            for fields in izip(*[getattr(structure, x) for x in CoorRecord.fields]):
                if len(fields[-2]) == 2 or len(fields[2]) == 4:
                    f.write(CoorRecord.line2.format(*fields))
                else:
                    f.write(CoorRecord.line1.format(*fields))
            f.write(EndRecord.line)


class Record(object):

    @classmethod
    def parse_line(cls, line):
        values = {}
        for field, column, dtype in izip(cls.fields, cls.columns, cls.dtypes):
            values[field] = dtype(line[slice(*column)].strip())
        return values


class ModelRecord(Record):
    fields = 'record modelid'
    columns = [(0, 6), (11, 15)]
    dtypes = (str, int)
    line = '{:6s}' + ' ' * 5 + '{:6d}\n'


class CoorRecord(Record):
    fields = 'record atomid atomname altloc resn chain resi icode x y z q b e charge'.split()
    columns = [
        (0, 6), (6, 11), (12, 16), (16, 17), (17, 20), (21, 22), (22, 26),
        (26, 27), (30, 38), (38, 46), (46, 54), (54, 60), (60, 66), (76, 78), (78, 80),
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
