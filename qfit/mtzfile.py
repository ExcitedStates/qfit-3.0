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

import struct

import numpy as np


class Record:

    LENGTH = 80

    @classmethod
    def parse_line(cls, line):
        values = {}
        for field, column, dtype in zip(cls.fields, cls.columns, cls.dtypes):
            values[field] = dtype(line[slice(*column)].decode().strip())
        return values

class VersionRecord(Record):
    record = b'VERS'
    fields = 'record version'.split()
    columns = [(0, 4), (5, 80)]
    dtypes = (str, str)

class TitleRecord(Record):
    record = b'TITLE'
    fields = 'record title'.split()
    columns = ([0, 6], [6, 80])
    dtypes = (str, str)

class NColRecord(Record):
    record = b'NCOL'
    fields = 'record ncol nref numbat'.split()
    columns = ([0,4], [5, 13], [14, 26], [27, 35])
    dtypes = (str, int, int, int)

class CellRecord(Record):
    record = b'CELL'
    fields = 'record a b c alpha beta gamma'.split()
    columns = ([0, 4], [5, 14], [15, 24], [25, 34], [35, 44], [45, 54], [55, 64])
    dtypes= (str, float, float, float, float, float, float)

class SortRecord(Record):
    record = b'SORT'
    fields = 'record isort1 isort2 isort3 isort3'.split()
    columns = ([0, 4], [6, 9], [10, 13], [14, 17], [18, 21])
    dtypes = (str, int, int, int, int)

class SymInfoRecord(Record):
    record = b'SYMINF'
    fields = 'record nsym nsymp symtyp ispg spgname pgname spg_confidence'.split()
    columns = [(0, 6), (7, 10), (11, 13), (14, 15), (16, 21), (22, 44), (45, 50), (51, 52)]
    dtypes = (str, int, int, str, int, str, str, str)

class SymRecord(Record):
    record = b'SYMM'
    fields = 'record symline'.split()
    columns = [(0, 4), (5, 79)]
    dtypes = (str, str)

class ResolutionRecord(Record):
    record = b'RESO'
    fields = 'record resmin resmax'.split()
    columns = [(0, 4), (5, 25), (26, 46)]
    dtypes = (str, float, float)

class ValMRecord(Record):
    record = b'VALM'
    fields = 'record value'.split()
    columns = [(0, 4), (5, 25)]
    dtypes = (str, float)

class ColumnRecord(Record):
    record = b"COLUMN"
    fields = 'record label type min max setid'.split()
    columns = [(0, 6), (7, 37), (38, 39), (40, 57), (58, 75), (76, 80)]
    dtypes = (str, str, str, float, float, int)

class ColSrcRecord(Record):
    record = b"COLSRC"
    fields = 'record label source setid'.split()
    columns = [(0, 6), (7, 37), (38, 74), (76, 80)]
    dtypes = (str, str, str, int)

class ColumnGroup(Record):
    record = b"COLGRP"
    fields = 'record grpname grptype grpposn setid'.split()
    columns = [(0, 6), (7, 37), (38, 68), (69, 73), (73, 74), (75, 79)]
    dtypes = (str, str, str, str, str, int)

class NDifRecord(Record):
    record = b'NDIF'
    fields = 'record numset'.split()
    columns = [(0, 4), (5, 13)]
    dtypes= (str, int)

class ProjectRecord(Record):
    record = b'PROJECT'
    fields = 'record setid pname'.split()
    columns = [(0, 7), (8, 15), (16, 80)]
    dtypes = (str, int, str)

class CrystalRecord(Record):
    record = b'CRYSTAL'
    fields = 'record setid xname'.split()
    columns = [(0, 7), (8, 15), (16, 80)]
    dtypes = (str, int, str)

class DataSetRecord(Record):
    record = b'DATASET'
    fields = 'record setid dname'.split()
    columns = [(0, 7), (8, 15), (16, 80)]
    dtypes = (str, int, str)

class DCellRecord(Record):
    record = b'DCELL'
    fields = 'record setid a b c alpha beta gamma'.split()
    columns = [(0, 5), (9, 16), (17, 27), (27, 37), (37, 47),
               (47, 57), (57, 67), (67, 77)]
    dtypes= (str, int, float, float, float, float, float, float)

class DWavelengthRecord(Record):
    record = b'DWAVEL'
    fields = 'record setid wavelength'.split()
    columns = [(0, 6), (8, 15), (16, 26)]
    dtypes = (str, int, float)

class EndRecord(Record):
    record = b'END'
    fields = ['record']
    columns = [(0, 3)]
    dtypes = (str,)

class MTZEndRecord(Record):
    record = b'MTZENDOFHEADERS'
    fields = ['record']
    columns = [(0, 15)]
    dtypes = (str,)

#TODO implement BATCH related record

RECORDS = [VersionRecord, TitleRecord, NColRecord, CellRecord, SymInfoRecord,
           SymRecord, ResolutionRecord, ValMRecord, ColumnRecord, ColSrcRecord,
           ColumnGroup, NDifRecord, ProjectRecord, CrystalRecord, DataSetRecord,
           DCellRecord, DWavelengthRecord, EndRecord, MTZEndRecord]


class _Column:

    def __init__(self, label, type, min, max, source=''):
        self.label = label
        self.type = type
        self.min = min
        self.max = max
        self.source = source

    def __repr__(self):
        return f"<_Column: {self.label} | {self.type} | {self.min} to {self.max}>"


class _DataSet:

    def __init__(self, id, name):
        self.name = name
        self.id = id
        self.columns = []

    def __repr__(self):
        ncolumns = len(self.columns)
        return f"<_DataSet: {self.name} ({ncolumns})>"


class _Crystal:

    def __init__(self, xname='', pname=''):
        self.xname = xname
        self.pname = pname
        self.datasets = []
        self.a = None
        self.b = None
        self.c = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.setids = set()


class MTZFile:

    HEADER_WIDTH = 80

    def __init__(self, fname):

        with open(fname, 'rb') as f:
            self._process_file(f)
        self.ncrystals = len(self.crystals)

    @property
    def columns(self):
        for ds in self.datasets:
            for column in ds.columns:
                yield column

    @property
    def datasets(self):
        for crystal in self.crystals:
            for dataset in crystal.datasets:
                yield dataset

    def __getitem__(self, key):
        for crystal in self.crystals:
            if key in (crystal.xname, crystal.pname):
                return crystal
        for ds in self.datasets:
            if hasattr(ds, key):
                return getattr(ds, key)
        raise KeyError

    def __repr__(self):
        string = f"<MTZFile:\n    Crystals: {self.ncrystals}"
        return string

    def _process_file(self, f):
        mtz_str = f.read(4)
        assert mtz_str == b'MTZ '
        self.header_location = struct.unpack('i', f.read(4))[0]
        self.stamp = struct.unpack('c' * 4, f.read(4))

        # Move reader to header
        location = (self.header_location - 1) * 4
        f.seek(location)

        # 1st pass. Extract crystals and general values
        self.crystals = []
        lines_read = 0
        while True:
            line = f.read(self.HEADER_WIDTH)
            lines_read += 1
            if lines_read > 10000:
                raise RuntimeError("Can't read MTZ file. Check content")
            if line.startswith(b"NCOL"):
                record = NColRecord.parse_line(line)
                self.ncolumns = record['ncol']
                self.nreflections = record['nref']
                self.nbatches = record['numbat']
            elif line.startswith(b"NDIF"):
                record = NDifRecord.parse_line(line)
                self.nsets = record['numset']
            elif line.startswith(b"PROJ"):
                record = ProjectRecord.parse_line(line)
                pname = record['pname']
                if not pname:
                    pname = 'dummy'
                new_project = True
                for crystal in self.crystals:
                    if pname == crystal.pname:
                        new_project = False
                        break
                if new_project:
                    crystal = _Crystal(pname=pname, xname=pname)
                    crystal.setids.add(record['setid'])
                    self.crystals.append(crystal)
            elif line.startswith(b"CRYS"):
                record = CrystalRecord.parse_line(line)
                if not new_project:
                    crystal = _Crystal(xname=record['xname'])
                    crystal.setids.add(record['setid'])
                    self.crystals.append(crystal)
                else:
                    new_crystal = True
                    for crystal in self.crystals:
                        if record['xname'] == crystal.xname:
                            new_crystal = False
                            crystal.pname = pname
                            crystal.setids.add(record['setid'])
                            break
                    if new_crystal:
                        crystal = _Crystal(xname=record['xname'])
                        crystal.setids.add(record['setid'])
                        self.crystals.append(crystal)
            elif line.startswith(EndRecord.record):
                break

        # 2nd pass, extract datasets
        f.seek(location)
        while True:
            line = f.read(self.HEADER_WIDTH)

            if line.startswith(b"DATA"):
                record = DataSetRecord.parse_line(line)
                for crystal in self.crystals:
                    if record['setid'] in crystal.setids:
                        dataset = _DataSet(record['setid'], record['dname'])
                        crystal.datasets.append(dataset)
                        break
            elif line.startswith(b"DCELL"):
                record = DCellRecord.parse_line(line)
                for crystal in self.crystals:
                    if record['setid'] in crystal.setids:
                        for attr in 'a b c alpha beta gamma'.split():
                            setattr(crystal, attr, record[attr])
                        break
            elif line.startswith(b"DWAV"):
                record = DWavelengthRecord.parse_line(line)
                for dataset in self.datasets:
                    if dataset.id == record['setid']:
                        dataset.wavelength = record['wavelength']
            elif line.startswith(b"END"):
                break

        # 3rd pass, extract general data and columns
        f.seek(location)
        self.symops = []
        while True:
            line = f.read(self.HEADER_WIDTH)

            if line.startswith(b"VERS"):
                record = VersionRecord.parse_line(line)
                self.version = record['version']
            elif line.startswith(b"TITL"):
                record = TitleRecord.parse_line(line)
                self.title = record['title']
            elif line.startswith(b"CELL"):
                record = CellRecord.parse_line(line)
                for crystal in self.crystals:
                    if crystal.a is None:
                        for attr in 'a b c alpha beta gamma'.split():
                            setattr(crystal, attr, record[attr])
            elif line.startswith(b"SYMI"):
                record = SymInfoRecord.parse_line(line)
                self.symi = record
                self.ispg = record['ispg']
                self.spg = record['spgname'].strip("'")
            elif line.startswith(b"SYMM"):
                record = SymRecord.parse_line(line)
                self.symops.append(record['symline'].strip())
            elif line.startswith(b"COLU"):
                record = ColumnRecord.parse_line(line)
                for ds in self.datasets:
                    if ds.id == record['setid']:
                        column = _Column(record['label'], record['type'],
                                         record['min'], record['max'])
                        ds.columns.append(column)
            elif line.startswith(b"VALM"):
                record = ValMRecord.parse_line(line)
                self.valm = record['value']
            elif line.startswith(b"RESO"):
                record = ResolutionRecord.parse_line(line)
                self.resmax = record['resmax']
                self.resmin = record['resmin']
                for crystal in self.crystals:
                    crystal.resmax = record['resmax']
                    crystal.resmin = record['resmin']
            elif line.startswith(b"END"):
                break

        # Read in reflections
        f.seek(20 * 4)
        count = self.ncolumns * self.nreflections
        data = np.fromfile(f, dtype=np.float32,
                           count=count).reshape(-1, self.ncolumns)
        i = 0
        for ds in self.datasets:
            for column in ds.columns:
                if column.type == 'H':
                    column_data = data[:, i].astype(np.int32)
                else:
                    column_data = data[:, i].copy()
                setattr(ds, column.label, column_data)
                i += 1


if __name__ == '__main__':
    #f = open('5g4c_map_p1.mtz', 'rb')
    mtz = MTZFile('2oob_map.mtz')

    f = open('2oob_map.mtz', 'rb')

    mtz = f.read(4)
    header_location = struct.unpack('i', f.read(4))[0]
    stamp = struct.unpack('c' * 4, f.read(4))

    f.seek((header_location - 1) * 4)
    header = f.read()
    lines = [header[x * 80: (x + 1) * 80] for x in range(len(header)//80)]
    records = []
    for p in lines:
        print(p)
        for record in RECORDS:
            if p.startswith(record.record):
                values = record.parse_line(p)
                records.append(values)
                break
        if p.startswith(b'MTZBATS'):
            raise NotImplementedError("Batches are not implemented")

    ncol_record = None
    for record in records:
        if record['record'] == 'NCOL':
            ncol_record = record

    f.seek(20 * 4)
    nterms = ncol_record['nref'] * ncol_record['active']
    reflection_data = np.fromfile(f, dtype=np.float32, count=nterms).astype(np.float64)
    reflection_data = reflection_data.reshape(-1, ncol_record['active'])

    print(reflection_data)
    print(reflection_data.shape)
