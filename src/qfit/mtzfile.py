import numpy as np

from iotbx.file_reader import any_file

class _Column:

    def __init__(self, label, type, min, max, data, source=''):
        self.label = label
        self.type = type
        self.min = min
        self.max = max
        self.data = data
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

    def __init__(self, fname):
        self._process_file_iotbx(fname)
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

    def _process_file_iotbx(self, file_name):
        """
        Load arrays from MTZ using CCTBX methods
        """
        mtz = any_file(file_name).file_content.file_content()
        self.unit_cell = mtz.crystals()[0].unit_cell_parameters()
        self.resmin, self.resmax = mtz.max_min_resolution()
        self.ispg = mtz.space_group_number()
        self.spg = mtz.space_group_name()
        self.crystals = []
        for crystal in mtz.crystals():
            pname = crystal.name()
            cr = _Crystal(pname=pname, xname=pname)
            self.crystals.append(cr)
            for dataset in crystal.datasets():
                ds = _DataSet(dataset.id(), dataset.name())
                cr.datasets.append(ds)
                for column in dataset.columns():
                    column_data = column.extract_values().as_numpy_array()
                    if column.type() == "H":
                        column_data = column_data.astype(np.int32)
                    else:
                        column_data = column_data.copy()
                    col = _Column(column.label(),
                                  column.type(),
                                  np.min(column_data),
                                  np.max(column_data),
                                  column_data)
                    ds.columns.append(col)
                    setattr(ds, column.label(), column_data)
