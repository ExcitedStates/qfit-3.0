import re
import copy
import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple, Union, Iterator

class mmCIFError(Exception):
    """Base class of errors raised by Structure objects."""
    pass

class mmCIFSyntaxError(Exception):
    """Base class of errors raised by Structure objects."""
    def __init__(self, line_num, text):
        Exception.__init__(self)
        self.line_num = line_num
        self.text = text

    def __str__(self):
        return "[line: %d] %s" % (self.line_num, self.text)

class mmCIFRow(dict):
    """Contains one row of data. In a mmCIF file, this is one complete
    set of data found under a section. The data can be accessed by using
    the column names as class attributes.
    """
    __slots__ = ["table"]

    def __eq__(self, other):
        return id(self) == id(other)

    def __deepcopy__(self, memo):
        cif_row = mmCIFRow()
        for key, val in list(self.items()):
            cif_row[key] = val
        return cif_row

    def __contains__(self, column):
        return dict.__contains__(self, column.lower())

    def __setitem__(self, column, value):
        assert value is not None
        dict.__setitem__(self, column.lower(), value)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, column):
        return dict.__getitem__(self, column.lower())

    def getitem_lower(self, clower):
        return dict.__getitem__(self, clower)

    def __delitem__(self, column):
        dict.__delitem__(self, column.lower())

    def get(self, column, default=None):
        return dict.get(self, column.lower(), default)

    def get_lower(self, clower, default=None):
        return dict.get(self, clower, default)

    def has_key(self, column):
        return column.lower() in self

    def has_key_lower(self, clower):
        return clower in self

class mmCIFTable(list):
    """Contains columns and rows of data for a mmCIF section. Rows of data
    are stored as mmCIFRow classes.
    """
    __slots__ = ["name", "columns", "columns_lower", "data"]

    def __init__(self, name, columns=None):
        assert name is not None

        list.__init__(self)
        self.name = name
        if columns is None:
            self.columns = list()
            self.columns_lower = dict()
        else:
            self.set_columns(columns)

    def __deepcopy__(self, memo):
        table = mmCIFTable(self.name, self.columns[:])
        for row in self:
            table.append(copy.deepcopy(row, memo))
        return table

    def __eq__(self, other):
        return id(self) == id(other)

    def is_single(self):
        """Return true if the table is not a _loop table with multiple
        rows of data.
        """
        return len(self) <= 1

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, x):
        """Retrieves mmCIFRow at index x from the table if the argument is
        an integer. If the argument is a string, then the data from the
        first row is returned.
        """
        if isinstance(x, int):
            return list.__getitem__(self, x)

        elif isinstance(x, str):
            try:
                return self[0][x]
            except (IndexError, KeyError):
                raise KeyError

        raise TypeError(x)

    def __setitem__(self, x, value):
        assert value is not None

        if isinstance(x, int) and isinstance(value, mmCIFRow):
            value.table = self
            list.__setitem__(self, x, value)

        elif isinstance(x, str):
            try:
                self[0][x] = value
            except IndexError:
                row = mmCIFRow()
                row[x] = value
                self.append(row)

    def __delitem__(self, i):
        self.remove(self[i])

    def get(self, x, default=None):
        try:
            return self[x]
        except KeyError:
            return default

    def append(self, row):
        assert isinstance(row, mmCIFRow)
        row.table = self
        list.append(self, row)

    def insert(self, i, row):
        assert isinstance(row, mmCIFRow)
        row.table = self
        list.insert(self, i, row)

    def remove(self, row):
        assert isinstance(row, mmCIFRow)
        del row.table
        list.remove(self, row)

    def set_columns(self, columns):
        """Sets the list of column(subsection) names to the list of names in
        columns.
        """
        self.columns = list()
        self.columns_lower = dict()
        for column in columns:
            self.append_column(column)

    def append_column(self, column):
        """Appends a column(subsection) name to the table."""
        clower = column.lower()
        if clower in self.columns_lower:
            i = self.columns.index(self.columns_lower[clower])
            self.columns[i] = column
            self.columns_lower[clower] = column
        else:
            self.columns.append(column)
            self.columns_lower[clower] = column

    def has_column(self, column):
        """Tests if the table contains the column name."""
        return column.lower() in self.columns_lower

    def remove_column(self, column):
        """Removes the column name from the table."""
        clower = column.lower()
        if clower not in self.columns_lower:
            return
        self.columns.remove(self.columns_lower[clower])
        del self.columns_lower[clower]

    def autoset_columns(self):
        """Automatically sets the mmCIFTable column names by inspecting all
        mmCIFRow objects it contains.
        """
        clower_used = {}
        for cif_row in self:
            for clower in list(cif_row.keys()):
                clower_used[clower] = True
                if clower not in self.columns_lower:
                    self.append_column(clower)
        for clower in list(self.columns_lower.keys()):
            if clower not in clower_used:
                self.remove_column(clower)

    def get_row1(self, clower, value):
        """Return the first row which which has column data matching value."""
        fpred = lambda r: r.get_lower(clower) == value
        list(filter(fpred, self))
        for row in filter(fpred, self):
            return row
        return None

    def get_row(self, *args):
        """Performs a SQL-like 'AND' select aginst all the rows in the table,
        and returns the first matching row found. The arguments are a
        variable list of tuples of the form:
          (<lower-case-column-name>, <column-value>)
        For example:
          get_row(('atom_id','CA'),('entity_id', '1'))
        returns the first matching row with atom_id==1 and entity_id==1.
        """
        if len(args) == 1:
            clower, value = args[0]
            for row in self:
                if row.get_lower(clower) == value:
                    return row
        else:
            for row in self:
                match_row = True
                for clower, value in args:
                    if row.get_lower(clower) != value:
                        match_row = False
                        break
                if match_row:
                    return row
        return None

    def new_row(self):
        """Creates a new mmCIF rows, addes it to the table, and returns it."""
        cif_row = mmCIFRow()
        self.append(cif_row)
        return cif_row

    def iter_rows(self, *args):
        """This is the same as get_row, but it iterates over all matching
        rows in the table.
        """
        for cif_row in self:
            match_row = True
            for clower, value in args:
                if cif_row.get_lower(clower) != value:
                    match_row = False
                    break
            if match_row:
                yield cif_row

    def row_index_dict(self, clower):
        """Return a dictionary mapping the value of the row's value in
        column 'key' to the row itself. If there are multiple rows with
        the same key value, they will be overwritten with the last found
        row.
        """
        dictx = dict()
        for row in self:
            try:
                dictx[row.getitem_lower(clower)] = row
            except KeyError:
                pass
        return dictx

class mmCIFData(list):
    """Contains all information found under a data_ block in a mmCIF file.
    mmCIF files are represented differently here than their file format
    would suggest. Since a mmCIF file is more-or-less a SQL database dump,
    the files are represented here with their sections as "Tables" and
    their subsections as "Columns". The data is stored in "Rows".
    """
    __slots__ = ["name", "file"]

    def __init__(self, name):
        assert name is not None
        list.__init__(self)
        self.name = name

    def __str__(self):
        return "mmCIFData(name = %s)" % (self.name)

    def __deepcopy__(self, memo):
        data = mmCIFData(self.name)
        for table in self:
            data.append(copy.deepcopy(table, memo))
        return data

    def __eq__(self, other):
        return id(self) == id(other)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, x):
        if isinstance(x, int):
            return list.__getitem__(self, x)

        elif isinstance(x, str):
            name = x.lower()
            for ctable in self:
                if ctable.name.lower() == name:
                    return ctable
            raise KeyError(x)

        raise TypeError(x)

    def __setitem__(self, x, table):
        """ """
        assert isinstance(table, mmCIFTable)

        try:
            old_table = self[x]
        except (KeyError, IndexError):
            pass
        else:
            self.remove(old_table)

        if isinstance(x, int):
            table.data = self
            list.__setitem__(self, x, table)

        elif isinstance(x, str):
            self.append(table)

    def __delitem__(self, x):
        """Remove a mmCIFTable by index or table name."""
        self.remove(self[x])

    def append(self, table):
        """Append a mmCIFTable. This will trigger the removal of any table
        with the same name.
        """
        assert isinstance(table, mmCIFTable)
        try:
            del self[table.name]
        except KeyError:
            pass
        table.data = self
        list.append(self, table)

    def insert(self, i, table):
        assert isinstance(table, mmCIFTable)
        try:
            del self[table.name]
        except KeyError:
            pass
        table.data = self
        list.insert(self, i, table)

    def remove(self, table):
        assert isinstance(table, mmCIFTable)
        del table.data
        list.remove(self, table)

    def has_key(self, x):
        try:
            self[x]
        except KeyError:
            return False
        else:
            return True

    def get(self, x, default=None):
        try:
            return self[x]
        except KeyError:
            return default

    def has_table(self, x):
        try:
            self[x]
        except KeyError:
            return False
        else:
            return True

    def get_table(self, name: str) -> Optional[mmCIFTable]:
        """Looks up and returns a stored mmCIFTable class by its name. This
        name is the section key in the mmCIF file.
        
        Parameters
        ----------
        name : str
            Name of the table to retrieve.
            
        Returns
        -------
        Optional[mmCIFTable]
            The requested table, or None if not found.
        """
        try:
            return self[name]
        except KeyError:
            return None
        except IndexError:
            return None

    def new_table(self, name, columns=None):
        """Creates and returns a mmCIFTable object with the given name.
        The object is added to this object before it is returned.
        """
        cif_table = mmCIFTable(name, columns)
        self.append(cif_table)
        return cif_table

    def split_tag(self, tag):
        cif_table_name, cif_column_name = tag[1:].split(".")
        return cif_table_name.lower(), cif_column_name.lower()

    def join_tag(self, cif_table_name, cif_column_name):
        return "_%s.%s" % (cif_table_name, cif_column_name)

    def get_tag(self, tag):
        """Get."""
        table_name, column = self.split_tag(tag)
        try:
            return self[table_name][column]
        except KeyError:
            return None

    def set_tag(self, tag, value):
        """Set.x"""
        table_name, column = self.split_tag(tag)
        self[table_name][column] = value

class mmCIFSave(mmCIFData):
    """Class to store data from mmCIF dictionary save_ blocks. We treat
    them as non-nested sections along with data_ sections.
    This may not be correct!
    """
    pass

class mmCIFFile(list):
    """Class representing a mmCIF files."""

    # mmCIF Maximum Line Length
    MAX_LINE = 2048

    def __deepcopy__(self, memo):
        cif_file = mmCIFFile()
        for data in self:
            cif_file.append(copy.deepcopy(data, memo))
        return cif_file

    def __str__(self):
        l = [str(cdata) for cdata in self]
        return "mmCIFFile([%s])" % (", ".join(l))

    def __eq__(self, other):
        return id(self) == id(other)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, x):
        """Retrieve a mmCIFData object by index or name."""
        if isinstance(x, int):
            return list.__getitem__(self, x)

        elif isinstance(x, str):
            name = x.lower()
            for cdata in self:
                if cdata.name.lower() == name:
                    return cdata
            raise KeyError(x)

        raise TypeError(x)

    def __delitem__(self, x):
        """Remove a mmCIFData by index or data name. Raises IndexError
        or KeyError if the mmCIFData object is not found, the error raised
        depends on the argument type.
        """
        self.remove(self[x])

    def append(self, cdata):
        """Append a mmCIFData object. This will trigger the removal of any
        mmCIFData object in the file with the same name.
        """
        assert isinstance(cdata, mmCIFData)
        try:
            del self[cdata.name]
        except KeyError:
            pass
        cdata.file = self
        list.append(self, cdata)

    def insert(self, i, cdata):
        assert isinstance(cdata, mmCIFData)
        try:
            del self[cdata.name]
        except KeyError:
            pass
        cdata.file = self
        list.insert(self, i, cdata)

    def has_key(self, x):
        for cdata in self:
            if cdata.name == x:
                return True
        return False

    def get(self, x, default=None):
        try:
            return self[x]
        except KeyError:
            return default

    def load_file(self, fil: Union[str, Any]) -> None:
        """Load and append the mmCIF data from file object fil into self.
        The fil argument must be a file object or implement its iterface.
        
        Parameters
        ----------
        fil : Union[str, Any]
            Path to the file or a file-like object.
        """
        if isinstance(fil, str):
            fileobj = open(fil, "r")
        else:
            fileobj = fil
        mmCIFFileParser().parse_file(fileobj, self)

    def save_file(self, fil: Union[str, Any]) -> None:
        """Write the mmCIF data to a file.
        
        Parameters
        ----------
        fil : Union[str, Any]
            Path to the file or a file-like object.
        """
        if isinstance(fil, str):
            fileobj = open(fil, "w")
        else:
            fileobj = fil
        mmCIFFileWriter().write_file(fileobj, self)

    def get_data(self, name: str) -> Optional[mmCIFData]:
        """Returns the mmCIFData object with the given name. Returns None
        if no such object exists.
        
        Parameters
        ----------
        name : str
            Name of the mmCIFData object to retrieve.
            
        Returns
        -------
        Optional[mmCIFData]
            The requested mmCIFData object, or None if not found.
        """
        try:
            return self[name]
        except KeyError:
            return None
        except IndexError:
            return None

    def new_data(self, name: str) -> mmCIFData:
        """Creates a new mmCIFData object with the given name, adds it
        to this mmCIFFile, and returns it.
        
        Parameters
        ----------
        name : str
            Name of the new mmCIFData object.
            
        Returns
        -------
        mmCIFData
            The newly created mmCIFData object.
        """
        cif_data = mmCIFData(name)
        self.append(cif_data)
        return cif_data

    @classmethod
    def read(cls, fname: str) -> 'mmCIFFile':
        """Read a mmCIF file and construct a mmCIFFile object with attributes
        compatible with PDBFile for use with the Structure class.
        
        Args:
            fname (str): filename of mmCIF file to read
            
        Returns:
            mmCIFFile: object containing parsed sections of the mmCIF file
        """
        # Initialize object
        cif_file = cls()
        cif_file.source = fname
        
        # Set up required data structures
        cif_file.coor = defaultdict(list)
        cif_file.anisou = defaultdict(list)
        cif_file.link = defaultdict(list)
        cif_file.cryst1 = {}
        cif_file.scale = []
        cif_file.cryst_info = []
        cif_file.resolution = None
        
        # Load and parse file
        cif_file.load_file(fname)
        
        # Extract data from parsed mmCIF structure
        if len(cif_file) > 0:
            block = cif_file[0]  # Get first data block
            cif_file._extract_data(block)
        
        return cif_file

    @classmethod
    def write(cls, fname, structure):
        """Write a structure to a mmCIF file.
    
        Parameters
        ----------
        fname : str
            Filename to write to
        structure : Structure
            A structure object to convert to mmCIF
        """
        cif_file = cls()
        data_block = mmCIFData(os.path.basename(fname).split('.')[0])
        cif_file.append(data_block)
    
        # Create atom_site table
        atom_site = data_block.new_table("atom_site")
        atom_site.set_columns([
            "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
            "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id",
            "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
            "B_iso_or_equiv", "auth_seq_id", "auth_comp_id", "auth_asym_id",
            "auth_atom_id", "pdbx_PDB_model_num"
        ])
    
        # Add coordinates data
        for i in range(structure.natoms):
            row = atom_site.new_row()
            row["group_PDB"] = structure.record[i]
            row["id"] = str(structure.atomid[i])
            row["type_symbol"] = structure.e[i]
            row["label_atom_id"] = structure.name[i]
            row["label_alt_id"] = structure.altloc[i] if structure.altloc[i] else "."
            row["label_comp_id"] = structure.resn[i]
            row["label_asym_id"] = structure.chain[i]
            row["label_entity_id"] = "1"  # Default entity ID
            row["label_seq_id"] = str(structure.resi[i])
            row["pdbx_PDB_ins_code"] = structure.icode[i] if structure.icode[i] else "?"
            row["Cartn_x"] = f"{structure.x[i]:.3f}"
            row["Cartn_y"] = f"{structure.y[i]:.3f}"
            row["Cartn_z"] = f"{structure.z[i]:.3f}"
            row["occupancy"] = f"{structure.q[i]:.2f}"
            row["B_iso_or_equiv"] = f"{structure.b[i]:.2f}"
            row["auth_seq_id"] = str(structure.resi[i])
            row["auth_comp_id"] = structure.resn[i]
            row["auth_asym_id"] = structure.chain[i]
            row["auth_atom_id"] = structure.name[i]
            row["pdbx_PDB_model_num"] = "1"
    
        # Add cell parameters if available
        if hasattr(structure, "unit_cell"):
            cell = data_block.new_table("cell")
            cell.set_columns(["length_a", "length_b", "length_c", 
                              "angle_alpha", "angle_beta", "angle_gamma"])
            row = cell.new_row()
            row["length_a"] = f"{structure.unit_cell.a:.3f}"
            row["length_b"] = f"{structure.unit_cell.b:.3f}"
            row["length_c"] = f"{structure.unit_cell.c:.3f}"
            row["angle_alpha"] = f"{structure.unit_cell.alpha:.2f}"
            row["angle_beta"] = f"{structure.unit_cell.beta:.2f}"
            row["angle_gamma"] = f"{structure.unit_cell.gamma:.2f}"
            
            symmetry = data_block.new_table("symmetry")
            symmetry.set_columns(["space_group_name_H-M"])
            row = symmetry.new_row()
            row["space_group_name_H-M"] = structure.unit_cell.spg
    
        # Add LINK records if available
        if structure.link_data and len(structure.link_data.get('record', [])) > 0:
            struct_conn = data_block.new_table("struct_conn")
            struct_conn.set_columns([
                "id", "conn_type_id", 
                "ptnr1_auth_atom_id", "pdbx_ptnr1_label_alt_id", "ptnr1_auth_comp_id", 
                "ptnr1_auth_asym_id", "ptnr1_auth_seq_id", "pdbx_ptnr1_PDB_ins_code",
                "ptnr2_auth_atom_id", "pdbx_ptnr2_label_alt_id", "ptnr2_auth_comp_id", 
                "ptnr2_auth_asym_id", "ptnr2_auth_seq_id", "pdbx_ptnr2_PDB_ins_code",
                "pdbx_ptnr1_symmetry", "pdbx_ptnr2_symmetry", "pdbx_dist_value"
            ])
            
            for i in range(len(structure.link_data['record'])):
                row = struct_conn.new_row()
                row["id"] = f"link{i+1}"
                row["conn_type_id"] = structure.link_data['record'][i]
                row["ptnr1_auth_atom_id"] = structure.link_data['name1'][i]
                row["pdbx_ptnr1_label_alt_id"] = structure.link_data['altloc1'][i] or "."
                row["ptnr1_auth_comp_id"] = structure.link_data['resn1'][i]
                row["ptnr1_auth_asym_id"] = structure.link_data['chain1'][i]
                row["ptnr1_auth_seq_id"] = str(structure.link_data['resi1'][i])
                row["pdbx_ptnr1_PDB_ins_code"] = structure.link_data['icode1'][i] or "?"
                row["ptnr2_auth_atom_id"] = structure.link_data['name2'][i]
                row["pdbx_ptnr2_label_alt_id"] = structure.link_data['altloc2'][i] or "."
                row["ptnr2_auth_comp_id"] = structure.link_data['resn2'][i]
                row["ptnr2_auth_asym_id"] = structure.link_data['chain2'][i]
                row["ptnr2_auth_seq_id"] = str(structure.link_data['resi2'][i])
                row["pdbx_ptnr2_PDB_ins_code"] = structure.link_data['icode2'][i] or "?"
                row["pdbx_ptnr1_symmetry"] = structure.link_data['sym1'][i] or "1_555"
                row["pdbx_ptnr2_symmetry"] = structure.link_data['sym2'][i] or "1_555"
                if structure.link_data['length'][i]:
                    row["pdbx_dist_value"] = f"{structure.link_data['length'][i]:.2f}"
                else:
                    row["pdbx_dist_value"] = "?"
        
        # Write the CIF file
        cif_file.save_file(fname)

    def _extract_data(self, block: mmCIFData) -> None:
        """Extract data from mmCIF block into PDBFile-compatible structures.
        
        Parameters
        ----------
        block : mmCIFData
            The data block to extract information from.
        """
        # Extract atom site data (coordinates)
        self._extract_atom_site(block)
        
        # Extract anisotropic displacement parameters
        self._extract_anisou(block)
        
        # Extract link data
        self._extract_link(block)
        
        # Extract crystallographic data
        self._extract_cell(block)
        
        # Extract resolution data
        self._extract_resolution(block)

    def _extract_atom_site(self, block: mmCIFData) -> None:
        """Extract atom coordinate data from _atom_site category.
        
        Parameters
        ----------
        block : mmCIFData
            The data block to extract information from.
        """
        # Map mmCIF columns to PDBFile attribute names
        column_map = {
            "group_PDB": "record",
            "id": "atomid",
            "auth_atom_id": "name",
            "label_alt_id": "altloc",
            "auth_comp_id": "resn", 
            "auth_asym_id": "chain",
            "auth_seq_id": "resi",
            "pdbx_PDB_ins_code": "icode",
            "Cartn_x": "x",
            "Cartn_y": "y",
            "Cartn_z": "z",
            "occupancy": "q",
            "B_iso_or_equiv": "b",
            "type_symbol": "e",
            "pdbx_formal_charge": "charge"
        }
        
        # Get atom_site table
        atom_site = block.get_table("atom_site")
        if not atom_site:
            return
        
        # Initialize storage for each column
        for attr in column_map.values():
            self.coor[attr] = []
        
        # Extract data from each row
        for row in atom_site:
            for cif_col, attr in column_map.items():
                value = self._get_value_from_row(row, cif_col)
                if attr in ("atomid", "resi"):
                    value = self._try_int(value)
                elif attr in ("x", "y", "z", "q", "b"):
                    value = self._try_float(value)
                self.coor[attr].append(value)

    def _extract_anisou(self, block: mmCIFData) -> None:
        """Extract anisotropic displacement parameters.
        
        Parameters
        ----------
        block : mmCIFData
            The data block to extract information from.
        """
        # Map mmCIF columns to PDBFile attribute names
        column_map = {
            "id": "atomid",
            "U[1][1]": "u00",
            "U[2][2]": "u11", 
            "U[3][3]": "u22",
            "U[1][2]": "u01",
            "U[1][3]": "u02",
            "U[2][3]": "u12",
            "type_symbol": "e",
            "pdbx_formal_charge": "charge"
        }
        
        # Get anisotropic data table
        anisou = block.get_table("atom_site_anisotrop")
        if not anisou:
            return
        
        # Extract required fields
        fields = ["record", "atomid", "atomname", "altloc", "resn", "chain", "resi", "icode"]
        fields.extend(["u00", "u11", "u22", "u01", "u02", "u12", "e", "charge"])
        
        # Initialize storage for each field
        for field in fields:
            self.anisou[field] = []
        
        # Map atom IDs to PDB coordinates for additional data
        atom_id_map = {}
        for i, atom_id in enumerate(self.coor["atomid"]):
            atom_id_map[atom_id] = i
        
        # Extract data from each row
        for row in anisou:
            atom_id = self._try_int(self._get_value_from_row(row, "id"))
            
            # Set record type
            self.anisou["record"].append("ANISOU")
            
            # Get values from anisou table
            for cif_col, attr in column_map.items():
                if cif_col != "id":  # id already processed
                    value = self._get_value_from_row(row, cif_col)
                    if attr in ["u00", "u11", "u22", "u01", "u02", "u12"]:
                        # Convert from standard uncertainty to atomic values (×10⁴)
                        value = self._try_float(value)
                        if value is not None:
                            value = int(value * 10000)
                    self.anisou[attr].append(value)
            
            # Copy additional fields from atom_site table
            if atom_id in atom_id_map:
                idx = atom_id_map[atom_id]
                self.anisou["atomid"].append(self.coor["atomid"][idx])
                self.anisou["atomname"].append(self.coor["name"][idx])
                self.anisou["altloc"].append(self.coor["altloc"][idx])
                self.anisou["resn"].append(self.coor["resn"][idx])
                self.anisou["chain"].append(self.coor["chain"][idx])
                self.anisou["resi"].append(self.coor["resi"][idx])
                self.anisou["icode"].append(self.coor["icode"][idx])

    def _extract_link(self, block: mmCIFData) -> None:
        """Extract bond/link data.
        
        Parameters
        ----------
        block : mmCIFData
            The data block to extract information from.
        """
        # Map mmCIF columns to PDBFile attribute names
        column_map = {
            "conn_type_id": "record",
            "ptnr1_auth_atom_id": "name1",
            "pdbx_ptnr1_label_alt_id": "altloc1", 
            "ptnr1_auth_comp_id": "resn1",
            "ptnr1_auth_asym_id": "chain1",
            "ptnr1_auth_seq_id": "resi1",
            "pdbx_ptnr1_PDB_ins_code": "icode1",
            "ptnr2_auth_atom_id": "name2",
            "pdbx_ptnr2_label_alt_id": "altloc2",
            "ptnr2_auth_comp_id": "resn2", 
            "ptnr2_auth_asym_id": "chain2",
            "ptnr2_auth_seq_id": "resi2",
            "pdbx_ptnr2_PDB_ins_code": "icode2",
            "pdbx_ptnr1_symmetry": "sym1",
            "pdbx_ptnr2_symmetry": "sym2",
            "pdbx_dist_value": "length"
        }
        
        # Get struct_conn table
        links = block.get_table("struct_conn")
        if not links:
            return
        
        # Initialize storage for each column
        for attr in column_map.values():
            self.link[attr] = []
        
        # Extract data from each row
        for row in links:
            for cif_col, attr in column_map.items():
                value = self._get_value_from_row(row, cif_col)
                if attr in ("resi1", "resi2"):
                    value = self._try_int(value)
                elif attr == "length":
                    value = self._try_float(value)
                self.link[attr].append(value)

    def _extract_cell(self, block: mmCIFData) -> None:
        """Extract crystallographic cell parameters.
        
        Parameters
        ----------
        block : mmCIFData
            The data block to extract information from.
        """
        # Map mmCIF columns to PDBFile attribute names
        column_map = {
            "length_a": "a",
            "length_b": "b", 
            "length_c": "c",
            "angle_alpha": "alpha",
            "angle_beta": "beta",
            "angle_gamma": "gamma",
            "space_group_name_H-M": "spg"
        }
        
        # Get cell table
        cell = block.get_table("cell")
        if not cell or len(cell) == 0:
            return
        
        # Get symmetry table for space group
        symmetry = block.get_table("symmetry")
        
        # Extract values from first row of cell table
        row = cell[0]
        for cif_col, attr in column_map.items():
            if cif_col != "space_group_name_H-M":
                value = self._get_value_from_row(row, cif_col)
                if value is not None:
                    value = self._try_float(value)
                if value is not None:
                    self.cryst1[attr] = value
        
        # Get space group from symmetry category if available
        if symmetry and len(symmetry) > 0:
            spg = self._get_value_from_row(symmetry[0], "space_group_name_H-M")
            if spg:
                self.cryst1["spg"] = spg
        
        # Create CRYST1 record string
        if all(k in self.cryst1 for k in ["a", "b", "c", "alpha", "beta", "gamma"]):
            spg = self.cryst1.get("spg", "")
            cryst_line = f"CRYST1{self.cryst1['a']:9.3f}{self.cryst1['b']:9.3f}{self.cryst1['c']:9.3f}"
            cryst_line += f"{self.cryst1['alpha']:7.2f}{self.cryst1['beta']:7.2f}{self.cryst1['gamma']:7.2f} {spg}"
            self.cryst_info.append(cryst_line)

    def _extract_resolution(self, block: mmCIFData) -> None:
        """Extract resolution data.
        
        Parameters
        ----------
        block : mmCIFData
            The data block to extract information from.
        """
        # First try from refine table
        refine = block.get_table("refine")
        if refine and len(refine) > 0:
            value = self._get_value_from_row(refine[0], "ls_d_res_high")
            self.resolution = self._try_float(value)
            if self.resolution is not None:
                return
                
        # If not found, try from reflns table
        reflns = block.get_table("reflns")
        if reflns and len(reflns) > 0:
            value = self._get_value_from_row(reflns[0], "d_resolution_high")
            self.resolution = self._try_float(value)
            if self.resolution is not None:
                return
                
        # Finally try from em_3d_reconstruction table for cryo-EM structures
        em_data = block.get_table("em_3d_reconstruction")
        if em_data and len(em_data) > 0:
            value = self._get_value_from_row(em_data[0], "resolution")
            self.resolution = self._try_float(value)

    def _get_value_from_row(self, row: mmCIFRow, column: str) -> Optional[str]:
        """Extract value from a row with appropriate handling of missing values.
        
        Parameters
        ----------
        row : mmCIFRow
            The row to extract value from.
        column : str
            Column name to extract.
            
        Returns
        -------
        Optional[str]
            Extracted value, or None if not found or null.
        """
        value = row.get(column)
        if value in ("?", "."):
            return None
        return value

    def _try_int(self, value: Any) -> Optional[int]:
        """Try to convert value to integer, return original value on failure.
        
        Parameters
        ----------
        value : Any
            Value to convert.
            
        Returns
        -------
        Optional[int]
            Converted integer value, or original value if conversion failed.
        """
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return value

    def _try_float(self, value: Any) -> Optional[float]:
        """Try to convert value to float, return original value on failure.
        
        Parameters
        ----------
        value : Any
            Value to convert.
            
        Returns
        -------
        Optional[float]
            Converted float value, or original value if conversion failed.
        """
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return value

class mmCIFDictionary(mmCIFFile):
    """Class representing a mmCIF dictionary. The constructor of this class
    takes two arguments. The first is the string path for the file, or
    alternativly a file object.
    """
    pass

class mmCIFFileParser(object):
    """Stateful parser which uses the mmCIFElementFile tokenizer to read
    a mmCIF file and convert it into the mmCIFData/mmCIFTable/mmCIFRow
    data hierarchy.
    """

    def parse_file(self, fileobj, cif_file):
        """Parse a mmCIF file and populate the cif_file data structure.
        
        Parameters
        ----------
        fileobj : file-like object
            The file object to parse.
        cif_file : mmCIFFile
            The file object to populate with parsed data.
        """
        self.line_number = 0
        token_iter = self.gen_token_iter(fileobj)

        try:
            self.parse(token_iter, cif_file)
        except StopIteration:
            # This is expected at the end of file
            pass
        except Exception as e:
            # For other exceptions, raise mmCIFError
            raise mmCIFError(f"Error parsing mmCIF file: {str(e)}")

    def syntax_error(self, err):
        raise mmCIFSyntaxError(self.line_number, err)

    def split_token(self, tokx):
        """Returns the mmCIF token split into a 2-tuple:
        (reserved word, name) where directive is one of the mmCIF
        reserved words: data_, loop_, global_, save_, stop_
        """
        i = tokx.find("_")
        if i == -1:
            return None, None

        rword = tokx[:i].lower()
        if rword not in ("data", "loop", "global", "save", "stop"):
            return None, None

        name = tokx[i + 1 :]
        return rword, name

    def parse(self, token_iter, cif_file):
        """Stateful parser for mmCIF files.

        XXX: loop_, data_, save_ tags are handled in a case-sensitive
             manor. These tokens are case-insensitive.
        """

        cif_table_cache = dict()
        cif_data = None
        cif_table = None
        cif_row = None
        state = ""

        ## ignore anything in the input file until a reserved word is found
        try:
            while True:
                tblx, colx, strx, tokx = next(token_iter)
                if tokx is None:
                    continue
                rword, name = self.split_token(tokx)
                if rword is not None:
                    break
        except StopIteration:
            # Empty file or no data blocks
            return
    
        try:
            while True:
                # PROCESS STATE CHANGES
                if tblx is not None:
                    state = "RD_SINGLE"
                elif tokx is not None:
                    rword, name = self.split_token(tokx)
    
                    if rword == "loop":
                        state = "RD_LOOP"
                    elif rword == "data":
                        state = "RD_DATA"
                    elif rword == "save":
                        state = "RD_SAVE"
                    elif rword == "stop":
                        return
                    elif rword == "global":
                        self.syntax_error("unable to handle global_ syntax")
                    else:
                        self.syntax_error("bad token #1: " + str(tokx))
                else:
                    self.syntax_error("bad token #2")
                    return
    
                # PROCESS DATA IN RD_SINGLE STATE
                if state == "RD_SINGLE":
                    try:
                        cif_table = cif_table_cache[tblx]
                    except KeyError:
                        cif_table = cif_table_cache[tblx] = mmCIFTable(tblx)
    
                        try:
                            cif_data.append(cif_table)
                        except AttributeError:
                            self.syntax_error("section not contained in data_ block")
                            return
    
                        cif_row = mmCIFRow()
                        cif_table.append(cif_row)
                    else:
                        try:
                            cif_row = cif_table[0]
                        except IndexError:
                            self.syntax_error("bad token #3")
                            return
    
                    # Check for duplicate entries
                    if colx in cif_table.columns:
                        self.syntax_error("redefined subsection (column)")
                        return
                    else:
                        cif_table.append_column(colx)
    
                    # Get the next token from the file
                    try:
                        tx, cx, strx, tokx = next(token_iter)
                        if tx is not None or (strx is None and tokx is None):
                            self.syntax_error("missing data for _%s.%s" % (tblx, colx))
    
                        if tokx is not None:
                            # Check token for reserved words
                            rword, name = self.split_token(tokx)
                            if rword is not None:
                                if rword == "stop":
                                    return
                                self.syntax_error("unexpected reserved word: %s" % (rword))
    
                            if tokx != ".":
                                cif_row[colx] = tokx
    
                        elif strx is not None:
                            cif_row[colx] = strx
                        else:
                            self.syntax_error("bad token #4")
                    except StopIteration:
                        self.syntax_error("unexpected end of file")
                        return
    
                    try:
                        tblx, colx, strx, tokx = next(token_iter)
                    except StopIteration:
                        return
                    continue
    
                # PROCESS DATA IN RD_LOOP STATE
                elif state == "RD_LOOP":
                    try:
                        # The first section.subsection (tblx.colx) is read
                        tblx, colx, strx, tokx = next(token_iter)
    
                        if tblx is None or colx is None:
                            self.syntax_error("bad token #5")
                            return
    
                        if tblx in cif_table_cache:
                            self.syntax_error("_loop section duplication")
                            return
    
                        cif_table = mmCIFTable(tblx)
    
                        try:
                            cif_data.append(cif_table)
                        except AttributeError:
                            self.syntax_error("_loop section not contained in data_ block")
                            return
    
                        cif_table.append_column(colx)
    
                        # Read the remaining subsection definitions for the loop_
                        while True:
                            tblx, colx, strx, tokx = next(token_iter)
    
                            if tblx is None:
                                break
    
                            if tblx != cif_table.name:
                                self.syntax_error("changed section names in loop_")
                                return
    
                            cif_table.append_column(colx)
    
                        # Before starting to read data, check tokx for control tokens
                        if tokx is not None:
                            rword, name = self.split_token(tokx)
                            if rword is not None:
                                if rword == "stop":
                                    return
                                else:
                                    self.syntax_error("unexpected reserved word: %s" % (rword))
    
                        # Now read all the data
                        while True:
                            cif_row = mmCIFRow()
                            cif_table.append(cif_row)
    
                            for col in cif_table.columns:
                                if tokx is not None:
                                    if tokx != ".":
                                        cif_row[col] = tokx
                                elif strx is not None:
                                    cif_row[col] = strx
    
                                tblx, colx, strx, tokx = next(token_iter)
    
                            # The loop ends with new table or reserved word
                            if tblx is not None:
                                break
    
                            if tokx is not None:
                                rword, name = self.split_token(tokx)
                                if rword is not None:
                                    break
                    except StopIteration:
                        return
    
                    continue
    
                elif state == "RD_DATA":
                    cif_data = mmCIFData(tokx[5:])
                    cif_file.append(cif_data)
                    cif_table_cache = dict()
                    cif_table = None
    
                    try:
                        tblx, colx, strx, tokx = next(token_iter)
                    except StopIteration:
                        return
    
                elif state == "RD_SAVE":
                    cif_data = mmCIFSave(tokx[5:])
                    cif_file.append(cif_data)
                    cif_table_cache = dict()
                    cif_table = None
    
                    try:
                        tblx, colx, strx, tokx = next(token_iter)
                    except StopIteration:
                        return
        except StopIteration:
            # Normal end of file
            return

    def gen_token_iter(self, fileobj):
        """Generate tokens from mmCIF file, handling end-of-file conditions properly.
        
        Parameters
        ----------
        fileobj : file-like object
            The file object to read tokens from.
            
        Yields
        ------
        tuple
            Token data as (table_name, column_name, string_value, token_value).
        """
        re_tok = re.compile(
            r"(?:"
            r"(?:_(.+?)[.](\S+))"
            "|"  # _section.subsection
            r"(?:['\"](.*?)(?:['\"]\s|['\"]$))"
            "|"  # quoted strings
            r"(?:\s*#.*$)"
            "|"  # comments
            r"(\S+)"  # unquoted tokens
            r")"
        )

        file_iter = iter(fileobj)

        ## parse file, yielding tokens for self.parser()
        try:
            while True:
                try:
                    ln = next(file_iter)
                    self.line_number += 1

                    ## skip comments
                    if ln.startswith("#"):
                        continue

                    ## semi-colon multi-line strings
                    if ln.startswith(";"):
                        lmerge = [ln[1:]]
                        try:
                            while True:
                                ln = next(file_iter)
                                self.line_number += 1
                                if ln.startswith(";"):
                                    break
                                lmerge.append(ln)
                        except StopIteration:
                            # Handle EOF in multiline string
                            self.syntax_error("Unexpected end of file in multiline string")
                            return

                        lmerge[-1] = lmerge[-1].rstrip()
                        yield (None, None, "".join(lmerge), None)
                        continue

                    ## split line into tokens
                    tok_iter = re_tok.finditer(ln)

                    for tokm in tok_iter:
                        groups = tokm.groups()
                        if groups != (None, None, None, None):
                            yield groups
                except StopIteration:
                    # End of file - exit gracefully
                    return
        except Exception as e:
            self.syntax_error(f"Error parsing file: {str(e)}")


class mmCIFFileWriter(object):
    """Writes out a mmCIF file using the data in the mmCIFData list."""

    def write_file(self, fil, cif_data_list):
        self.fil = fil

        ## constant controlls the spacing between columns
        self.SPACING = 2

        ## iterate through the data sections and write them
        ## out to the file
        for cif_data in cif_data_list:
            self.cif_data = cif_data
            self.write_cif_data()

    def write(self, x):
        self.fil.write(x)

    def writeln(self, x=""):
        self.fil.write(x + "\n")

    def write_mstring(self, mstring):
        self.write(self.form_mstring(mstring))

    def form_mstring(self, mstring):
        l = [";"]

        lw = mmCIFFile.MAX_LINE - 2
        for x in mstring.split("\n"):
            if x == "":
                l.append("\n")
                continue

            while len(x) > 0:
                l.append(x[:lw])
                l.append("\n")

                x = x[lw:]

        l.append(";\n")
        return "".join(l)

    def data_type(self, x):
        """Analyze x and return its type: token, qstring, mstring"""
        assert x is not None

        if not isinstance(x, str):
            x = str(x)
            return x, "token"

        if x == "" or x == ".":
            return ".", "token"

        if x.find("\n") != -1:
            return x, "mstring"

        if x.count(" ") != 0 or x.count("\t") != 0 or x.count("#") != 0:
            if len(x) > (mmCIFFile.MAX_LINE - 2):
                return x, "mstring"
            if x.count("' ") != 0 or x.count('" ') != 0:
                return x, "mstring"
            return x, "qstring"

        if len(x) < mmCIFFile.MAX_LINE:
            return x, "token"
        else:
            return x, "mstring"

    def write_cif_data(self):
        if isinstance(self.cif_data, mmCIFSave):
            self.writeln("save_%s" % self.cif_data.name)
        else:
            self.writeln("data_%s" % self.cif_data.name)

        self.writeln("#")

        for cif_table in self.cif_data:
            ## ignore tables without data rows
            if len(cif_table) == 0:
                continue

            ## special handling for tables with one row of data
            elif len(cif_table) == 1:
                self.write_one_row_table(cif_table)

            ## _loop tables
            elif len(cif_table) > 1 and len(cif_table.columns) > 0:
                self.write_multi_row_table(cif_table)

            else:
                raise mmCIFError()

            self.writeln("#")

    def write_one_row_table(self, cif_table):
        row = cif_table[0]

        ## determine max key length for formatting output
        kmax = 0
        table_len = len(cif_table.name) + 2
        for col in cif_table.columns:
            klen = table_len + len(col)
            assert klen < mmCIFFile.MAX_LINE
            kmax = max(kmax, klen)

        ## we need a space after the tag
        kmax += self.SPACING
        vmax = mmCIFFile.MAX_LINE - kmax - 1

        ## write out the keys and values
        for col in cif_table.columns:
            cif_key = "_%s.%s" % (cif_table.name, col)
            l = [cif_key.ljust(kmax)]

            try:
                x0 = row[col]
            except KeyError:
                x = "?"
                dtype = "token"
            else:
                x, dtype = self.data_type(x0)

            if dtype == "token":
                if len(x) > vmax:
                    l.append("\n")
                l.append("%s\n" % (x))
                self.write("".join(l))

            elif dtype == "qstring":
                if len(x) > vmax:
                    l.append("\n")
                    self.write("".join(l))
                    self.write_mstring(x)

                else:
                    l.append("'%s'\n" % (x))
                    self.write("".join(l))

            elif dtype == "mstring":
                l.append("\n")
                self.write("".join(l))
                self.write_mstring(x)

    def write_multi_row_table(self, cif_table):
        ## write the key description for the loop_
        self.writeln("loop_")
        for col in cif_table.columns:
            key = "_%s.%s" % (cif_table.name, col)
            assert len(key) < mmCIFFile.MAX_LINE
            self.writeln(key)

        col_len_map = {}
        col_dtype_map = {}

        for row in cif_table:
            for col in cif_table.columns:
                ## get data and data type
                try:
                    x0 = row[col]
                except KeyError:
                    lenx = 1
                    dtype = "token"
                else:
                    x, dtype = self.data_type(x0)

                    ## determine write length of data
                    if dtype == "token":
                        lenx = len(x)
                    elif dtype == "qstring":
                        lenx = len(x) + 2
                    else:
                        lenx = 0

                try:
                    col_dtype = col_dtype_map[col]
                except KeyError:
                    col_dtype_map[col] = dtype
                    col_len_map[col] = lenx
                    continue

                ## update the column charactor width if necessary
                if col_len_map[col] < lenx:
                    col_len_map[col] = lenx

                ## modify column data type if necessary
                if col_dtype != dtype:
                    if dtype == "mstring":
                        col_dtype_map[col] = "mstring"
                    elif col_dtype == "token" and dtype == "qstring":
                        col_dtype_map[col] = "qstring"

        ## form a write list of the column names with values of None to
        ## indicate a newline
        wlist = []
        llen = 0
        for col in cif_table.columns:
            dtype = col_dtype_map[col]

            if dtype == "mstring":
                llen = 0
                wlist.append((None, None, None))
                wlist.append((col, dtype, None))
                continue

            lenx = col_len_map[col]
            if llen == 0:
                llen = lenx
            else:
                llen += self.SPACING + lenx

            if llen > (mmCIFFile.MAX_LINE - 1):
                wlist.append((None, None, None))
                llen = lenx

            wlist.append((col, dtype, lenx))

        ## write out the data
        spacing = " " * self.SPACING
        add_space = False
        listx = []

        for row in cif_table:
            for col, dtype, lenx in wlist:
                if col is None:
                    add_space = False
                    listx.append("\n")
                    continue

                if add_space == True:
                    add_space = False
                    listx.append(spacing)

                if dtype == "token":
                    x = str(row.get(col, "."))
                    if x == "":
                        x = "."
                    x = x.ljust(lenx)
                    listx.append(x)
                    add_space = True

                elif dtype == "qstring":
                    x = row.get(col, ".")
                    if x == "":
                        x = "."
                    elif x != "." and x != "?":
                        x = "'%s'" % (x)
                    x = x.ljust(lenx)
                    listx.append(x)
                    add_space = True

                elif dtype == "mstring":
                    try:
                        listx.append(self.form_mstring(row[col]))
                    except KeyError:
                        listx.append(".\n")
                    add_space = False

            add_space = False
            listx.append("\n")

            ## write out strx if it gets big to avoid using a lot of
            ## memory
            if len(listx) > 1024:
                self.write("".join(listx))
                listx = []

        ## write out the _loop section
        self.write("".join(listx))

