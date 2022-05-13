import gzip
from collections import defaultdict
import itertools as itl
from math import inf
import logging

import iotbx.pdb.hierarchy
from libtbx import group_args, smart_open

__all__ = ["read_pdb", "write_pdb", "ANISOU_FIELDS"]
logger = logging.getLogger(__name__)

ANISOU_FIELDS = ['u00', 'u11', 'u22', 'u01', 'u02', 'u12']

def _extract_record_type(atoms):
    records = []
    for atom in atoms:
        records.append("ATOM" if not atom.hetero else "HETATM")
    return records


def _extract_link_records(pdb_inp):
    link = defaultdict(list)
    for line in pdb_inp.extract_LINK_records():
        try:
            values = LinkRecord.parse_line(line)
            for field in LinkRecord.fields:
                link[field].append(values[field])
        except Exception as e:
            logger.error(str(e))
            logger.error("read_pdb: could not parse LINK data.")
    return link


def read_pdb(fname):
    """
    Parsed PDB file representation by section.

    Attributes:
        coor (dict[str, list): coordinate data
        anisou (dict[str, list]): anisotropic displacement parameter data
        link (dict[str, list]): link records
        crystal_symmetry (cctbx.crystal.symmetry): CRYST1 (and implicitly SCALE)
        resolution (Optional[float]): resolution of pdb file
        unit_cell (Optional[UnitCell]): unit cell object
        pdb_hierarchy (Optional[iotbx.pdb.hierarchy]): CCTBX PDB object
    """

    """Read a pdb file using CCTBX and construct a PDBFile object."""
    # this handles .gz extensions automatically
    pdb_inp = iotbx.pdb.pdb_input(file_name=fname, source_info=None)
    pdb_hierarchy = pdb_inp.construct_hierarchy()
    if len(pdb_hierarchy.models()) > 1:
        raise NotImplementedError("MODEL record is not implemented.")
    crystal_symmetry = pdb_inp.crystal_symmetry()
    atoms = pdb_hierarchy.atoms()
    coordinates = atoms.extract_xyz()
    # FIXME this is just a bad idea, we should work with the IOTBX objects
    # if possible
    data = {
        "record": _extract_record_type(atoms),
        "atomid": atoms.extract_serial(),
        "name": [a.name.strip() for a in atoms],
        "x": [xyz[0] for xyz in coordinates],
        "y": [xyz[1] for xyz in coordinates],
        "z": [xyz[2] for xyz in coordinates],
        "b": atoms.extract_b().as_numpy_array(),
        "q": atoms.extract_occ().as_numpy_array(),
        "resn": [a.parent().resname.strip() for a in atoms],
        "resi": [a.parent().parent().resseq_as_int() for a in atoms],
        "icode": [a.parent().parent().icode.strip() for a in atoms],
        "e": [a.element.strip() for a in atoms],
        "charge": ["" for a in atoms],
        "chain": [a.chain().id.strip() for a in atoms],
        "altloc": [a.parent().altloc.strip() for a in atoms]
    }
    anisou = defaultdict(list)
    for atom in atoms:
        if atom.uij != (-1, -1, -1, -1, -1, -1):
            anisou["atomid"].append(atom.serial)
            for key, n_frac in zip(ANISOU_FIELDS, atom.uij):
                anisou[key].append(int(n_frac*10000))
    # FIXME this will only work for PDB, not mmCIF
    link = _extract_link_records(pdb_inp)
    return group_args(coor=data,
                      anisou=anisou,
                      link=link,
                      resolution=pdb_inp.resolution(),
                      crystal_symmetry=crystal_symmetry,
                      pdb_hierarchy=pdb_hierarchy)


def write_pdb(fname, structure):
    """
    Write a structure to a PDB file using the iotbx.pdb API
    """
    with smart_open.for_writing(fname, gzip_mode='wt') as f:
        if structure.crystal_symmetry:
            f.write("{}\n".format(iotbx.pdb.format_cryst1_and_scale_records(
                structure.crystal_symmetry)))
        if structure.link_data:
            _write_link_data(f, structure)
        d = structure.data
        for i_seq in range(len(d["record"])):
            atom = iotbx.pdb.make_atom_with_labels(
                xyz=d["coor"][i_seq],
                occ=d["q"][i_seq],
                b=d["b"][i_seq],
                hetero=d["record"][i_seq] == "HETATM",
                serial=i_seq+1,
                name=d["name"][i_seq],
                element=d["e"][i_seq],
                charge=d["charge"][i_seq],
                chain_id=d["chain"][i_seq],
                resseq=str(d["resi"][i_seq]),
                icode=d["icode"][i_seq],
                altloc=d["altloc"][i_seq],
                resname=d["resn"][i_seq])
            f.write("{}\n".format(atom.format_atom_record_group()))
        f.write("END")


def _write_link_data(f, structure):
    for record in zip(*[structure.link_data[x] for x in LinkRecord.fields]):
        record = dict(zip(LinkRecord.fields, record))
        if not record['length']:
            # If the LINK length is 0, then leave it blank.
            # This is a deviation from the PDB standard.
            record['length'] = ''
            fmtstr = LinkRecord.fmtstr.replace('{:>5.2f}', '{:5s}')
            f.write(fmtstr.format(*record.values()))
        else:
            f.write(LinkRecord.fmtstr.format(*record.values()))


class RecordParser(object):
    """
    Interface class to provide record parsing routines for a PDB file.  This
    is no longer used for parsing ATOM records or crystal symmetry, which are
    handled by CCTBX, but it remains useful as a generic fixed-column-width
    parser for other records that CCTBX leaves unstructured.

    Deriving classes should have class variables for {fields, columns, dtypes, fmtstr}.
    """

    __slots__ = ("fields", "columns", "dtypes", "fmtstr", "fmttrs")

    @classmethod
    def parse_line(cls, line):
        """Common interface for parsing a record from a PDB file.

        Args:
            line (str): A record, as read from a PDB file.

        Returns:
            dict[str, Union[str, int, float]]: fields that were parsed
                from the record.
        """
        values = {}
        for field, column, dtype in zip(cls.fields, cls.columns, cls.dtypes):
            try:
                values[field] = dtype(line[slice(*column)].strip())
            except ValueError:
                logger.error(f"RecordParser.parse_line: could not parse "
                             f"{field} ({line[slice(*column)]}) as {dtype}")
                values[field] = dtype()
        return values

    @classmethod
    def format_line(cls, values):
        """Formats record values into a line.

        Args:
            values (Iterable[Union[str, int, float]]): Values to be formatted.
        """
        assert len(values) == len(cls.fields)

        # Helper
        flatten = lambda iterable: sum(iterable, ())

        # Build list of spaces
        column_indices = flatten(cls.columns)
        space_columns = zip(column_indices[1:-1:2], column_indices[2:-1:2])
        space_lengths = map(lambda colpair: colpair[1] - colpair[0], space_columns)
        spaces = map(lambda n: " " * n, space_lengths)

        # Build list of fields
        field_lengths = map(lambda colpair: colpair[1] - colpair[0], cls.columns)
        formatted_values = map(lambda args: cls._fixed_length_format(*args),
                               zip(values, cls.fmttrs, field_lengths, cls.dtypes))

        # Intersperse formatted values with spaces
        line = itl.zip_longest(formatted_values, spaces, fillvalue="")
        line = "".join(flatten(line)) + "\n"
        return line

    @staticmethod
    def _fixed_length_format(value, formatter, maxlen, dtype):
        """Formats a value, ensuring the length does not exceed available cols.

        If the value exceeds available length, it will be replaced with a
            "bad value" marker ('X', inf, or 0).

        Args:
            value (Union[str, int, float]): Value to be formatted
            formatter (str): Format-spec string
            maxlen (int): Maximum width of the formatted value
            dtype (type): Type of the field

        Returns:
            str: The formatted value, no wider than maxlen.
        """
        field = formatter.format(value)
        if len(field) > maxlen:
            if dtype is str:
                replacement_field = "X" * maxlen
            elif dtype is float:
                replacement_field = formatter.format(inf)
            elif dtype is int:
                replacement_field = formatter.format(0)
            logger.warning(f"{field} exceeds field width {maxlen} chars. "
                           f"Using {replacement_field}.")
            return replacement_field
        else:
            return field


class LinkRecord(RecordParser):
    # http://www.wwpdb.org/documentation/file-format-content/format33/sect6.html#LINK
    fields  = ("record",
               "name1",  "altloc1", "resn1",  "chain1", "resi1",  "icode1",
               "name2",  "altloc2", "resn2",  "chain2", "resi2",  "icode2",
               "sym1",   "sym2",    "length")
    columns = ((0, 6),
               (12, 16), (16, 17),  (17, 20), (21, 22), (22, 26), (26, 27),
               (42, 46), (46, 47),  (47, 50), (51, 52), (52, 56), (56, 57),
               (59, 65), (66, 72),  (73, 78))
    dtypes  = (str,
               str,      str,       str,      str,      int,      str,
               str,      str,       str,      str,      int,      str,
               str,      str,       float)
    fmtstr  = ('{:<6s}' + ' ' * 6
               + ' ' + '{:<3s}{:1s}{:>3s}' + ' ' + '{:1s}{:>4d}{:1s}' + ' ' * 15
               + ' ' + '{:<3s}{:1s}{:>3s}' + ' ' + '{:1s}{:>4d}{:1s}' + ' ' * 2
               + '{:>6s} {:>6s} {:>5.2f}' + '\n')
