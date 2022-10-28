import gzip
from collections import defaultdict
import itertools as itl
from math import inf
import logging

import iotbx.pdb.hierarchy
from libtbx import group_args

logger = logging.getLogger(__name__)

ANISOU_FIELDS = ["u00", "u11", "u22", "u01", "u02", "u12"]


def _extract_record_type(atoms):
    records = []
    for atom in atoms:
        records.append("ATOM" if not atom.hetero else "HETATM")
    return records


def _extract_link_records(pdb_inp):
    link = {}
    for line in pdb_inp.extract_LINK_records():
        try:
            values = LinkRecord.parse_line(line)
            for field in LinkRecord.fields:
                link[field].append(values[field])
        except:
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
        "resn": [a.parent().resname for a in atoms],
        "resi": [a.parent().parent().resseq_as_int() for a in atoms],
        "icode": [a.parent().parent().icode for a in atoms],
        "e": [a.element.strip() for a in atoms],
        "charge": ["" for a in atoms],
        "chain": [a.chain().id for a in atoms],
        "altloc": [a.parent().altloc for a in atoms],
    }
    anisou = defaultdict(list)
    for atom in atoms:
        if atom.uij != (-1, -1, -1, -1, -1, -1):
            anisou["atomid"].append(atom.serial)
            for key, n_frac in zip(ANISOU_FIELDS, atom.uij):
                anisou[key].append(int(n_frac * 10000))
    # FIXME this will only work for PDB, not mmCIF
    link = _extract_link_records(pdb_inp)
    return group_args(
        coor=data,
        anisou=anisou,
        link=link,
        resolution=pdb_inp.resolution(),
        crystal_symmetry=crystal_symmetry,
        pdb_hierarchy=pdb_hierarchy,
    )


def write_pdb(fname, structure):
    """Write a structure to a pdb file.

    Note:
        This is not complete. At the moment, we only write out LINK data
        and coordinate (ATOM) data.

    Args:
        fname (str): filename to write to
        structure (qfit.structure.Structure): a structure object to convert
            to PDB.
    """
    with open(fname, "w") as f:
        if structure.crystal_symmetry:
            f.write(
                "{}\n".format(
                    iotbx.pdb.format_cryst1_and_scale_records(
                        structure.crystal_symmetry
                    )
                )
            )
        if structure.link_data:
            for record in zip(*[structure.link_data[x] for x in LinkRecord.fields]):
                record = dict(zip(LinkRecord.fields, record))
                if not record["length"]:
                    # If the LINK length is 0, then leave it blank.
                    # This is a deviation from the PDB standard.
                    record["length"] = ""
                    fmtstr = LinkRecord.fmtstr.replace("{:>5.2f}", "{:5s}")
                    f.write(fmtstr.format(*record.values()))
                else:
                    f.write(LinkRecord.fmtstr.format(*record.values()))

        # Write ATOM records
        atomid = 1
        for record in zip(*[getattr(structure, x) for x in CoorRecord.fields]):
            record = dict(zip(CoorRecord.fields, record))
            record[
                "atomid"
            ] = atomid  # Overwrite atomid for consistency within this file.
            # If the element name is a single letter,
            # PDB specification says the atom name should start one column in.
            if len(record["e"]) == 1 and not len(record["name"]) == 4:
                record["name"] = " " + record["name"]

            # Write file
            try:
                f.write(CoorRecord.format_line(record.values()))
            except TypeError:
                logger.error(f"PDBFile.write: could not write: {record}")
            atomid += 1

        # Write EndRecord
        f.write(EndRecord.fmtstr)


class RecordParser(object):
    """Interface class to provide record parsing routines for a PDB file.

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
                logger.error(
                    f"RecordParser.parse_line: could not parse "
                    f"{field} ({line[slice(*column)]}) as {dtype}"
                )
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
        formatted_values = map(
            lambda args: cls._fixed_length_format(*args),
            zip(values, cls.fmttrs, field_lengths, cls.dtypes),
        )

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
            logger.warning(
                f"{field} exceeds field width {maxlen} chars. "
                f"Using {replacement_field}."
            )
            return replacement_field
        else:
            return field


class ModelRecord(RecordParser):
    # http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#MODEL
    fields = ("record", "modelid")
    columns = [(0, 6), (10, 14)]
    dtypes = (str, int)
    fmtstr = "{:<6s}" + " " * 4 + "{:>4d}" + "\n"


class LinkRecord(RecordParser):
    # http://www.wwpdb.org/documentation/file-format-content/format33/sect6.html#LINK
    fields = (
        "record",
        "name1",
        "altloc1",
        "resn1",
        "chain1",
        "resi1",
        "icode1",
        "name2",
        "altloc2",
        "resn2",
        "chain2",
        "resi2",
        "icode2",
        "sym1",
        "sym2",
        "length",
    )
    columns = (
        (0, 6),
        (12, 16),
        (16, 17),
        (17, 20),
        (21, 22),
        (22, 26),
        (26, 27),
        (42, 46),
        (46, 47),
        (47, 50),
        (51, 52),
        (52, 56),
        (56, 57),
        (59, 65),
        (66, 72),
        (73, 78),
    )
    dtypes = (
        str,
        str,
        str,
        str,
        str,
        int,
        str,
        str,
        str,
        str,
        str,
        int,
        str,
        str,
        str,
        float,
    )
    fmtstr = (
        "{:<6s}"
        + " " * 6
        + " "
        + "{:<3s}{:1s}{:>3s}"
        + " "
        + "{:1s}{:>4d}{:1s}"
        + " " * 15
        + " "
        + "{:<3s}{:1s}{:>3s}"
        + " "
        + "{:1s}{:>4d}{:1s}"
        + " " * 2
        + "{:>6s} {:>6s} {:>5.2f}"
        + "\n"
    )


class CoorRecord(RecordParser):
    # http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
    fields = (
        "record",
        "atomid",
        "name",
        "altloc",
        "resn",
        "chain",
        "resi",
        "icode",
        "x",
        "y",
        "z",
        "q",
        "b",
        "e",
        "charge",
    )
    columns = (
        (0, 6),
        (6, 11),
        (12, 16),
        (16, 17),
        (17, 20),
        (21, 22),
        (22, 26),
        (26, 27),
        (30, 38),
        (38, 46),
        (46, 54),
        (54, 60),
        (60, 66),
        (76, 78),
        (78, 80),
    )
    dtypes = (
        str,
        int,
        str,
        str,
        str,
        str,
        int,
        str,
        float,
        float,
        float,
        float,
        float,
        str,
        str,
    )
    fmtstr = (
        "{:<6s}"
        + "{:>5d} {:<4s}{:1s}{:>3s} {:1s}{:>4d}{:1s}"
        + " " * 3
        + "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}"
        + " " * 10
        + "{:>2s}{:>2s}"
        + "\n"
    )
    fmttrs = (
        "{:<6s}",
        "{:>5d}",
        "{:<4s}",
        "{:1s}",
        "{:>3s}",
        "{:1s}",
        "{:>4d}",
        "{:1s}",
        "{:8.3f}",
        "{:8.3f}",
        "{:8.3f}",
        "{:6.2f}",
        "{:6.2f}",
        "{:>2s}",
        "{:>2s}",
    )


class AnisouRecord(RecordParser):
    # http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ANISOU
    fields = (
        "record",
        "atomid",
        "atomname",
        "altloc",
        "resn",
        "chain",
        "resi",
        "icode",
        "u00",
        "u11",
        "u22",
        "u01",
        "u02",
        "u12",
        "e",
        "charge",
    )
    columns = (
        (0, 6),
        (6, 11),
        (12, 16),
        (16, 17),
        (17, 20),
        (21, 22),
        (22, 26),
        (26, 27),
        (28, 35),
        (35, 42),
        (42, 49),
        (49, 56),
        (56, 63),
        (63, 70),
        (76, 78),
        (78, 80),
    )
    dtypes = (
        str,
        int,
        str,
        str,
        str,
        str,
        int,
        str,
        float,
        float,
        float,
        float,
        float,
        float,
        str,
        str,
    )
    fmtstr = (
        "{:<6s}"
        + "{:>5d} {:<4s}{:1s}{:>3s} {:1s}{:>4d}{:1s}"
        + " "
        + "{:>7d}" * 6
        + " " * 6
        + "{:>2s}{:>2s}"
        + "\n"
    )


class ExpdtaRecord(RecordParser):
    fields = ("record", "cont", "technique")
    columns = ((0, 6), (8, 10), (10, 79))
    dtypes = (str, str, str)


class RemarkRecord(RecordParser):
    fields = ("record", "remarkid", "text")
    columns = ((0, 6), (7, 10), (11, 79))
    dtypes = (str, int, str)


class Remark2DiffractionRecord(RecordParser):
    # For diffraction experiments
    fields = ("record", "remarkid", "RESOLUTION", "resolution", "ANGSTROM")
    columns = ((0, 6), (9, 10), (11, 22), (23, 30), (31, 41))
    dtypes = (str, str, str, float, str)


class Remark2NonDiffractionRecord(RecordParser):
    # For diffraction experiments
    fields = ("record", "remarkid", "NOTAPPLICABLE")
    columns = ((0, 6), (9, 10), (11, 38))
    dtypes = (str, str, str)


class Cryst1Record(RecordParser):
    fields = ("record", "a", "b", "c", "alpha", "beta", "gamma", "spg")
    columns = (
        (0, 6),
        (6, 15),
        (15, 24),
        (24, 33),
        (33, 40),
        (40, 47),
        (47, 54),
        (55, 66),
        (66, 70),
    )
    dtypes = (str, float, float, float, float, float, float, str, int)


class EndRecord(RecordParser):
    fields = ("record",)
    columns = ((0, 6),)
    dtypes = (str,)
    fmtstr = "END   " + " " * 74 + "\n"
