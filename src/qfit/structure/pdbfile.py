from collections import defaultdict, namedtuple
import itertools as itl
from math import inf
import logging

import numpy as np

import iotbx.pdb
import iotbx.cif.model
from libtbx import smart_open

__all__ = ["read_pdb", "write_pdb", "read_pdb_or_mmcif", "write_mmcif", "ANISOU_FIELDS"]
logger = logging.getLogger(__name__)

ANISOU_FIELDS = ("u00", "u11", "u22", "u01", "u02", "u12")


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


def _extract_mmcif_links(mmcif_inp):
    try:
        _ = mmcif_inp.cif_block[LinkRecord.cif_fields[0]]
    except KeyError:
        return {}
    link = {}
    for field_name, cif_key, dtype in zip(
        LinkRecord.fields, LinkRecord.cif_fields, LinkRecord.dtypes
    ):

        def _to_value(x):
            if x == "?":
                return ""
            else:
                return dtype(x)

        raw_values = mmcif_inp.cif_block[cif_key]
        link[field_name] = [_to_value(x) for x in raw_values]
    return link


def get_pdb_hierarchy(pdb_inp):
    """
    Prepare an iotbx.pdb.hierarchy object from an iotbx.pdb.input object
    """
    pdb_hierarchy = pdb_inp.construct_hierarchy()
    if len(pdb_hierarchy.models()) > 1:
        raise NotImplementedError("Multi-model support is not implemented.")
    atoms = pdb_hierarchy.atoms()
    atoms.reset_i_seq()
    atoms.reset_serial()
    atoms.reset_tmp()
    atoms.set_chemical_element_simple_if_necessary()
    return pdb_hierarchy


def read_pdb_or_mmcif(fname):
    """
    Parse a PDB or mmCIF file and return the iotbx.pdb object and associated
    content.
    """
    iotbx_in = iotbx.pdb.pdb_input_from_any(
        file_name=fname, source_info=None, raise_sorry_if_format_error=True
    )
    pdb_inp = iotbx_in.file_content()
    link_data = {}
    if iotbx_in.file_format == "pdb":
        link_data = _extract_link_records(pdb_inp)
    else:
        link_data = _extract_mmcif_links(pdb_inp)
    for attr, array in link_data.items():
        link_data[attr] = np.asarray(array)
    input_cls = namedtuple("PDBInput", ["pdb_in", "link_data", "file_format"])
    return input_cls(pdb_inp, link_data, iotbx_in.file_format)


def read_pdb(fname):
    return read_pdb_or_mmcif(fname)


def write_pdb(fname, structure):
    """
    Write a structure to a PDB file using the iotbx.pdb API
    """
    # FIXME this does not save resolution!
    with smart_open.for_writing(fname, gzip_mode="wt") as f:
        if structure.crystal_symmetry:
            f.write(
                "{}\n".format(
                    iotbx.pdb.format_cryst1_and_scale_records(
                        structure.crystal_symmetry
                    )
                )
            )
        if structure.link_data:
            _write_pdb_link_data(f, structure)
        for atom in structure.get_selected_atoms():
            atom_labels = atom.fetch_labels()
            f.write("{}\n".format(atom_labels.format_atom_record_group()))
        f.write("END")


def _write_pdb_link_data(f, structure):
    for record in zip(*[structure.link_data[x] for x in LinkRecord.fields]):
        record = dict(zip(LinkRecord.fields, record))
        # this will be different if the input was mmCIF
        record["record"] = "LINK"
        if not record["length"]:
            # If the LINK length is 0, then leave it blank.
            # This is a deviation from the PDB standard.
            record["length"] = ""
            fmtstr = LinkRecord.fmtstr.replace("{:>5.2f}", "{:5s}")
            f.write(fmtstr.format(*record.values()))
        else:
            f.write(LinkRecord.fmtstr.format(*record.values()))


def _to_mmcif_link_records(structure):
    if len(structure.link_data) > 0:
        conn_loop = iotbx.cif.model.loop(header=LinkRecord.cif_fields)
        for field_id, cif_key in zip(LinkRecord.fields, LinkRecord.cif_fields):
            for x in structure.link_data[field_id]:
                conn_loop[cif_key].append(str(x))
        return conn_loop
    return None


def load_combined_atoms(*atom_lists):
    """
    Utility to take any number of atom arrays and combine them into a new
    PDB hierarchy.  This is used to combine structures, but also to reorder
    atoms within a new hierarchy (since the hierarchy won't take unsorted
    selections).
    """
    atom_labels = []
    for atoms in atom_lists:
        atom_labels.extend([atom.fetch_labels() for atom in atoms])
    return load_atoms_from_labels(atom_labels)


def load_atoms_from_labels(atom_labels):
    atom_lines = [atom.format_atom_record_group() for atom in atom_labels]
    return iotbx.pdb.pdb_input(source_info="qfit_structure",
                               lines=atom_lines)


def write_mmcif(fname, structure):
    """
    Write a structure to an mmCIF file using the iotbx APIs
    """
    atoms = [atom.fetch_labels() for atom in structure.get_selected_atoms()]
    atom_lines = [atom.format_atom_record_group() for atom in atoms]
    pdb_in = iotbx.pdb.pdb_input(source_info="qfit_structure",
                                 lines=atom_lines)
    hierarchy = pdb_in.construct_hierarchy()
    cif_block = hierarchy.as_cif_block(crystal_symmetry=structure.crystal_symmetry)
    if structure.link_data:
        link_loop = _to_mmcif_link_records(structure)
        if link_loop:
            cif_block.add_loop(link_loop)
    with smart_open.for_writing(fname, gzip_mode="wt") as f:
        cif_object = iotbx.cif.model.cif()
        cif_object["qfit"] = cif_block
        print(cif_object, file=f)


class RecordParser:
    """
    Interface class to provide record parsing routines for a PDB file.  This
    is no longer used for parsing ATOM records or crystal symmetry, which are
    handled by CCTBX, but it remains useful as a generic fixed-column-width
    parser for other records that CCTBX leaves unstructured.

    Deriving classes should have class variables for {fields, columns, dtypes,
    fmtstr}.
    """
    fields = tuple()
    columns = tuple()
    dtypes = tuple()
    fmtstr = tuple()
    fmttrs = tuple()

    # prevent assigning additional attributes
    __slots__ = []

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
    # for mmCIF we need to fetch arrays equivalent to each of the column-based
    # fields in the PDB LINK records
    # https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v40.dic/Categories/struct_conn.html
    cif_fields = [
        "_struct_conn.id",  # not really
        "_struct_conn.ptnr1_label_atom_id",
        "_struct_conn.pdbx_ptnr1_label_alt_id",
        "_struct_conn.ptnr1_auth_comp_id",
        "_struct_conn.ptnr1_auth_asym_id",
        "_struct_conn.ptnr1_auth_seq_id",
        "_struct_conn.pdbx_ptnr1_PDB_ins_code",
        "_struct_conn.ptnr2_label_atom_id",
        "_struct_conn.pdbx_ptnr2_label_alt_id",
        "_struct_conn.ptnr2_auth_comp_id",
        "_struct_conn.ptnr2_auth_asym_id",
        "_struct_conn.ptnr2_auth_seq_id",
        "_struct_conn.pdbx_ptnr2_PDB_ins_code",
        "_struct_conn.ptnr1_symmetry",
        "_struct_conn.ptnr2_symmetry",
        "_struct_conn.pdbx_dist_value",
    ]
