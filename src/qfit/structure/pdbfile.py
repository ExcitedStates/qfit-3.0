import gzip
from collections import defaultdict
import itertools as itl
from math import inf
import logging

import numpy as np

import iotbx.pdb
import iotbx.cif.model
from libtbx import group_args, smart_open

__all__ = ["read_pdb", "write_pdb", "read_pdb_or_mmcif", "write_mmcif", "ANISOU_FIELDS"]
logger = logging.getLogger(__name__)

ANISOU_FIELDS = ('u00', 'u11', 'u22', 'u01', 'u02', 'u12')

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
    for field_name, cif_key, dtype in zip(LinkRecord.fields,
                                          LinkRecord.cif_fields,
                                          LinkRecord.dtypes):
        def _to_value(x):
            if x == "?":
                return ""
            else:
                return dtype(x)
        raw_values = mmcif_inp.cif_block[cif_key]
        link[field_name] = [_to_value(x) for x in raw_values]
    return link


def _from_iotbx_pdb_hierarchy(pdb_hierarchy,
                              crystal_symmetry,
                              resolution,
                              link,
                              file_format="pdb"):
    if len(pdb_hierarchy.models()) > 1:
        raise NotImplementedError("Multi-model support is not implemented.")
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
            for key, uij_n_frac in zip(ANISOU_FIELDS, atom.uij):
                uij_n_int = int(np.round(uij_n_frac*10000))
                anisou[key].append(uij_n_int)
    return group_args(coor=data,
                      anisou=anisou,
                      link=link,
                      resolution=resolution,
                      crystal_symmetry=crystal_symmetry,
                      pdb_hierarchy=pdb_hierarchy,
                      file_format=file_format)


def read_pdb_or_mmcif(fname):
    """
    Parsed PDB or mmCIF file representation by section.

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
    iotbx_in = iotbx.pdb.pdb_input_from_any(
        file_name=fname,
        source_info=None,
        raise_sorry_if_format_error =True)
    pdb_inp = iotbx_in.file_content()
    pdb_hierarchy = pdb_inp.construct_hierarchy()
    link = {}
    if iotbx_in.file_format == "pdb":
        link = _extract_link_records(pdb_inp)
    else:
        link = _extract_mmcif_links(pdb_inp)
    return _from_iotbx_pdb_hierarchy(
        pdb_hierarchy=pdb_hierarchy,
        resolution=pdb_inp.resolution(),
        crystal_symmetry=pdb_inp.crystal_symmetry(),
        link=link,
        file_format=iotbx_in.file_format)


def read_pdb(fname):
    return read_pdb_or_mmcif(fname)


def _structure_to_iotbx_atoms(data):
    for i_seq in range(len(data["record"])):
        yield iotbx.pdb.make_atom_with_labels(
            xyz=data["coor"][i_seq],
            occ=data["q"][i_seq],
            b=data["b"][i_seq],
            hetero=data["record"][i_seq] == "HETATM",
            serial=i_seq+1,
            name=data["name"][i_seq],
            element=data["e"][i_seq],
            charge=data["charge"][i_seq],
            chain_id=data["chain"][i_seq],
            resseq=str(data["resi"][i_seq]),
            icode=data["icode"][i_seq],
            altloc=data["altloc"][i_seq],
            resname=data["resn"][i_seq])


def write_pdb(fname, structure):
    """
    Write a structure to a PDB file using the iotbx.pdb API
    """
    # FIXME this does not save resolution!
    with smart_open.for_writing(fname, gzip_mode='wt') as f:
        if structure.crystal_symmetry:
            f.write("{}\n".format(iotbx.pdb.format_cryst1_and_scale_records(
                structure.crystal_symmetry)))
        if structure.link_data:
            _write_pdb_link_data(f, structure)
        for atom in _structure_to_iotbx_atoms(structure.data):
            f.write("{}\n".format(atom.format_atom_record_group()))
        f.write("END")


def _write_pdb_link_data(f, structure):
    for record in zip(*[structure.link_data[x] for x in LinkRecord.fields]):
        record = dict(zip(LinkRecord.fields, record))
        # this will be different if the input was mmCIF
        record['record'] = "LINK"
        if not record['length']:
            # If the LINK length is 0, then leave it blank.
            # This is a deviation from the PDB standard.
            record['length'] = ''
            fmtstr = LinkRecord.fmtstr.replace('{:>5.2f}', '{:5s}')
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


def write_mmcif(fname, structure):
    """
    Write a structure to an mmCIF file using the iotbx APIs
    """
    # FIXME this is really gross, just a quick hack to make mmCIF writable
    atoms = _structure_to_iotbx_atoms(structure.data)
    atom_lines = [atom.format_atom_record_group() for atom in atoms]
    pdb_in = iotbx.pdb.pdb_input(source_info="qfit_structure",
                                 lines=atom_lines)
    hierarchy = pdb_in.construct_hierarchy()
    cif_block = hierarchy.as_cif_block(
        crystal_symmetry=structure.crystal_symmetry)
    if structure.link_data:
        link_loop = _to_mmcif_link_records(structure)
        if link_loop:
            cif_block.add_loop(link_loop)
    with smart_open.for_writing(fname, gzip_mode='wt') as f:
        cif_object = iotbx.cif.model.cif()
        cif_object["qfit"] = cif_block
        print(cif_object, file=f)


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
    # for mmCIF we need to fetch arrays equivalent to each of the column-based
    # fields in the PDB LINK records
    # https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v40.dic/Categories/struct_conn.html
    cif_fields = [
        "_struct_conn.id", # not really
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
        "_struct_conn.pdbx_dist_value"
    ]
