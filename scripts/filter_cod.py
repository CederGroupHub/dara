"""A script to filter and group the cod, removing duplicate structures and classifying
symmetry.
"""

import logging
from pathlib import Path

import ray
from dara import SETTINGS
from dara.cif import Cif
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element

path_to_cod = Path(SETTINGS.PATH_TO_COD)

MAX_NUM_ATOMS = 128


@ray.remote(num_cpus=1)
def parse_structures(filenames):
    """Parse a structure from a CIF file."""
    all_data = []
    for filename in filenames:
        try:
            cif = Cif.from_file(filename)
        except Exception as err:
            logging.error(
                f"Error reading CIF from file: {filename} due to error: {err}"
            )
            continue

        try:
            structure = cif.to_structure(merge_tol=0.01, occupancy_tolerance=100)
        except Exception:
            logging.error(f"Error parsing CIF from file: {filename}. Skipping...")
            continue

        if len(structure) > MAX_NUM_ATOMS:
            logging.error(f"Structure too big: {filename}. Skipping...")
            continue

        composition = structure.composition

        # directly skip some chemical systems
        if composition.chemical_system in {
            "C-H-N-O": 49626,
            "C-Cl-H-N-O": 10999,
            "C-H-O": 17741,
            "C-H-N-O-S": 16300,
        }:
            continue
        try:
            if (
                (Element("C") in composition.elements)
                and (Element("H") in composition.elements)
                and (
                    get_dimensionality_larsen(
                        CrystalNN().get_bonded_structure(
                            structure, on_disorder="take_max_species"
                        )
                    )
                    < 2
                )
            ):
                continue
        except (TypeError, AttributeError, ValueError) as e:
            continue
            # raise Exception(filename) from e

        metadata = cif.data[next(iter(cif.data.keys()))].data

        date = metadata.get("_journal_year", "1900")
        temp = str(metadata.get("_diffrn_ambient_temperature", 0)).split(
            "(", maxsplit=1
        )[
            0
        ]  # remove uncertainty
        if temp == "?":
            temp = "0"

        if temp == "0":
            temp = str(metadata.get("_cell_measurement_temperature", 0)).split(
                "(", maxsplit=1
            )[
                0
            ]  # remove uncertainty
        try:
            temp = float(temp)
        except ValueError:
            logging.error(
                f"Error parsing temperature from file: {filename}. Setting to 0."
            )
            temp = 0

        cod_id = str(
            metadata.get("_cod_database_code", filename.stem)
        )  # prefer whats in CIF
        data = {"structure": structure, "temp": temp, "date": date, "cod_id": cod_id}
        all_data.append(data)
    return all_data


def load_cod_structures():
    """Load COD structures and metadata. Groups by chemical system.

    This assumes you have a COD_2024 folder acquired through the standard rsync approach:
        rsync -av --delete rsync://www.crystallography.net/cif/ COD_2024/

    Note that this folder will probably include two levels of sub folders (e.g.
    COD_2024/7/03/03/...). This will not affect parsing.

    See DaraSettings for the default path or to configure with a dara.yaml file.
    """
    cod_data = {}

    all_cifs = sorted(path_to_cod.rglob("*.cif"))
    all_data = ray.get(
        [
            parse_structures.remote(all_cifs[i : i + 50])
            for i in range(0, len(all_cifs), 50)
        ]
    )

    for a_data in all_data:
        for data in a_data:
            if data is None:
                continue
            chemsys = data["structure"].composition.chemical_system

            if chemsys in cod_data:
                cod_data[chemsys].append(data)
            else:
                cod_data[chemsys] = [data]

    return cod_data


@ray.remote(num_cpus=1)
def group_data(data, mp_struct_info):
    matcher = StructureMatcher(allow_subset=True)
    all_data = []
    # pre-sort by spacegroup to speed up matching
    grouped_structures = {}
    for item in data:
        if len(item["structure"]) > MAX_NUM_ATOMS:
            print("skipping (too big):", item["cod_id"])
            continue
        try:
            sg_data = SpacegroupAnalyzer(
                structure=item["structure"], symprec=0.1
            )._space_group_data
        except Exception:
            sg_data = None

        if sg_data is None:
            try:  # different tolerance
                sg_data = SpacegroupAnalyzer(
                    structure=item["structure"], symprec=0.01
                )._space_group_data
            except Exception:
                try:
                    sg_data = SpacegroupAnalyzer(
                        structure=item["structure"],
                        symprec=0.01,
                        angle_tolerance=10,
                    )._space_group_data
                except Exception:
                    print("No symmetry data for cod ID", item["cod_id"])
                    continue

        if sg_data is None:
            print("No data: Excluding cod ID", item["cod_id"])
            continue

        sg = sg_data["number"]

        try:
            if sg in grouped_structures:
                grouped_structures[sg].append(item)
            else:
                grouped_structures[sg] = [item]
        except Exception as e:
            print(e)
            print(sg_data)
            continue

    # group matched structures for downstream filtering
    sorted_data = []
    for sg, group in grouped_structures.items():
        grouped_structures_sg = {}
        for item in group:
            structure = item["structure"]
            added = False

            for k, v in grouped_structures_sg.items():
                if matcher.fit(structure, v[0]["structure"]):
                    grouped_structures_sg[k].append(item)
                    added = True
                    break

            if not added:
                grouped_structures_sg[len(grouped_structures_sg)] = [item]

        sorted_data.append((sg, list(grouped_structures_sg.values())))

    for sg, groups in sorted_data:
        for group in groups:
            group_sorted = sorted(
                group, key=lambda x: (abs(x["temp"] - 293.0), x["date"])
            )
            selected = group_sorted[0]
            formula = selected["structure"].composition.reduced_formula
            data = [formula, selected["cod_id"], sg, None]

            if formula in mp_struct_info:
                for e_hull, mp_sg in mp_struct_info[formula]:
                    if sg == mp_sg[1]:  # exact structure matching too imprecise
                        data[3] = e_hull  # always gets lowest e_hull of match
                        break
            all_data.append(tuple(data))
    return all_data


def filter_cod_structures(cod_data, mp_struct_info):
    """Filter and prioritize COD structures based on space group and MP data.

    Specifically:

    1) Remove duplicate structures using StructureMatcher
    2) Remove structures that are too big (>128 atoms)
    3) Prioritize structures based on proximity to room temperature
    4) Associate MP data (formation energy, space group) with COD data
    """
    filtered_data = {}

    mp_struct_info_obj_id = ray.put(mp_struct_info)

    all_data = ray.get(
        [group_data.remote(data, mp_struct_info_obj_id) for data in cod_data.values()]
    )
    for chemsys, data in zip(cod_data.keys(), all_data):
        if chemsys in filtered_data:
            filtered_data[chemsys].extend(data)
        else:
            filtered_data[chemsys] = data
    return filtered_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = Path(__file__).resolve()

    logging.info(
        "Loading information on MP structures (formulas, space groups, e_hulls)..."
    )
    mp_struct_info = loadfn(
        Path(__file__).resolve().parent.parent / "src/dara/data/mp_struct_info.json.gz"
    )

    logging.info("Loading COD structures...")
    cod_data = load_cod_structures()

    filtered_data = filter_cod_structures(cod_data, mp_struct_info)

    logging.info("Saving filtered COD data...")

    dumpfn(filtered_data, "cod_filtered_info_2024.json.gz")
