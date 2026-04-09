"""Convert CIF to Str format for BGMN."""
from __future__ import annotations

import datetime
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from asteval import Interpreter

from dara.utils import (
    POSSIBLE_SPECIES,
    fuzzy_compare,
    load_symmetrized_structure,
    process_phase_name,
    standardize_coords,
)

if TYPE_CHECKING:
    from pymatgen.core import Lattice
    from pymatgen.core.periodic_table import DummySpecie, Element, Specie
    from pymatgen.symmetry.structure import SymmetrizedStructure

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


class CIF2StrError(Exception):
    """CIF2Str error."""


def process_specie_string(sp: str | Specie | Element | DummySpecie) -> str:
    """Reverse the charge notation of a species."""
    specie = re.sub(r"(\d+)([+-])", r"\2\1", str(sp))
    if specie.endswith(("+", "-")):
        specie += "1"
    specie = specie.upper()

    if specie not in POSSIBLE_SPECIES:
        # remove the valence and try again
        specie = re.search(r"[A-Z]+", specie).group(0)
        if specie not in POSSIBLE_SPECIES:
            raise CIF2StrError(
                f"Unknown species {specie}, the original specie string is {sp}"
            )
    return specie


def get_lattice_parameters_from_lattice(
    lattice: Lattice,
    crystal_system: Literal[
        "Monoclinic",
        "Cubic",
        "Hexagonal",
        "Trigonal",
        "Orthorhombic",
        "Triclinic",
        "Tetragonal",
        "Rhombohedral",
    ],
) -> dict[str, float]:
    """
    Get lattice parameters from lattice based on the type of lattice.

    .. note::
        The lattice parameters are in nm
    """
    if crystal_system == "Triclinic":
        return {
            "A": lattice.a / 10,
            "B": lattice.b / 10,
            "C": lattice.c / 10,
            "ALPHA": lattice.alpha,
            "BETA": lattice.beta,
            "GAMMA": lattice.gamma,
        }
    if crystal_system == "Monoclinic":
        return {
            "A": lattice.a / 10,
            "B": lattice.b / 10,
            "C": lattice.c / 10,
            "BETA": lattice.beta,
        }
    if crystal_system == "Orthorhombic":
        return {
            "A": lattice.a / 10,
            "B": lattice.b / 10,
            "C": lattice.c / 10,
        }
    if crystal_system == "Tetragonal":
        return {
            "A": lattice.a / 10,
            "C": lattice.c / 10,
        }
    if crystal_system == "Rhombohedral":
        return {
            "A": lattice.a / 10,
            "GAMMA": lattice.alpha,
        }
    # it seems that the trigonal and hexagonal lattices are the same in BGMN
    if crystal_system == "Hexagonal" or crystal_system == "Trigonal":
        return {
            "A": lattice.a / 10,
            "C": lattice.c / 10,
        }
    if crystal_system == "Cubic":
        return {
            "A": lattice.a / 10,
        }

    raise CIF2StrError(f"Unknown crystal system {crystal_system}")


def get_std_position(
    spacegroup_setting: dict[str, Any],
    wyckoff_letter: str,
    positions: list[list[float]],
) -> tuple[list[float], bool]:
    """Get the standard position of a site based on the hall number and wyckoff notation."""
    wyckoff = spacegroup_setting["wyckoffs"].get(wyckoff_letter, {})

    if not wyckoff:
        logger.debug(f"Spacegroup setting: {spacegroup_setting}")
        raise CIF2StrError(f"Cannot find the wyckoff letter {wyckoff_letter}")

    std_notations = wyckoff["std_notations"]

    positions = [standardize_coords(*position) for position in positions]

    for position in positions:
        variable_dict = {
            "x": position[0],
            "y": position[1],
            "z": position[2],
        }
        for std_notation in std_notations:
            constraints = std_notation.split(" ")

            aeval = Interpreter(use_numpy=False, symtable=variable_dict)
            wx, wy, wz = (aeval.eval(constraint) for constraint in constraints)
            logger.debug([position, (wx, wy, wz)])
            if (
                fuzzy_compare(wx, position[0])
                and fuzzy_compare(wy, position[1])
                and fuzzy_compare(wz, position[2])
            ):
                return position, True
    logger.debug(
        f"Cannot find the standard position for {wyckoff_letter} {std_notations}, using the first position. "
        f"The positions are: \n{positions}"
    )
    return positions[0], False


def check_wyckoff(
    spacegroup_setting: dict[str, Any], structure: SymmetrizedStructure
) -> tuple[list[dict[str, Any]], int]:
    """
    Check if a given spacegroup setting is valid for a structure.

    Args:
        spacegroup_setting: the spacegroup setting
        structure: the symmetrized structure

    Returns
    -------
        the settings of the elements and the number of errors
    """
    element_settings = []
    error_count = 0

    for site_idx in structure.equivalent_indices:
        idx = site_idx[0]
        site = structure[idx]
        wyckoff_letter = structure.wyckoff_letters[idx]
        if wyckoff_letter == "A":
            wyckoff_letter = "alpha"

        std_position, ok = get_std_position(
            spacegroup_setting,
            wyckoff_letter,
            [structure[idx].frac_coords for idx in site_idx],
        )

        if not ok:
            logger.debug(f"Site {site_idx} is not in the standard position")
            error_count += 1

        if site.is_ordered:
            species_string = process_specie_string(str(next(iter(site.species))))
        else:
            sorted_species = sorted(site.species)
            species_string = ",".join(
                f"{process_specie_string(ssp)}({site.species[ssp]:.6f})"
                for ssp in sorted_species
            )
            species_string = f"({species_string})"

        element_setting = {
            "E": species_string,
            "Wyckoff": wyckoff_letter,
            "x": f"{std_position[0]:.6f}",
            "y": f"{std_position[1]:.6f}",
            "z": f"{std_position[2]:.6f}",
            "TDS": f"{0.01:.6f}",
        }
        element_settings.append(element_setting)

    return element_settings, error_count


def make_spacegroup_setting_str(spacegroup_setting: dict[str, Any]) -> str:
    """Make the spacegroup setting string."""
    return (
        " ".join([f"{k}={v}" for k, v in spacegroup_setting["setting"].items()]) + " //"
    )


def make_lattice_parameters_str(
    spacegroup_setting: dict[str, Any],
    structure: SymmetrizedStructure,
    lattice_range: float | Literal["fixed"],
) -> str:
    """Make the lattice parameters string."""
    crystal_system = spacegroup_setting["setting"]["Lattice"]
    lattice_parameters = get_lattice_parameters_from_lattice(
        structure.lattice, crystal_system
    )

    if lattice_range == "fixed":
        lattice_parameters_str = " ".join(
            [f"{k}={v:.5f}" for k, v in lattice_parameters.items()]
        )
    else:
        lattice_parameters_str = " ".join(
            [
                f"PARAM={k}={v:.5f}_{v * (1 - lattice_range):.5f}^{v * (1 + lattice_range):.5f}"
                for k, v in lattice_parameters.items()
            ]
        )
    lattice_parameters_str += " //"
    return lattice_parameters_str


def make_peak_parameter_str(k1: str, k2: str, b1: str, gewicht: str, rp: int) -> str:
    """Make the peak parameter string."""
    return (
        f"RP={rp} "
        + (f"PARAM=k1={k1} " if k1 != "fixed" else "k1=0 ")
        + (f"PARAM=k2={k2} " if k2 != "fixed" else "k2=0 ")
        + (f"PARAM=B1={b1} " if b1 != "fixed" else "B1=0 ")
        + (f"GEWICHT={gewicht} //" if gewicht != "0_0" else "PARAM=GEWICHT=0_0 //")
    )


def cif2str(
    cif_path: Path,
    phase_name_suffix: str = "",
    working_dir: Path | None = None,
    *,
    lattice_range: float = 0.1,
    gewicht: str = "0_0",
    rp: int = 4,
    k1: str = "0_0^0.01",
    k2: str = "0_0^0.01",
    b1: str = "0_0^0.01",
    lebail: bool = False,
    custom_params: list[str] | None = None,
    custom_params_map: dict[str, dict] | None = None,
) -> Path:
    """
    Convert CIF to Str format.

    Args:
        cif_path: the path to the CIF file
        phase_name_suffix: the suffix of the phase name
        working_dir: the folder to hold the processed str file
        lattice_range: the range of the lattice parameters to be refined
        gewicht: the weight fraction of the phase to be refined. Options: 0_0, SPHAR0, and SPHAR2. If 0_0, then no
            preferred orientation. Read more in the BGMN manual.
        rp: the peak function to be used in the refinement. Read more in the BGMN manual.
        k1: the first peak parameter to be refined. Read more in the BGMN manual.
        k2: the second peak parameter to be refined. Read more in the BGMN manual.
        b1: the third peak parameter to be refined. Read more in the BGMN manual.
        lebail: whether to use the Le Bail method
        custom_params: optional list of custom BGMN string parameters to inject. This allows for defining complex
            mathematical equations, global parameters, or fractional occupancies
            (e.g., ["PARAM=Bglobal=0.05_0.01^0.20 //", "PARAM=BO=0.1_0.02^0.3 //"]).
        custom_params_map: optional dictionary mapping element symbols to dictionaries of parameters to add
            or overwrite. You can use the wildcard key "*" to apply parameters to all elements that are not
            specifically matched in the dictionary (e.g., {"*": {"TDS": "Bglobal"}, "O": {"TDS": "BO", "Occ": "OccO"}}).

    An example of the output .str file when using
    custom_params=["PARAM=Bglobal=0.05_0.01^0.20 //", "PARAM=BO=0.1_0.02^0.3 //"]
    and custom_params_map={"*": {"TDS": "Bglobal"}, "O": {"TDS": "BO", "Occ": "OccO"}}:

    PHASE=BariumzirconiumtinIVoxide105053 // ICSD_43137
    Reference=ICSD_43137 //
    Formula=Ba1_O3_Sn0.5_Zr0.5 //
    SpacegroupNo=221 HermannMauguin=P4/m-32/m Setting=1 Lattice=Cubic //
    PARAM=A=0.416280_0.412117^0.420443 //
    RP=4 k1=0 k2=0 PARAM=B1=0_0^0.01 GEWICHT=SPHAR4 //
    GOAL:BariumzirconiumtinIVoxide105053=GEWICHT*ifthenelse(ifdef(d),exp(my*d*3/4),1) //
    PARAM=Bglobal=0.05_0.01^0.20 //
    PARAM=BO=0.1_0.02^0.3 //
    E=BA+2 Wyckoff=b x=0.500000 y=0.500000 z=0.500000 TDS=Bglobal
    E=(ZR+4(0.5000),SN+4(0.5000)) Wyckoff=a x=0.000000 y=0.000000 z=0.000000 TDS=Bglobal
    E=O-2 Wyckoff=d x=0.500000 y=0.000000 z=0.000000 TDS=BO Occ=OccO

    """
    str_path = (
        cif_path.parent / f"{cif_path.stem}.str"
        if working_dir is None
        else working_dir / f"{cif_path.stem}.str"
    )

    structure, spg = load_symmetrized_structure(cif_path)

    hall_number = str(spg.get_symmetry_dataset().hall_number)
    with (Path(__file__).parent / "data" / "spglib_db" / "spg.json").open(
        "r", encoding="utf-8"
    ) as f:
        spg_group_db = json.load(f)
    settings = spg_group_db[hall_number]["settings"]

    best_setting = None
    for spacegroup_setting in settings:
        element_settings, error_count = check_wyckoff(spacegroup_setting, structure)
        if best_setting is None or error_count < best_setting[2]:
            best_setting = (spacegroup_setting, element_settings, error_count)

        if error_count == 0:
            break

    spacegroup_setting, element_settings, error_count = best_setting

    if error_count > 0:
        logger.debug(f"CIF file: {cif_path.read_text()}")
        logger.debug(f"Symmetry dataset: {spg.get_symmetry_dataset()}")
        raise CIF2StrError(
            f"Cannot find a valid lattice symmetry setting for {cif_path}."
        )

    logger.debug(
        f"Using setting {spacegroup_setting['setting']} for {cif_path}, with {error_count} errors"
    )

    # start to construct the str file string
    str_text = ""

    # add some metadata
    phase_name = process_phase_name(cif_path.stem + phase_name_suffix)
    str_text += f"PHASE={phase_name} // generated by pymatgen {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    formula = structure.composition.reduced_formula
    str_text += f"FORMULA={formula} //\n"

    # add spacegroup setting
    str_text += make_spacegroup_setting_str(spacegroup_setting) + "\n"

    # add lattice
    str_text += (
        make_lattice_parameters_str(
            spacegroup_setting, structure, lattice_range=lattice_range
        )
        + "\n"
    )

    # add RP
    str_text += make_peak_parameter_str(k1, k2, b1, gewicht, rp) + "\n"

    # add lebail
    if lebail:
        str_text += "LeBail=1\n"

    # add goals
    str_text += f"GOAL:{phase_name}=GEWICHT*ifthenelse(ifdef(d),exp(my*d*3/4),1) //\nGOAL=GrainSize(1,1,1) //\n"

    # add custom params injected from python dict
    if custom_params is not None:
        for line in custom_params:
            str_text += f"{line}\n"

    if custom_params_map is None:
        custom_params_map = {}

    # add wyckoff positions and overwrite parameters if mapped
    element_settings_str = []
    for element_setting in element_settings:
        element_name = element_setting.get("E", "")

        assigned_dict = None

        # First, check if there is a specific match for this element
        for element_key, custom_dict in custom_params_map.items():
            if element_key != "*" and element_key in element_name:
                assigned_dict = custom_dict
                break

        # If no specific match was found, check if a wildcard was provided
        if assigned_dict is None and "*" in custom_params_map:
            assigned_dict = custom_params_map["*"]

        # If we found either a specific match or a wildcard, update the dictionary
        if assigned_dict is not None:
            element_setting.update(assigned_dict)

        line_str = " ".join([f"{k}={v}" for k, v in element_setting.items()])
        element_settings_str.append(line_str)

    str_text += "\n".join(element_settings_str)

    with open(str_path, "w") as f:
        f.write(str_text)

    return str_path
