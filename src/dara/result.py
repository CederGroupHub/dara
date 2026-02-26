"""The parser for the result from the refinement."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pymatgen.core import Composition, Lattice, Structure, get_el_sp
from pymatgen.symmetry.groups import SpaceGroup

from dara.plot import visualize
from dara.utils import (
    angular_correction,
    get_number,
    get_wavelength,
    intensity_correction,
)

if TYPE_CHECKING:
    from pathlib import Path


class PhaseResult(BaseModel):
    """The result for each phase."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    spacegroup_no: Optional[int] = Field(alias="SpacegroupNo")
    hermann_mauguin: Optional[str] = Field(alias="HermannMauguin")
    xray_density: Optional[float] = Field(alias="XrayDensity")
    rphase: Optional[float] = Field(alias="Rphase")
    unit: str = Field(alias="UNIT")
    gewicht: Union[float, tuple[float, float], None] = Field(alias="GEWICHT")
    gewicht_name: Optional[str] = Field(alias="GEWICHT_NAME")

    a: Optional[Union[float, tuple[float, float]]] = Field(None, alias="A")
    b: Optional[Union[float, tuple[float, float]]] = Field(None, alias="B")
    c: Optional[Union[float, tuple[float, float]]] = Field(None, alias="C")
    alpha: Optional[Union[float, tuple[float, float]]] = Field(None, alias="ALPHA")
    beta: Optional[Union[float, tuple[float, float]]] = Field(None, alias="BETA")
    gamma: Optional[Union[float, tuple[float, float]]] = Field(None, alias="GAMMA")

    atom_positions_string: Optional[str] = Field(
        None, alias="Atomic positions for phase"
    )

    @model_validator(mode="before")
    @classmethod
    def check_gewicht(cls, values):
        if "GEWICHT" in values and isinstance(values["GEWICHT"], str):
            geweicht = values["GEWICHT"]
            geweicht_mean = float(re.search(r"(\d+\.\d+)", geweicht).group(1))
            geweicht_name = re.search(r"^([A-Z\d]+)", geweicht).group(1)
            values["GEWICHT"] = geweicht_mean
            values["GEWICHT_NAME"] = geweicht_name
        else:
            values["GEWICHT_NAME"] = None
        return values

    def get_structure(self) -> Structure:
        """
        Get the refined structure from the phase result.

        Returns
        -------
            the refined structure as ``pymatgen.Structure`` object
        """
        if not self.atom_positions_string:
            raise ValueError(
                "Cannot find the atomic positions from the phase result. "
                "Please make sure the result is refined using dara >= 0.9.1"
            )
        # get lattice
        lattice_data = {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }
        lattice_data = {
            k: get_number(v) for k, v in lattice_data.items() if v is not None
        }
        for k in ["a", "b", "c"]:
            if k in lattice_data:
                lattice_data[k] = lattice_data[k] * 10
        spacegroup = SpaceGroup.from_int_number(self.spacegroup_no)
        crystal_system = spacegroup.crystal_system
        if crystal_system == "trigonal":
            crystal_system = "hexagonal"
        lattice = getattr(Lattice, crystal_system, Lattice.from_parameters)(
            **lattice_data
        )

        # get species and coords
        all_coords = []
        all_species = []
        for line in self.atom_positions_string.split("\n"):
            if not line:
                continue
            line = line.strip().split()
            coords = [float(i) for i in line[1:4]]
            all_coords.append(coords)

            species = {}
            specie_stirng = re.search(r"E=\((.+)\)", line[-1]).group(
                1
            )  # Sp(Occ), Sp(Occ), ...
            for specie in specie_stirng.split(","):
                specie_string = specie.split("(")[0].capitalize()
                # parse the specie into pymatgen Species
                specie_string = re.sub(r"([+-])(\d+)", r"\2\1", specie_string)
                sp = get_el_sp(specie_string)
                occupancy = float(re.search(r"\((\d+\.\d+)\)", specie).group(1))
                species[sp] = occupancy
            species = Composition(species)
            all_species.append(species)

        return Structure.from_spacegroup(
            sg=self.spacegroup_no,
            lattice=lattice,
            species=all_species,
            coords=all_coords,
            tol=2e-4,  # BGMN only has 4 decimal places for atomic positions
        )


class LstResult(BaseModel):
    """Refinement result parsed from the .lst file."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    raw_lst: str
    pattern_name: str

    num_steps: int
    rp: float = Field(alias="Rp")
    rpb: float = Field(alias="Rpb")
    r: float = Field(alias="R")
    rwp: float = Field(alias="Rwp")
    rexp: float = Field(alias="Rexp")
    d: float = Field(alias="d")
    rho: float = Field(alias="1-rho")
    phases_results: dict[str, PhaseResult]


class DiaResult(BaseModel):
    """Refinement result parsed from the .dia file. Mainly some x-y data for plotting."""

    model_config = ConfigDict(populate_by_name=True)

    x: list[float]
    y_obs: list[float]
    y_calc: list[float]
    y_bkg: list[float]
    structs: dict[str, list[float]]


class RefinementResult(BaseModel):
    """The result from the refinement, which is parsed from the .lst and .dia files."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    lst_data: LstResult
    plot_data: DiaResult = Field(repr=False)
    peak_data: pd.DataFrame = Field(repr=False)

    @field_validator("peak_data", mode="before")
    @classmethod
    def transform(cls, data: dict) -> pd.DataFrame:
        """Create pandas dataframe from peak data dict."""
        return pd.DataFrame(data)

    def visualize(self, diff_offset=False):
        return visualize(self, diff_offset=diff_offset)

    def get_phase_weights(self, normalize=True) -> dict[str, float]:
        """Return the weights for each phase. Default is to normalize and return weight fractions.

        Args:
            normalize: Whether to normalize and return weight fractions. Defaults to True.

        Returns
        -------
            A dictionary of phase names and their weights.
        """
        weights = {}
        for phase, data in self.lst_data.phases_results.items():
            weights[phase] = get_number(data.gewicht)

        if normalize:
            tot = np.sum(list(weights.values()))
            weights = {k: v / tot for k, v in weights.items()}
        return weights

    def export_structure(self, phase_name: str) -> Structure:
        """
        Export the refined structure from the phase result.

        Args:
            phase_name: the name of the phase

        Returns
        -------
            the refined structure as ``pymatgen.Structure`` object
        """
        return self.lst_data.phases_results[phase_name].get_structure()


class ParseError(Exception):
    """Error when parsing the result."""


def get_result(control_file: Path) -> RefinementResult:
    """
    Get the result from the refinement.

    :param control_file: the path to the control file (.sav)
    """
    # get phase names from sav file first
    # example
    # STRUC[1]=Bi2Fe4O9.str
    # STRUC[2]=Bi25FeO39.str
    # STRUC[3]=BiFeO3.str
    try:
        sav_text = control_file.read_text()
        phase_names = re.findall(r"STRUC\[\d+]=(.+?)\.str", sav_text)

        lst_path = control_file.parent / f"{control_file.stem}.lst"
        dia_path = control_file.parent / f"{control_file.stem}.dia"
        par_path = control_file.parent / f"{control_file.stem}.par"

        result = {
            "lst_data": parse_lst(lst_path, phase_names=phase_names),
            "plot_data": parse_dia(dia_path, phase_names=phase_names),
            "peak_data": parse_par(par_path, phase_names=phase_names),
        }

        return RefinementResult(**result)
    except Exception as e:
        raise ParseError(f"Error in parsing the result from {control_file}") from e


def parse_lst(lst_path: Path, phase_names: list[str]) -> LstResult:
    """
    Get results from the .lst file. This file mainly contains some numbers for the refinement.

    Example of the .lst file:

    .. code-block:: none

        Rietveld refinement to file(s) Mg3MnNi3O8.xy
        BGMN version 4.2.23, 8301 measured points, 78 peaks, 20 parameters
        Start: Mon Dec 18 11:43:20 2023; End: Mon Dec 18 11:43:21 2023
        43 iteration steps

        Rp=4.14%  Rpb=50.39%  R=13.55%  Rwp=8.98% Rexp=1.47%
        Durbin-Watson d=0.06
        1-rho=13.6%

        Global parameters and GOALs
        ****************************
        QMg3MnNi3O8166sym=0.0700+-0.0046
        QNiO=0.9300+-0.0046
        EPS2=-0.001657+-0.000033

        Local parameters and GOALs for phase Mg3MnNi3O8166sym
        ******************************************************
        SpacegroupNo=166
        HermannMauguin=R-32/m
        XrayDensity=4.943
        Rphase=26.64%
        UNIT=NM
        A=0.5898+-0.0013
        C=1.4449+-0.0062
        k1=1.00000
        B1=0.00492+-0.00076
        GEWICHT=0.0288+-0.0019
        GrainSize(1,1,1)=64.7+-10.0
        Atomic positions for phase Mg3MnNi3O8166sym
        ---------------------------------------------
          9     0.5000  0.0000  0.0000     E=(MG(1.0000))
          3     0.0000  0.0000  0.0000     E=(MN(1.0000))
          9     0.5000  0.0000  0.5000     E=(NI(1.0000))
         18     0.0268 -0.0268  0.7429     E=(O(1.0000))
          6     0.0000  0.0000  0.2511     E=(O(1.0000))

        Local parameters and GOALs for phase NiO
        ******************************************************
        SpacegroupNo=225
        HermannMauguin=F4/m-32/m
        XrayDensity=6.760
        Rphase=11.31%
        UNIT=NM
        A=0.418697+-0.000027
        k1=0
        B1=0.00798+-0.00022
        GEWICHT=0.3827+-0.0049
        GrainSize(1,1,1)=53.2+-1.5
        Atomic positions for phase NiO
        ---------------------------------------------
          4     0.0000  0.0000  0.0000     E=(NI+2(1.0000))
          4     0.5000  0.5000  0.5000     E=(O-2(1.0000))

    Args:
        lst_path:

    Returns
    -------
        phase_results: a dictionary of the results for each phase

    """

    def parse_values(v_: str) -> float | tuple[float, float] | None | str | int:
        try:
            v_ = v_.strip("%")
            if v_ == "ERROR" or v_ == "UNDEF":
                return None
            if "+-" in v_:
                v_ = (float(v_.split("+-")[0]), float(v_.split("+-")[1]))
            else:
                v_ = float(v_)
                if v_.is_integer():
                    v_ = int(v_)
        except ValueError:
            pass
        return v_

    def parse_section(text: str) -> dict[str, Any]:
        section = dict(re.findall(r"^(\w+)=(.+?)$", text, re.MULTILINE))
        return {k: parse_values(v) for k, v in section.items()}

    if not lst_path.exists():
        raise FileNotFoundError(f"Cannot find the .lst file from {lst_path}")

    with lst_path.open() as f:
        texts = f.read()

    pattern_name = re.search(r"Rietveld refinement to file\(s\) (.+?)\n", texts).group(
        1
    )
    result = {"raw_lst": texts, "pattern_name": pattern_name}

    num_steps = int(re.search(r"(\d+) iteration steps", texts).group(1))
    result["num_steps"] = num_steps

    for var in ["Rp", "Rpb", "R", "Rwp", "Rexp"]:
        result[var] = float(re.search(rf"{var}=(\d+(\.\d+)?)%", texts).group(1))
    result["d"] = (
        float(d.group(1))
        if (d := re.search(r"Durbin-Watson d=(\d+(\.\d+)?)", texts))
        else None
    )
    result["1-rho"] = (
        float(rho.group(1))
        if (rho := re.search(r"1-rho=(\d+(\.\d+)?)%", texts))
        else None
    )

    # global goals
    global_parameters_text = re.search(
        r"Global parameters and GOALs\n(.*?)\n(?:\n|\Z)", texts, re.DOTALL
    )
    if global_parameters_text:
        global_parameters_text = global_parameters_text.group(1)
        global_parameters = parse_section(global_parameters_text)
        result.update(global_parameters)

    phases_results = re.findall(
        r"Local parameters and GOALs for phase .+?\n(.*?)\n(?:\n|\Z)",
        texts,
        re.DOTALL,
    )

    result["phases_results"] = {
        phase_name: parse_section(phase_result)
        for phase_name, phase_result in zip(phase_names, phases_results)
    }

    # add atomic positions
    for phase_name, phase_result in zip(phase_names, phases_results):
        atom_section = re.search(
            r"Atomic positions for phase .+?\n(-+)\n(.*?)$",
            phase_result,
            re.DOTALL,
        ).group(2)
        result["phases_results"][phase_name]["atom_positions_string"] = atom_section
    return LstResult(**result)


def parse_dia(dia_path: Path, phase_names: list[str]) -> DiaResult:
    """
    Get the results from the .dia file. This file mainly contains curves for the refinement.

    // layout of the scanHeap:
    // [0] = 2theta
    // [1] = iObs
    // [2] = iCalc
    // [3] = iBkgr
    // [4...n] = strucs
    """
    if not dia_path.exists():
        raise FileNotFoundError(f"Cannot find the .dia file from {dia_path}")

    # read first line to get the keys
    dia_text = dia_path.read_text().split("\n")

    raw_data = np.loadtxt(dia_text[1:])
    data = {
        "x": raw_data[:, 0].tolist(),
        "y_obs": raw_data[:, 1].tolist(),
        "y_calc": raw_data[:, 2].tolist(),
        "y_bkg": raw_data[:, 3].tolist(),
        "structs": {
            name: raw_data[:, i + 4].tolist() for i, name in enumerate(phase_names)
        },
    }
    return DiaResult(**data)


def parse_par(par_file: Path, phase_names: list[str]) -> pd.DataFrame:
    """
    Get the parameters from the .par file (hkl).

    Only work for Cu K alpha!!!
    """

    def _make_dataframe(peak_list) -> pd.DataFrame:
        return pd.DataFrame(
            peak_list,
            columns=[
                "2theta",
                "intensity",
                "b1",
                "b2",
                "h",
                "k",
                "l",
                "phase",
                "phase_idx",
            ],
        ).astype(
            {
                "2theta": float,
                "intensity": float,
                "b1": float,
                "b2": float,
                "h": int,
                "k": int,
                "l": int,
                "phase": str,
                "phase_idx": int,
            }
        )

    content = par_file.read_text().split("\n")
    peak_list = []

    if len(content) < 2:
        return _make_dataframe(peak_list)

    peak_num = re.search(r"PEAKZAHL=(\d+)", content[0])

    if not peak_num:
        return _make_dataframe(peak_list)

    # parse some global parameters
    eps1 = re.search(r"EPS1=(\d+(\.\d+)?)", content[0])
    eps2 = re.search(r"EPS2=([+-]?\d+(\.\d+)?)", content[0])
    pol = re.search(r"POL=(\d+(\.\d+)?)", content[0])
    wavelength = re.search(r"LAMBDA=(\S+)", content[0])
    if not wavelength:
        wavelength = re.search(r"SYNCHROTRON=(\S+)", content[0])
    if not wavelength:
        raise ValueError("Cannot find the wavelength from the .par file")

    eps1 = float(eps1.group(1)) if eps1 else 0.0
    eps2 = float(eps2.group(1)) if eps2 else 0.0
    pol = float(pol.group(1)) if pol else 1.0
    wavelength = get_wavelength(wavelength.group(1))
    peak_num = int(peak_num.group(1))

    # get the mapping between the peak's phase name to the actual phase name
    all_peak_phase_names = re.findall(r"PHASE=(\w+)", "\n".join(content))
    peak_phase_names = list(dict.fromkeys(all_peak_phase_names))
    phase_names_mapping = {
        peak_phase_name: (phase_name, i)
        for i, (peak_phase_name, phase_name) in enumerate(
            zip(peak_phase_names, phase_names)
        )
    }

    for i in range(1, peak_num + 1):
        if i >= len(content):
            break

        numbers = re.split(r"\s+", content[i])

        if numbers:
            rp = int(numbers[0])
            intensity = float(numbers[1])
            d_inv = float(numbers[2])
            gsum = re.search(r"GSUM=(\d+(\.\d+)?)", content[i])
            gsum = float(gsum.group(1)) if gsum is not None else 1.0
            intensity = intensity_correction(
                intensity=intensity,
                d_inv=d_inv,
                gsum=gsum,
                wavelength=wavelength,
                pol=pol,
            )
            if rp == 2:
                b1 = 0
                b2 = 0
            elif rp == 3:
                b1 = float(numbers[3])
                b2 = 0
            elif rp == 4:
                b1 = float(numbers[3])
                b2 = float(numbers[4]) ** 2
            else:
                b1 = 0
                b2 = 0

            h = int(numbers[-3])
            k = int(numbers[-2])
            l = int(numbers[-1])  # noqa: E741

            phase = re.search(r"PHASE=(\w+)", content[i]).group(1)
            phase, idx = phase_names_mapping[phase]

            if intensity > 0:
                peak_list.append([d_inv, intensity, b1, b2, h, k, l, phase, idx])

    # from d_inv to two theta
    two_theta = (
        np.arcsin(wavelength * np.array([p[0] for p in peak_list]) / 2)
        * 180
        / np.pi
        * 2
    )

    # apply eps1 and eps2
    two_theta += angular_correction(two_theta, eps1, eps2)
    peak_list = [
        [two_theta[i]] + peak_list[i][1:] for i in range(len(peak_list))  # noqa: RUF005
    ]

    return _make_dataframe(peak_list)
