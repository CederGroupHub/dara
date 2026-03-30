"""Utility functions for the dara package."""

from __future__ import annotations

import itertools
import logging
import os
import random
import re
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
from monty.json import MontyDecoder
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure
from scipy import signal
from sklearn.cluster import AgglomerativeClustering

if TYPE_CHECKING:
    import pandas as pd
    from maggma.stores.mongolike import MongoStore


DEPRECATED = "DEPRECATED"

with open(Path(__file__).parent / "data" / "possible_species.txt") as f:
    POSSIBLE_SPECIES = {sp.strip() for sp in f}


def process_phase_name(phase_name: str) -> str:
    """Process the phase name to remove special characters."""
    processed_name = re.sub(r"[\s()_/\\+–\-*.]", "", phase_name)
    if processed_name.isdigit():
        processed_name = f"P{processed_name}"
    # if empty, return UnknownPhase + some random alphabet
    if not processed_name:
        processed_name = "P" + "".join(
            random.choices(
                list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
                k=5,
            )
        )

    return processed_name


def bool2yn(value: bool) -> str:
    """Convert boolean to Y (yes) or N (no)."""
    return "Y" if value else "N"


def get_number(s: Union[float, None, tuple[float, float]]) -> Union[float, None]:
    """Get the number from a float or tuple of floats."""
    if isinstance(s, tuple):
        return s[0]
    return s


def load_symmetrized_structure(
    cif_path: Path,
) -> tuple[SymmetrizedStructure, SpacegroupAnalyzer]:
    """Load the symmetrized structure from a CIF file. This function will symmetrize the
    structure and provide a spacegroup analyzer object as well.
    """
    with warnings.catch_warnings():  # suppress the warnings from pymatgen
        warnings.filterwarnings("ignore")
        try:
            structure = SpacegroupAnalyzer(
                Structure.from_file(
                    cif_path.as_posix(), site_tolerance=1e-3, occupancy_tolerance=100
                )
            ).get_refined_structure()
            spg = SpacegroupAnalyzer(structure)
            symmetrized_structure: SymmetrizedStructure = (
                spg.get_symmetrized_structure()
            )
        except Exception:  # try with a higher site tolerance
            structure = SpacegroupAnalyzer(
                Structure.from_file(
                    cif_path.as_posix(), site_tolerance=1e-2, occupancy_tolerance=100
                )
            ).get_refined_structure()
            spg = SpacegroupAnalyzer(structure)
            symmetrized_structure = spg.get_symmetrized_structure()

    return symmetrized_structure, spg


def get_optimal_max_two_theta(
    peak_data: pd.DataFrame,
    fraction: float = 0.7,
    intensity_filter=0.1,
) -> float:
    """Get the optimal 2theta max given detected peaks. The range is determined by
    proportion of the detected peaks.

    Args:
        fraction: The fraction of the detected peaks. Defaults to 0.7.
        intensity_filter: The intensity filter; the fraction of the max intensity that
            is required for a peak to be acknowledged in this analysis. Defaults to 0.1.
        min_threshold: The minimum threshold for the 2theta range. Defaults to 50. If set
            to None, the minimum threshold will be the 2theta of the last peak.

    Returns
    -------
        A tuple of the optimal 2theta range.
    """
    max_angle = peak_data["2theta"].max()
    min_angle = peak_data["2theta"].min()
    angle_threshold = 0.4 * (max_angle - min_angle) + min_angle
    max_intensity = peak_data.intensity.max()
    peak_data = peak_data[peak_data.intensity > intensity_filter * max_intensity]
    peak_data = peak_data.sort_values("2theta")
    peak_data = peak_data.reset_index(drop=True)

    num_peaks = len(peak_data)
    end_idx = round(fraction * num_peaks) - 1
    if end_idx < 0:
        end_idx = 0

    buffer = 1  # include the full last peak

    threshold = round(peak_data["2theta"].iloc[end_idx], 2) + buffer

    return max(angle_threshold, threshold)


def read_phase_name_from_str(str_path: Path) -> str:
    """Get the phase name from the str file path.

    Example of str:
    PHASE=BaSnO3 // generated from pymatgen
    FORMULA=BaSnO3 //
    Lattice=Cubic HermannMauguin=P4/m-32/m Setting=1 SpacegroupNo=221 //
    PARAM=A=0.41168_0.40756^0.41580 //
    RP=4 PARAM=k1=0_0^1 k2=0 PARAM=B1=0_0^0.01 PARAM=GEWICHT=0_0 //
    GOAL:BaSnO3=GEWICHT //
    GOAL=GrainSize(1,1,1) //
    E=BA+2 Wyckoff=b x=0.500000 y=0.500000 z=0.500000 TDS=0.010000
    E=SN+4 Wyckoff=a x=0.000000 y=0.000000 z=0.000000 TDS=0.010000
    E=O-2 Wyckoff=d x=0.000000 y=0.000000 z=0.500000 TDS=0.010000
    """
    text = str_path.read_text()
    try:
        return re.search(r"PHASE=(\S*)", text).group(1)
    except AttributeError as e:
        raise ValueError(
            f"Could not find phase name in {str_path}. The content is: {text}"
        ) from e


def standardize_coords(x, y, z):
    # Adjust coordinates to specific fractional values if close
    fractions = {
        0.3333: 1 / 3,
        0.6667: 2 / 3,
        0.1667: 1 / 6,
        0.8333: 5 / 6,
        0.0833: 1 / 12,
        0.4167: 5 / 12,
        0.5833: 7 / 12,
        0.9167: 11 / 12,
    }

    for key, value in fractions.items():
        if abs(x - key) < 0.0001:
            x = value
        if abs(y - key) < 0.0001:
            y = value
        if abs(z - key) < 0.0001:
            z = value

    return x, y, z


def fuzzy_compare(a: float, b: float):
    fa = round(a, 6)
    fb = round(b, 6)

    # Normalizing the fractional parts to be within [0, 1]
    while fa < 0.0:
        fa += 1.0
    while fb < 0.0:
        fb += 1.0
    while fa >= 1.0:
        fa -= 1.0
    while fb >= 1.0:
        fb -= 1.0

    # Checking specific fractional values
    fractions = [
        (0.3333, 0.3334),  # 1/3
        (0.6666, 0.6667),  # 2/3
        (0.1666, 0.1667),  # 1/6
        (0.8333, 0.8334),  # 5/6
        (0.0833, 0.0834),  # 1/12
        (0.4166, 0.4167),  # 5/12
        (0.5833, 0.5834),  # 7/12
        (0.9166, 0.9167),  # 11/12
    ]

    for lower, upper in fractions:
        if lower <= fa <= upper and lower <= fb <= upper:
            return True

    # Fuzzy comparison for general case
    def is_close(_a, _b, rel_tol=0, abs_tol=1e-3):
        # Custom implementation of fuzzy comparison
        return abs(_a - _b) <= max(rel_tol * max(abs(_a), abs(_b)), abs_tol)

    return is_close(fa, fb)


def copy_and_rename_files(
    file_map: dict, dest_directory: Path | str, verbose: bool = True
):
    """Copy files (and rename them) into a destination directory using a provided mapping.

    src_directory: Path to the source directory
    dest_directory: Path to the destination directory
    file_map: Dictionary where keys are original filenames and values are new filenames
    """
    # Ensure the destination directory exists
    dest_directory = Path(dest_directory)

    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    # Copy and rename each specified file
    for src_file, dest_filename in file_map.items():
        src_file = Path(src_file)
        dest_file = Path(dest_directory / dest_filename)

        # Check if file exists and is a file (not a directory)
        if os.path.isfile(src_file):
            shutil.copy(src_file, dest_file)
            if verbose:
                print(
                    f"Successfully copied {src_file.name} to {dest_file.name} in {dest_directory}"
                )
        else:
            if verbose:
                print(f"ERROR: File {src_file} not found!")


def get_chemsys_from_formulas(formulas: list[str]):
    """Convert a list of formulas to a chemsys."""
    elements = set()
    for formula in formulas:
        elements.update(Composition(formula).elements)

    return "-".join(sorted([str(e) for e in elements]))


def get_entries_in_chemsys_mp(chemsys: str):
    """Download ComputedStructureEntry objects from Materials Project."""
    try:
        from mp_api.client import MPRester
    except ImportError as e:
        raise ImportError("Please install the mp-api package.") from e

    with MPRester() as mpr:
        return mpr.get_entries_in_chemsys(chemsys)


def get_entries_db(db: MongoStore, chemsys: str):
    """Get entries for a specific chemical system from a database."""
    decoder = MontyDecoder()
    for doc in db.query(criteria={"chemsys": chemsys}, properties=["entry"]):
        yield decoder.process_decoded(doc["entry"])


def get_entries_in_chemsys_db(db: MongoStore, chemsys: list[str] | str):
    """Get all computed entries from a database covering all possible sub-chemical systems.

    This is equivalent to MPRester.get_entries_in_chemsys.

    Args:
        db: the database (must be connected!)
        chemsys: a chemical system, either as a string (e.g., "Li-Fe-O") or as a list of elements.

    """
    if isinstance(chemsys, str):  # noqa: SIM108
        elements = chemsys.split("-")
    else:
        elements = chemsys

    elements_set = set(elements)  # remove duplicate elements

    entries = []
    for i in range(len(elements_set)):
        for els in itertools.combinations(elements_set, i + 1):
            entries.extend(get_entries_db(db, "-".join(sorted(els))))

    return entries


def angular_correction(tt, eps1, eps2):
    deps1 = -eps1 * 360.0 / np.pi
    deps2 = 2.0 * (-eps2 * np.cos(tt * np.pi / 360)) * 180.0 / np.pi
    # deps3 = -eps3 * np.sin(np.deg2rad(tt)) * 180.0 / np.pi

    return deps1 + deps2  # + deps3


def intensity_correction(
    intensity: float, d_inv: float, gsum: float, wavelength: float, pol: float = 1
):
    """
    Translated from Profex source (bgmnparparser.cpp:L112)

    Args:
        intensity: the intensity of the peak
        gsum: the gsum of the peak
        d_inv: the inverse of the d-spacing
        wavelength: the wavelength of the X-ray
        pol: the polarization factor, defaults to 1

    Returns
    -------
        the corrected intensity
    """
    # double sinx2 = std::pow(0.5 * dinv * pl.waveLength, 2.0);
    sinx2 = (0.5 * d_inv * wavelength) ** 2
    # double intens = gsum * 360.0 * intens * 0.5 / (M_PI * std::sqrt(1.0 - sinx2) / pl.waveLength);
    # if (pl.polarization > 0.0) intens *= (0.5 * (1.0 + pl.polarization * std::pow(1.0 - 2.0 * sinx2, 2.0)));
    intensity = (
        gsum * 360.0 * intensity * 0.5 / (np.pi * np.sqrt(1.0 - sinx2) / wavelength)
    )
    if pol > 0.0:
        intensity *= 0.5 * (1.0 + pol * (1.0 - 2.0 * sinx2) ** 2.0)

    return intensity


def rwp(y_calc: np.ndarray, y_obs: np.ndarray) -> float:
    """
    Calculate the Rietveld weighted profile (RWP) for a refinement.

    The result is in percentage.

    Args:
        y_calc: the calculated intensity
        y_obs: the observed intensity
    Returns:
        the RWP
    """
    y_calc = np.array(y_calc)
    y_obs = np.array(y_obs)
    y_obs = np.clip(y_obs, 1e-6, None)
    return np.sqrt(np.sum((y_calc - y_obs) ** 2 / y_obs) / np.sum(y_obs)) * 100


def rpb(y_calc: np.ndarray, y_obs: np.ndarray, y_bkg) -> float:
    """
    Calculate the Rietveld profile without background (RPB) for a refinement.

    The result is in percentage.

    Args:
        y_calc: the calculated intensity
        y_obs: the observed intensity
    Returns:
        the RPB
    """
    y_calc = np.array(y_calc)
    y_obs = np.array(y_obs)
    return np.sum(np.abs(y_calc - y_obs)) / np.sum(np.abs(y_obs - y_bkg)) * 100


def get_logger(
    name: str,
    level=logging.DEBUG,
    log_format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
):
    """Code borrowed from the atomate package.

    Helper method for acquiring logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(log_format)

    if logger.hasHandlers():
        logger.handlers.clear()

    sh = logging.StreamHandler(stream=stream)
    sh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.propagate = False

    return logger


def datetime_str() -> str:
    """Get a string representation of the current time."""
    return str(datetime.utcnow())


def find_optimal_score_threshold(
    scores: list[float] | np.ndarray,
) -> tuple[float, np.ndarray]:
    """Find the inflection point from a list of scores. We will calculate the percentile first."""
    if len(scores) == 0:
        return 0.0, np.array([]).reshape(-1)

    scores = np.array(scores)
    score_percentile = np.percentile(scores, np.arange(0, 101))
    score_percentile = signal.savgol_filter(score_percentile, 5, 1)

    second_derivative = np.diff(score_percentile, n=2)
    threshold = score_percentile[np.argmax(second_derivative)].item()

    # add a small tolerance to the threshold
    threshold -= 0.01
    return threshold, score_percentile


def find_optimal_intensity_threshold(
    intensities: list[float] | np.ndarray, percentile: float = 90
) -> float:
    """
    Find the intensity threshold that captures percentile% of the intensities.

    Args:
        intensities: the list of intensities
        percentile: the percentile to capture, defaults to 90

    Returns
    -------
        the intensity threshold
    """
    if len(intensities) == 0:
        return 0.0
    intensities_ = np.sort(intensities)[::-1]
    intensities_cum = np.cumsum(intensities_)
    intensities_cum /= intensities_cum[-1]
    idx = np.argmax(intensities_cum >= percentile / 100)
    return intensities_[idx]


def get_composition_from_filename(file_name: str | Path) -> Composition:
    """
    Get the composition from the filename. The composition is assumed to be the first
    part of the filename. For example, "BaSnO3_01.xrdml" will return "BaSnO3".
    """
    if isinstance(file_name, str):
        file_name = Path(file_name)

    return Composition(file_name.name.split("_")[0])


def get_composition_distance(
    comp1: Composition | str, comp2: Composition | str, order: int = 2
) -> float:
    """
    Calculate the distance between two compositions.

    The default is the Manhattan.
    """
    comp1 = Composition(comp1, allow_negative=True).fractional_composition
    comp2 = Composition(comp2, allow_negative=True).fractional_composition

    delta_composition = comp1 - comp2
    delta_composition = {
        k: v / (comp1[k] + comp2[k]) for k, v in delta_composition.items()
    }

    return np.linalg.norm(np.array(list(delta_composition.values())), ord=order)


def compositions_to_array(compositions: list[str] | list[Composition]):
    """Convert a list of compositions/formulas to an array of their fractional
    elemental components.
    """
    comps = [Composition(c).fractional_composition for c in compositions]
    elems = sorted({e for comp in comps for e in comp.elements})
    arr = np.zeros((len(compositions), len(elems)))
    for idx, comp in enumerate(comps):
        vec = [comp[el] for el in elems]
        arr[idx] = vec
    return arr


def get_compositional_clusters(
    paths: list[Path | str], distance_threshold: float = 0.1
) -> list[list[Path | str]]:
    """Get similar clusters of compositions based on their compositional similarity.
    Uses AgglomerativeClustering with a distance threshold of 0.1.
    """
    if len(paths) == 0:
        return []
    if len(paths) == 1:
        return [[paths[0]]]

    compositions = [get_composition_from_filename(p) for p in paths]
    raw_clusters = AgglomerativeClustering(
        None, distance_threshold=distance_threshold
    ).fit_predict(compositions_to_array(compositions))
    clusters: list[list[Path]] = [[] for _ in range(len(set(raw_clusters)))]
    for c, path in zip(raw_clusters, paths):
        clusters[c].append(path)

    return clusters


def get_head_of_compositional_cluster(paths: list[str | Path]) -> Composition:
    """Get head of a compositional cluster. This returns the closest stoichiometric
    composition to the average composition. If no stoichiometric composition is found,
    then the nonstoichiometric composition with the smallest distance to the average composition is returned.
    """
    compositions = [get_composition_from_filename(p) for p in paths]
    frac_comps = [
        Composition(c, allow_negative=True).fractional_composition for c in compositions
    ]
    comp_sum = Composition(allow_negative=True)
    for comp in frac_comps:
        comp_sum += comp

    mean = comp_sum / len(frac_comps)

    diffs = {}
    for comp, frac_comp in zip(compositions, frac_comps):
        if comp in diffs:
            continue
        diffs[comp] = sum(abs(i) for i in (frac_comp - mean).values())

    sorted_comps = sorted(compositions, key=lambda i: diffs[i])
    for comp in sorted_comps:
        if all(
            v.is_integer() for v in Composition(comp).values()
        ):  # prefer stoichiometric always
            return comp

    return sorted_comps[0]


def get_wavelength(wavelength_or_target_metal: float | str) -> float:
    element_data = {
        "cu": (1.540598, 1.544426, 1.392250),
        "cr": (2.289760, 2.293663, 2.084920),
        "fe": (1.936042, 1.93998, 1.75661),
        "co": (1.789010, 1.792900, 1.620830),
        "ni": (1.65791, 1.661747, 1.48862),
        "mo": (0.709319, 0.713609, 0.632305),
        "ag": (0.5594075, 0.563798, 0.497069),
        "w": (0.20901, 0.213828, 0.184374),
    }
    try:
        wavelength_or_target_metal = float(wavelength_or_target_metal)
    except ValueError:
        pass
    if isinstance(wavelength_or_target_metal, str):
        if wavelength_or_target_metal.lower() in element_data:
            return (
                # convert to nm
                element_data[wavelength_or_target_metal.lower()][0]
                / 10
            )
        raise ValueError(
            f"Invalid target metal: {wavelength_or_target_metal}. "
            "Please choose from 'Cu', 'Co', 'Cr', 'Fe', 'Mo'."
        )
    return wavelength_or_target_metal


def estimate_snr(intensities: np.ndarray, window: int = 80) -> float:
    """Estimate the signal-to-noise ratio (SNR) of an XRD pattern in dB.

    Uses a rolling-ball approach (minimum filter + smoothing) to estimate the
    background, then computes SNR = 20 * log10(max_signal / noise_std).

    Args:
        intensities: the raw intensity array of the XRD pattern.
        window: the window size for the minimum/uniform filters. Defaults to 80.

    Returns:
        the estimated SNR in dB.
    """
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    bg = minimum_filter1d(intensities, size=window)
    bg = uniform_filter1d(bg, size=window)
    residual = intensities - bg

    threshold = np.percentile(residual, 50)
    noise_mask = residual < threshold
    noise_std = np.std(residual[noise_mask]) if np.any(noise_mask) else np.std(residual)

    max_signal = np.max(residual)
    if noise_std == 0:
        noise_std = 1e-10

    return float(20 * np.log10(max_signal / noise_std))


def estimate_rpb_threshold(
    intensities: np.ndarray,
    center: float = 2.0,
    low: float = 1.0,
    high: float = 3.0,
    snr_low: float = 41.6,
    snr_high: float = 52.7,
) -> float:
    """Estimate an appropriate rpb_threshold based on the SNR of an XRD pattern.

    The mapping is linear between ``snr_low`` and ``snr_high`` (the IQR from
    empirical analysis of ~4500 patterns, 41.6–52.7 dB).  Higher SNR patterns
    get a stricter (lower) threshold, while noisier patterns get a more lenient
    (higher) threshold.  Values outside the IQR are clamped to ``[low, high]``.

    Default output range is [1, 3] centered at 2.

    Args:
        intensities: the raw intensity array of the XRD pattern.
        center: the expected (center) rpb_threshold value. Defaults to 2.0.
        low: the minimum rpb_threshold (returned for high-SNR patterns). Defaults to 1.0.
        high: the maximum rpb_threshold (returned for low-SNR patterns). Defaults to 3.0.
        snr_low: the SNR (dB) that maps to ``high`` rpb_threshold (Q1). Defaults to 41.6.
        snr_high: the SNR (dB) that maps to ``low`` rpb_threshold (Q3). Defaults to 52.7.

    Returns:
        the estimated rpb_threshold, clamped to [low, high].
    """
    snr_db = estimate_snr(intensities)

    # Linear interpolation: high SNR → low threshold, low SNR → high threshold
    snr_center = (snr_low + snr_high) / 2.0
    half_range = (snr_high - snr_low) / 2.0
    threshold_half_range = (high - low) / 2.0

    if half_range == 0:
        return center

    rpb_thresh = center - (snr_db - snr_center) / half_range * threshold_half_range
    return float(np.clip(rpb_thresh, low, high))


def parse_refinement_param(
    refinement_param: str | float,
) -> tuple[str | float, float | None, float | None]:
    if isinstance(refinement_param, float):
        return refinement_param, None, None
    if refinement_param == "fixed":
        return "fixed", None, None
    match = re.match(
        r"([-+]?\d*\.?\d+)_([-+]?\d*\.?\d+)\^([-+]?\d*\.?\d+)", refinement_param
    )
    if match:
        initial = float(match.group(1))
        lower = float(match.group(2))
        upper = float(match.group(3))
        return initial, lower, upper
    raise ValueError(f"Invalid refinement parameter: {refinement_param}")
