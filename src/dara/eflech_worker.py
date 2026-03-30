from __future__ import annotations

import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pybaselines as pb
from sklearn.cluster import AgglomerativeClustering

from dara.bgmn.download_bgmn import download_bgmn
from dara.generate_control_file import (
    copy_instrument_files,
    trim_pattern,
)
from dara.utils import get_logger, get_wavelength, intensity_correction
from dara.xrd import convert_pattern_to_xy

logger = get_logger(__name__)


class EflechWorker:
    """Functionality for running peak detection using BGMN's eflech and teil executables."""

    def __init__(self):
        self.bgmn_folder = (Path(__file__).parent / "bgmn" / "BGMNwin").absolute()

        self.eflech_path = self.bgmn_folder / "eflech"
        self.teil_path = self.bgmn_folder / "teil"

        if (
            not self.eflech_path.exists()
            and not self.eflech_path.with_suffix(".exe").exists()
        ):
            logger.warning("BGMN executable not found. Downloading BGMN.")
            download_bgmn()

        os.environ["EFLECH"] = self.bgmn_folder.as_posix()
        os.environ["PATH"] += os.pathsep + self.bgmn_folder.as_posix()

    def run_peak_detection(
        self,
        pattern: Path | np.ndarray | str,
        wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float = "Cu",
        instrument_profile: str | Path = "Aeris-fds-Pixcel1d-Medipix3",
        show_progress: bool = False,
        *,
        wmin: float = None,
        wmax: float = None,
        epsilon: float = None,
        possible_changes: str = None,
        nthreads: int = 8,
        timeout: int = 1800,
    ) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            instrument_name = copy_instrument_files(instrument_profile, temp_dir)
            if isinstance(pattern, np.ndarray):
                pattern_path_temp = temp_dir / "temp.xy"
                np.savetxt(pattern_path_temp.as_posix(), pattern, fmt="%.6f")
            else:
                if isinstance(pattern, str):
                    pattern = Path(pattern)
                pattern_path_temp = convert_pattern_to_xy(pattern, temp_dir)

            xy_content = np.loadtxt(pattern_path_temp, dtype=float)
            xy_content = trim_pattern(xy_content)
            np.savetxt(pattern_path_temp, xy_content, fmt="%.6f")

            control_file_path = self.generate_control_file(
                pattern_path_temp,
                wavelength=wavelength,
                instrument_name=instrument_name,
                wmin=wmin,
                wmax=wmax,
                epsilon=epsilon,
                possible_changes=possible_changes,
                nthreads=nthreads,
            )

            teil_output = self.run_eflech(
                control_file_path,
                mode="teil",
                working_dir=temp_dir,
                show_progress=False,  # we need the output for further processing
                timeout=timeout,
            )

            ru = re.search(r"RU=(\d+)", teil_output)
            if ru:
                ru = int(ru.group(1))
            else:
                raise RuntimeError(
                    "RU value not found in teil output. The output is: \n" + teil_output
                )
            control_file_content = control_file_path.read_text()
            all_wmin2 = re.findall(r"WMIN2\[\d+]==(.*?)\n", control_file_content)
            all_wmax2 = re.findall(r"WMAX2\[\d+]==(.*?)\n", control_file_content)
            all_wmin2 = [float(wmin) for wmin in all_wmin2]
            all_wmax2 = [float(wmax) for wmax in all_wmax2]

            two_theta_range = xy_content[-1, 0] - xy_content[0, 0]

            # if teil cannot find a good split of the pattern, we will try to split it manually
            if (
                not all_wmin2
                or not all_wmax2
                or any(
                    wmax - wmin > 0.5 * two_theta_range
                    for wmin, wmax in zip(all_wmin2, all_wmax2)
                )
            ):
                warnings.warn(
                    "teil cannot find a good split of the pattern. Trying to split it with dara-teil."
                )
                self.patch_control_file_after_teil(control_file_path, ru, xy_content)

            self.run_eflech(
                control_file_path,
                mode="eflech",
                working_dir=temp_dir,
                show_progress=show_progress,
                timeout=timeout,
            )

            return self.parse_peak_list(temp_dir, wavelength=wavelength)

    @staticmethod
    def generate_control_file(
        pattern_path: Path,
        wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float,
        instrument_name: str,
        *,
        wmin: float = None,
        wmax: float = None,
        possible_changes: str = None,
        epsilon: float = None,
        nthreads: int = None,
    ) -> Path:
        control_file_str = f"""
            VERZERR={instrument_name}.geq
            {f"LAMBDA={wavelength.upper()}" if isinstance(wavelength, str) else f"SYNCHROTRON={wavelength:.4f}"}
            % Measured data
            VAL[1]={pattern_path.name}
            {f"WMIN={wmin}" if wmin is not None else ""}
            {f"WMAX={wmax}" if wmax is not None else ""}
            {f"EPSILON={epsilon}" if epsilon is not None else ""}
            NTHREADS={nthreads if nthreads is not None else os.cpu_count()}
            {f"TEST={possible_changes}" if possible_changes is not None else ""}
            OUTPUTMASK=output-$
            TITELMASK=output-$"""

        control_file_str = "\n".join(
            [line.strip() for line in control_file_str.split("\n")]
        )
        control_file_path = pattern_path.parent / "control.sav"

        with control_file_path.open("w") as f:
            f.write(control_file_str)

        return control_file_path

    def run_eflech(
        self,
        control_file_path: Path,
        mode: Literal["eflech", "teil"],
        working_dir: Path,
        show_progress: bool = False,
        timeout: int = 1800,
    ) -> str | None:
        if mode == "eflech":
            cp = subprocess.run(
                [self.eflech_path.as_posix(), control_file_path.as_posix()],
                cwd=working_dir.as_posix(),
                capture_output=not show_progress,
                timeout=timeout,
                check=False,
            )
        elif mode == "teil":
            if show_progress:
                warnings.warn("show_progress=True is not supported for teil mode.")
            cp = subprocess.run(
                [self.teil_path.as_posix(), control_file_path.as_posix()],
                cwd=working_dir.as_posix(),
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if cp.returncode:
            raise RuntimeError(
                f"Error in BGMN {mode} for {control_file_path}. The exit code is {cp.returncode}\n"
                f"{cp.stdout}\n"
                f"{cp.stderr}"
            )
        if cp.stdout is not None:
            return cp.stdout.decode()
        return None

    def parse_peak_list(
        self,
        par_folder: Path,
        wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float,
    ) -> pd.DataFrame:
        all_par_files = list(par_folder.glob("output-*.par"))
        peak_list = []
        wavelength_float = get_wavelength(wavelength)
        for par_file in all_par_files:
            peak_list.extend(self.parse_par_file(par_file, wavelength=wavelength_float))

        peak_list = np.array(peak_list).reshape(-1, 4)

        d_inv = peak_list[:, 0]
        intensity = peak_list[:, 1]
        b1 = peak_list[:, 2]
        b2 = peak_list[:, 3]

        two_theta = np.arcsin(wavelength_float * d_inv / 2) * 180 / np.pi * 2

        peak_list_two_theta = np.column_stack((two_theta, intensity, b1, b2))
        peak_list_two_theta = peak_list_two_theta[peak_list_two_theta[:, 0].argsort()]

        return pd.DataFrame(
            peak_list_two_theta, columns=["2theta", "intensity", "b1", "b2"]
        ).astype(float)

    @staticmethod
    def parse_par_file(par_file: Path, wavelength: float) -> list[list[float]]:
        content = par_file.read_text().split("\n")
        peak_list = []

        if len(content) < 2:
            return peak_list

        peak_num = re.search(r"PEAKZAHL=(\d+)", content[0])
        pol = re.search(r"POL=(\d+(\.\d+)?)", content[0])
        pol = float(pol.group(1)) if pol else 1.0

        if not peak_num:
            return peak_list

        peak_num = int(peak_num.group(1))

        if peak_num == 0:
            return peak_list

        for i in range(1, peak_num + 1):
            if i >= len(content):
                break

            numbers = re.split(r"\s+", content[i])

            if numbers:
                rp = int(numbers[0])
                intensity = float(numbers[1])
                d_inv = float(numbers[2])
                gsum = (
                    1.0
                    if (gsum := re.search("GSUM=(\\d+(\\.\\d+)?)", content[i])) is None
                    else float(gsum.group(1))
                )
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

                # Only add peaks with intensity > 0
                if intensity > 0:
                    peak_list.append([d_inv, intensity, b1, b2])

        return peak_list

    def patch_control_file_after_teil(
        self, control_file_path: Path, ru: int, xy_content: np.ndarray
    ):
        """
        Patch the control file after teil is run. If the divided pattern is still too large,
        we will divide it further to save the computation time.

        Args:
            control_file_path: the path to the control file
            ru: the degree of background, get from teil output
            xy_content: the xy of the pattern
        """
        control_file_content = control_file_path.read_text()

        val_name = re.search(r"VAL\[1]=(\S+)", control_file_content).group(1)
        output_mask = re.search(r"OUTPUTMASK=(\S+)", control_file_content).group(1)
        titel_mask = re.search(r"TITELMASK=(\S+)", control_file_content).group(1)

        breakpoints = self.get_background_breakpoints(ru, xy_content)

        partial_pattern_str = "% dara-teil has computed the following angular ranges\n"
        for i, (breakpoint_1, breakpoint_2) in enumerate(
            zip(breakpoints[:-1], breakpoints[1:]), start=1
        ):
            partial_pattern_str += (
                f"% Dara Teil {i}\n"
                f"WMIN[{i}]=={breakpoint_1['lower_bound']:.6f}\n"
                f"WMIN2[{i}]=={breakpoint_1['mean']:.6f}\n"
                f"WMAX[{i}]=={breakpoint_2['upper_bound']:.6f}\n"
                f"WMAX2[{i}]=={breakpoint_2['mean']:.6f}\n"
                f"VAL[{i},1]={val_name}\n"
                f"OUTPUT[{i}]={output_mask.replace('$', str(i))}\n"
                f"TITEL[{i}]={titel_mask.replace('$', str(i))}\n"
            )

        control_file_content = re.sub(
            r"%teil has computed the following angular ranges.*"
            r"(?=\n%these constants have been notated by TEIL for internal use)",
            partial_pattern_str,
            control_file_content,
            flags=re.DOTALL,
        )
        control_file_path.write_text(control_file_content)

    @staticmethod
    def get_background_breakpoints(
        ru: int, xy_content: np.ndarray
    ) -> list[dict[str, float]]:
        """
        Get the background breakpoints from the teil output.

        Args:
            ru: the degree of background, get from teil output
            xy_content: the xy of the pattern

        Returns
        -------
            the background breakpoints
        """
        baseline_fitter = pb.Baseline(xy_content[:, 0])

        bkg = baseline_fitter.penalized_poly(xy_content[:, 1], poly_order=ru)[0]

        # use the negative residuals as an approximation of the background std
        residuals = xy_content[:, 1] - bkg
        bkg_std = residuals[residuals < 0].std()

        # allow a two sigma std (assuming the noise is normally distributed)
        bkg_area = np.arange(len(xy_content))[residuals < bkg_std * 2]

        points_per_deg = int(len(xy_content) / (xy_content[-1, 0] - xy_content[0, 0]))

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=points_per_deg * 0.05,
            linkage="single",
        )
        clustering.fit(bkg_area.reshape(-1, 1))

        # include the start and end points of the whole pattern
        breakpoints = [
            {
                "mean": xy_content[0, 0],
                "lower_bound": xy_content[0, 0],
                "upper_bound": xy_content[0, 0],
            },
            {
                "mean": xy_content[-1, 0],
                "lower_bound": xy_content[-1, 0],
                "upper_bound": xy_content[-1, 0],
            },
        ]
        for i in range(clustering.n_clusters_):
            all_points_in_cluster = xy_content[bkg_area, 0][clustering.labels_ == i]

            if all_points_in_cluster[-1] - all_points_in_cluster[0] < 0.5:
                continue

            breakpoint_ = {
                "mean": all_points_in_cluster.mean(),
                "lower_bound": np.percentile(all_points_in_cluster, 25),
                "upper_bound": np.percentile(all_points_in_cluster, 75),
            }

            breakpoints.append(breakpoint_)

        # sorted by mean
        return sorted(breakpoints, key=lambda x: x["mean"])
