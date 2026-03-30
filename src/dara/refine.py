"""Perform refinements with BGMN."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dara.bgmn_worker import BGMNWorker
from dara.cif2str import cif2str
from dara.generate_control_file import generate_control_file
from dara.result import RefinementResult, get_result
from dara.xrd import convert_pattern_to_xy


class RefinementPhase(BaseModel, frozen=True):
    """
    Input phase for refinement.

    Contains the path to the phase file and the specific parameters for the phase.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    path: Path = Field(..., description="The path to the phase file.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        kw_only=True,
        description="The specific parameters for the phase.",
    )

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, v):
        return Path(v)

    def __hash__(self):
        return hash(self.path)

    def __eq__(self, other: RefinementPhase):
        return self.path == other.path

    @classmethod
    def make(cls, path_obj: RefinementPhase | Path | str) -> RefinementPhase:
        """
        Make an RefinementPhase object from a path object. If the path object is already an
        RefinementPhase object, return it.
        If the path object is a string or Path object, create an RefinementPhase object
        with the path object with no specific parameters (the default parameters will be used).

        Args:
            path_obj: the path object, can be a string, Path object, or RefinementPhase object.

        Returns
        -------
            RefinementPhase object
        """
        return (
            path_obj
            if isinstance(path_obj, RefinementPhase)
            else RefinementPhase(path=Path(path_obj))
        )


def do_refinement(
    pattern_path: Path | str,
    phases: list[RefinementPhase | Path | str],
    wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float = "Cu",
    instrument_profile: str | Path = "Aeris-fds-Pixcel1d-Medipix3",
    working_dir: Path | str | None = None,
    phase_params: dict | None = None,
    refinement_params: dict | None = None,
    show_progress: bool = False,
) -> RefinementResult:
    """Refine the structure using BGMN."""
    pattern_path = Path(pattern_path)
    working_dir = (
        Path(working_dir)
        if working_dir is not None
        else pattern_path.parent / f"refinement_{pattern_path.stem}"
    )

    if not working_dir.exists():
        working_dir.mkdir(exist_ok=True, parents=True)

    if phase_params is None:
        phase_params = {}

    if refinement_params is None:
        refinement_params = {}

    if pattern_path.suffix.lower() not in (".xy",):
        pattern_path = convert_pattern_to_xy(pattern_path, working_dir)

    str_paths = []
    for phase_path in phases:
        phase = RefinementPhase.make(phase_path)
        phase_path_ = phase.path
        phase_params_ = phase_params.copy()
        # Update the default phase parameters with the specific parameters for the phase
        phase_params_.update(phase.params)
        if phase_path_.suffix == ".cif":
            str_path = cif2str(phase_path_, "", working_dir, **phase_params_)
        else:
            if phase_path_.parent != working_dir:
                shutil.copy(phase_path_, working_dir)
            str_path = working_dir / phase_path_.name
        str_paths.append(str_path)

    control_file_path = generate_control_file(
        pattern_path=pattern_path,
        str_paths=str_paths,
        instrument_profile=instrument_profile,
        working_dir=working_dir,
        wavelength=wavelength,
        **refinement_params,
    )

    bgmn_worker = BGMNWorker()
    bgmn_worker.run_refinement_cmd(control_file_path, show_progress=show_progress)
    return get_result(control_file_path)


def do_refinement_no_saving(
    pattern_path: Path,
    phases: list[RefinementPhase | Path | str],
    wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float = "Cu",
    instrument_profile: str | Path = "Aeris-fds-Pixcel1d-Medipix3",
    phase_params: dict | None = None,
    refinement_params: dict | None = None,
    show_progress: bool = False,
) -> RefinementResult:
    """Refine the structure using BGMN in a temporary directory without saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        working_dir = Path(tmpdir)

        return do_refinement(
            pattern_path=pattern_path,
            phases=phases,
            wavelength=wavelength,
            instrument_profile=instrument_profile,
            working_dir=working_dir,
            phase_params=phase_params,
            refinement_params=refinement_params,
            show_progress=show_progress,
        )
