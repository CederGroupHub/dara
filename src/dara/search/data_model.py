"""Search node data model."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field
from pymatgen.core import Composition

from dara.plot import visualize
from dara.refine import RefinementPhase
from dara.result import RefinementResult
from dara.search.peak_matcher import PeakMatcher
from dara.utils import (
    get_compositional_clusters,
    get_head_of_compositional_cluster,
)


class PeakMatchingStrategy(BaseModel, frozen=True):
    """Coefficients for peak matching score calculation.

    Args:
        matched_coeff: the coefficient of the matched peaks
        wrong_intensity_coeff: the coefficient of the peaks with wrong intensities
        missing_coeff: the coefficient of the missing peaks
        extra_coeff: the coefficient of the extra peaks
    """

    matched_coeff: float
    wrong_intensity_coeff: float
    missing_coeff: float
    extra_coeff: float

    @classmethod
    def from_tuple(
        cls, coeffs: tuple[float, float, float, float]
    ) -> PeakMatchingStrategy:
        """Create from a tuple of (matched_coeff, wrong_intensity_coeff, missing_coeff, extra_coeff)."""
        return cls(
            matched_coeff=coeffs[0],
            wrong_intensity_coeff=coeffs[1],
            missing_coeff=coeffs[2],
            extra_coeff=coeffs[3],
        )

    @classmethod
    def default(cls) -> PeakMatchingStrategy:
        """Return the default strategy used in search_phases."""
        return cls(
            matched_coeff=1.0,
            wrong_intensity_coeff=1.0,
            missing_coeff=-0.05,
            extra_coeff=-0.5,
        )

    @classmethod
    def default_tree(cls) -> PeakMatchingStrategy:
        """Return the default strategy used in the search tree."""
        return cls(
            matched_coeff=1.0,
            wrong_intensity_coeff=1.0,
            missing_coeff=-0.01,
            extra_coeff=-1.0,
        )

    def as_kwargs(self) -> dict[str, float]:
        """Return the coefficients as keyword arguments for PeakMatcher.score()."""
        return {
            "matched_coeff": self.matched_coeff,
            "wrong_intensity_coeff": self.wrong_intensity_coeff,
            "missing_coeff": self.missing_coeff,
            "extra_coeff": self.extra_coeff,
        }


class SearchNodeData(BaseModel):
    current_result: Optional[RefinementResult]
    current_phases: list[RefinementPhase]

    group_id: int = Field(default=-1, ge=-1)
    fom: float = Field(default=0, ge=0)
    lattice_strain: float = Field(default=0)

    status: Literal[
        "pending",
        "max_depth",
        "error",
        "no_improvement",
        "running",
        "expanded",
        "similar_structure",
        "low_weight_fraction",
    ] = "pending"

    isolated_missing_peaks: Optional[list[list[float]]] = None
    isolated_extra_peaks: Optional[list[list[float]]] = None

    peak_matcher_scores: Optional[dict[RefinementPhase, list[float]]] = None
    peak_matcher_threshold: Optional[float] = None

    @property
    def pretty_output(self):
        is_root_node = self.current_result is None and self.status == "expanded"

        if is_root_node:
            phase_string = "Root "
            if len(self.current_phases) > 1:
                phase_string += (
                    f"({', '.join([p.path.stem for p in self.current_phases])}) "
                )
            return phase_string

        import colorful as cf

        status_color = {
            "max_depth": cf.blue,
            "error": cf.red,
            "expanded": cf.green,
        }
        status_str = status_color.get(self.status, lambda x: x)(self.status)
        phase_string = f"({status_str}) "
        phase_string += self.current_phases[-1].path.stem
        total_len = 3 + len(self.status) + len(self.current_phases[-1].path.stem)

        phase_string += " " * max(60 - total_len, 0)

        return f"{phase_string}" + (
            f"Rwp: {self.current_result.lst_data.rwp if self.current_result is not None else 1 * 100:.2f}% | "
            f"Strain: {round(self.lattice_strain * 100, 2)}% | "
            f"Group {self.group_id}  "
            if self.current_result is not None
            else ""
        )

    def get_peak_matcher(self, peak_obs: np.ndarray, *args, **kwargs) -> PeakMatcher:
        return PeakMatcher(
            *args,
            peak_obs=peak_obs,
            peak_calc=self.current_result.peak_data[["2theta", "intensity"]].values,
            **kwargs,
        )


class SearchResult(BaseModel):
    refinement_result: RefinementResult
    phases: tuple[tuple[RefinementPhase, ...], ...]
    foms: tuple[tuple[float, ...], ...]
    lattice_strains: tuple[tuple[float, ...], ...]
    missing_peaks: Optional[list[list[float]]]
    extra_peaks: Optional[list[list[float]]]

    @property
    def grouped_phases(
        self,
    ) -> list[list[tuple[Composition, list[Path]]]]:
        """Group the phases based on their composition."""
        grouped_phases = []
        for phases in self.phases:
            grouped_phase = get_compositional_clusters([p.path for p in phases])
            grouped_phase_with_head = [
                (get_head_of_compositional_cluster(cluster), cluster)
                for cluster in grouped_phase
            ]
            grouped_phases.append(grouped_phase_with_head)
        return grouped_phases

    def visualize(self, diff_offset: bool = False):
        return visualize(
            result=self.refinement_result,
            diff_offset=diff_offset,
            missing_peaks=self.missing_peaks,
            extra_peaks=self.extra_peaks,
        )
