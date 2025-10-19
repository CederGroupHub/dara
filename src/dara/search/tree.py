from __future__ import annotations

import warnings
from itertools import zip_longest
from numbers import Number
from pathlib import Path
from subprocess import TimeoutExpired
from typing import TYPE_CHECKING, Literal

import jenkspy
import numpy as np
import pandas as pd
import ray
from sklearn.cluster import AgglomerativeClustering
from treelib import Node, Tree

from dara import do_refinement_no_saving
from dara.cif2str import CIF2StrError
from dara.peak_detection import detect_peaks
from dara.refine import RefinementPhase
from dara.search.data_model import SearchNodeData, SearchResult
from dara.search.peak_matcher import PeakMatcher
from dara.utils import (
    find_optimal_intensity_threshold,
    find_optimal_score_threshold,
    get_logger,
    get_number,
    get_optimal_max_two_theta,
    load_symmetrized_structure,
    parse_refinement_param,
    rpb,
)

if TYPE_CHECKING:
    from dara.result import RefinementResult


logger = get_logger(__name__, level="INFO")


@ray.remote(num_cpus=1)
def remote_do_refinement_no_saving(
    pattern_path: Path,
    cif_paths: list[Path],
    wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float,
    instrument_profile: str | Path,
    phase_params: dict[str, ...] | None,
    refinement_params: dict[str, float] | None,
) -> RefinementResult | None:
    """
    Perform the actual refinement in the remote process.

    If the refinement fails, None will be returned.
    """
    if len(cif_paths) == 0:
        return None
    try:
        result = do_refinement_no_saving(
            pattern_path,
            cif_paths,
            wavelength=wavelength,
            instrument_profile=instrument_profile,
            phase_params=phase_params,
            refinement_params=refinement_params,
        )
    except (RuntimeError, TimeoutExpired, CIF2StrError) as e:
        logger.debug(f"Refinement failed for {cif_paths}, the reason is {e}")
        return None
    if result.lst_data.rpb == 100:
        logger.debug(f"Refinement failed for {cif_paths}, the reason is RPB = 100.")
        return None
    return result


@ray.remote(num_cpus=1)
def remote_peak_matching(
    batch: list[tuple[np.ndarray, np.ndarray]],
    return_type: Literal["PeakMatcher", "score", "jaccard"],
) -> list[PeakMatcher | float]:
    results = []

    for peak_calc, peak_obs in batch:
        pm = PeakMatcher(peak_calc, peak_obs)

        if return_type == "PeakMatcher":
            results.append(pm)
        elif return_type == "score":
            results.append(pm.score())
        elif return_type == "jaccard":
            results.append(pm.jaccard_index())
        else:
            raise ValueError(f"Unknown return type {return_type}")

    return results


def batch_peak_matching(
    peak_calcs: list[np.ndarray],
    peak_obs: np.ndarray | list[np.ndarray],
    return_type: Literal["PeakMatcher", "score", "jaccard"] = "PeakMatcher",
    batch_size: int = 100,
) -> list[PeakMatcher | float]:
    if isinstance(peak_obs, np.ndarray):
        peak_obs = [peak_obs] * len(peak_calcs)

    if len(peak_calcs) != len(peak_obs):
        raise ValueError("Length of peak_calcs and peak_obs must be the same.")

    all_data = list(zip_longest(peak_calcs, peak_obs, fillvalue=None))
    batches = [
        all_data[i : i + batch_size] for i in range(0, len(all_data), batch_size)
    ]
    handles = [
        remote_peak_matching.remote(batch, return_type=return_type) for batch in batches
    ]
    return sum(ray.get(handles), [])


def batch_refinement(
    pattern_path: Path,
    cif_paths: list[list[RefinementPhase]],
    wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float = "Cu",
    instrument_profile: str | Path = "Aeris-fds-Pixcel1d-Medipix3",
    phase_params: dict[str, ...] | None = None,
    refinement_params: dict[str, float] | None = None,
) -> list[RefinementResult]:
    handles = [
        remote_do_refinement_no_saving.remote(
            pattern_path,
            cif_paths,
            wavelength=wavelength,
            instrument_profile=instrument_profile,
            phase_params=phase_params,
            refinement_params=refinement_params,
        )
        for cif_paths in cif_paths
    ]
    return ray.get(handles)


def calculate_fom_and_strain(
    phase: RefinementPhase, result: RefinementResult
) -> tuple[float, float]:
    """
    Calculate the figure of merit for a phase and lattice strain.

    For more detail, refer to https://journals.iucr.org/j/issues/2019/03/00/nb5231/.
    Args:
        result: the refinement result

    Returns
    -------
        the figure of merit of the target phase. If it cannot be calculated, return 0.
    """
    a = 1.0
    # we disable the weight for now
    b = 0.0
    # we disable the particle size for now
    c = 0.0
    b1_threshold = 2e-5

    phase_path = phase.path

    structure, _ = load_symmetrized_structure(phase_path)
    initial_lattice_abc = structure.lattice.abc

    refined_a = result.lst_data.phases_results[phase_path.stem].a
    refined_b = result.lst_data.phases_results[phase_path.stem].b
    refined_c = result.lst_data.phases_results[phase_path.stem].c

    geweicht = result.lst_data.phases_results[phase_path.stem].gewicht
    geweicht = get_number(geweicht)

    if hasattr(result.lst_data.phases_results[phase_path.stem], "B1"):
        b1 = get_number(result.lst_data.phases_results[phase_path.stem].B1) or 0
    else:
        b1 = 0

    if refined_a is None or geweicht is None:
        return 0, 0

    refined_lattice_abc = [
        refined_a,
        refined_b if refined_b is not None else refined_a,
        refined_c if refined_c is not None else refined_a,
    ]
    refined_lattice_abc = [get_number(x) for x in refined_lattice_abc]

    initial_lattice_abc = np.array(initial_lattice_abc) / 10  # convert to nm
    refined_lattice_abc = np.array(refined_lattice_abc)

    delta_u = (
        np.sum(np.abs(initial_lattice_abc - refined_lattice_abc) / initial_lattice_abc)
        * 100
    )

    lattice_strain = np.mean(
        (refined_lattice_abc - initial_lattice_abc) / initial_lattice_abc
    )

    if b1 is None or b1 < b1_threshold:
        c = 0
    else:
        c /= b1

    return (1 / (result.lst_data.rho + a * delta_u + 1e-4) + b * geweicht) / (
        1 + c
    ), lattice_strain


def group_phases(
    all_phases_result: dict[RefinementPhase, RefinementResult | None],
    distance_threshold: float = 0.1,
) -> dict[RefinementPhase, dict[str, float | int]]:
    """
    Group the phases based on their similarity.

    Args:
        all_phases_result: the result of all the phases
        distance_threshold: the distance threshold for clustering, default to 0.1

    Returns
    -------
        a dictionary containing the group id and the figure of merit for each phase
    """
    grouped_result = {}

    # handle the case where there is no result for a phase
    for phase, result in all_phases_result.items():
        if result is None:
            grouped_result[phase] = {"group_id": -1, "fom": 0, "lattice_strain": 0}

    all_phases_result = {
        phase: result
        for phase, result in all_phases_result.items()
        if result is not None
    }

    if len(all_phases_result) <= 1:
        for phase, result in all_phases_result.items():
            fom, lattice_strain = calculate_fom_and_strain(phase, result)
            grouped_result[phase] = {
                "group_id": 0,
                "fom": fom,
                "lattice_strain": lattice_strain,
            }
        return grouped_result

    peaks = []

    for phase, result in all_phases_result.items():
        all_peaks = result.peak_data
        peaks.append(
            all_peaks[all_peaks["phase"] == phase.path.stem][
                ["2theta", "intensity"]
            ].values
        )

    pairwise_similarity = batch_peak_matching(
        [p for p in peaks for _ in peaks],
        [p for _ in peaks for p in peaks],
        return_type="jaccard",
    )
    distance_matrix = 1 - np.array(pairwise_similarity).reshape(len(peaks), len(peaks))

    # current peak matching algorithm is not a symmetric metric.
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # clustering
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    clusterer.fit(distance_matrix)

    for i, cluster in enumerate(clusterer.labels_):
        phase = list(all_phases_result.keys())[i]
        result = list(all_phases_result.values())[i]
        fom, lattice_strain = calculate_fom_and_strain(phase, result)
        grouped_result[phase] = {
            "group_id": cluster,
            "fom": fom,
            "lattice_strain": lattice_strain,
        }

    return grouped_result


def remove_unnecessary_phases(
    result: RefinementResult, cif_paths: list[Path], rpb_threshold: float = 0.0
) -> list[Path]:
    """
    Remove unnecessary phases from the result.

    If a phase cannot cause increase in RWP, it will be removed.
    """
    phases_results = {k: np.array(v) for k, v in result.plot_data.structs.items()}
    y_obs = np.array(result.plot_data.y_obs)
    y_calc = np.array(result.plot_data.y_calc)
    y_bkg = np.array(result.plot_data.y_bkg)

    cif_paths_dict = {cif_path.stem: cif_path for cif_path in cif_paths}

    original_rpb = rpb(y_calc, y_obs, y_bkg)

    new_phases = []

    for excluded_phase in phases_results:  # noqa: PLC0206
        y_calc_excl = y_calc.copy()
        y_calc_excl -= phases_results[excluded_phase]

        new_rpb = rpb(y_calc_excl, y_obs, y_bkg)

        if new_rpb > original_rpb + rpb_threshold:
            new_phases.append(cif_paths_dict[excluded_phase])

    return new_phases


def get_natural_break_results(
    results: list[SearchResult], sorting: bool = True
) -> list[SearchResult]:
    """Get the natural break results based on (1-rho) value."""
    all_rhos = None

    # remove results that are too bad (dead end in the tree search)
    while all_rhos is None or max(all_rhos) > min(all_rhos) + 5:
        all_rhos = [result.refinement_result.lst_data.rho for result in results]
        if len(set(all_rhos)) > 2:
            # get the first natural break
            interval = jenkspy.jenks_breaks(all_rhos, n_classes=2)
            rho_cutoff = interval[1]
        elif len(set(all_rhos)) == 2 and max(all_rhos) - min(all_rhos) > 10:
            rho_cutoff = min(all_rhos) + 10
        else:
            break
        results = [
            result
            for result in results
            if result.refinement_result.lst_data.rho <= rho_cutoff
        ]
        all_rhos = [result.refinement_result.lst_data.rho for result in results]
    if sorting:
        results = sorted(results, key=lambda x: x.refinement_result.lst_data.rwp)

    return results


class BaseSearchTree(Tree):
    """
    A base class for the search tree. It is not intended to be used directly.

    Args:
        pattern_path: the path to the pattern
        all_phases_result: the result of all the phases
        peak_obs: the observed peaks
        refine_params: the refinement parameters, it will be passed to the refinement function.
        phase_params: the phase parameters, it will be passed to the refinement function.
        intensity_threshold: the intensity threshold to tell if a peak is significant
        instrument_profile: the name/path of the instrument file, it will be passed to the refinement function.
        maximum_grouping_distance: the maximum grouping distance, default to 0.1
        max_phases: the maximum number of phases
        rpb_threshold: the minimum RPB improvement in each step
        pinned_phases: the phases that are pinned and will be included in all the results
    """

    def __init__(
        self,
        pattern_path: Path,
        all_phases_result: dict[RefinementPhase, RefinementResult] | None,
        peak_obs: np.ndarray | None,
        refine_params: dict[str, ...] | None,
        phase_params: dict[str, ...] | None,
        intensity_threshold: float,
        wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float,
        instrument_profile: str | Path,
        express_mode: bool,
        maximum_grouping_distance: float,
        max_phases: float,
        rpb_threshold: float,
        pinned_phases: list[RefinementPhase] | None = None,
        record_peak_matcher_scores: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.pattern_path = pattern_path
        self.rpb_threshold = rpb_threshold
        self.refinement_params = refine_params if refine_params is not None else {}
        self.phase_params = phase_params if phase_params is not None else {}
        self.intensity_threshold = intensity_threshold
        self.wavelength = wavelength
        self.instrument_profile = instrument_profile
        self.express_mode = express_mode
        self.maximum_grouping_distance = maximum_grouping_distance
        self.max_phases = max_phases
        self.pinned_phases = pinned_phases
        self.record_peak_matcher_scores = record_peak_matcher_scores

        self.all_phases_result = all_phases_result
        self.peak_obs = peak_obs

    def expand_node(self, nid: str) -> list[str]:
        """
        Expand a node in the search tree.

        This method will first do a naive search match method to find the best matched phases. Then it will refine the
        best matched phases and add the results to the search tree.

        Args:
            nid: the node id
        """
        node: Node = self.get_node(nid)
        logger.info(
            f"Expanding node {nid} with current phases {node.data.current_phases}, "
            f"Rwp = {node.data.current_result.lst_data.rwp if node.data.current_result is not None else None}"
        )
        if node is None:
            raise ValueError(f"Node with id {nid} does not exist.")
        if node.data is None or node.data.status != "pending":
            raise ValueError(f"Node with id {nid} is not expandable.")

        node.data.status = "running"
        try:
            # remove phases that are already in the current result
            current_phases_set = set(node.data.current_phases)
            all_phases_result = {
                phase: result
                for phase, result in self.all_phases_result.items()
                if phase not in current_phases_set
            }
            best_phases, scores, threshold = self.score_phases(
                all_phases_result, node.data.current_result
            )

            if self.record_peak_matcher_scores:
                node.data.peak_matcher_scores = scores
                node.data.peak_matcher_threshold = threshold

            new_results = self.refine_phases(
                best_phases, pinned_phases=node.data.current_phases
            )

            # group the results
            grouped_results = group_phases(
                new_results,
                distance_threshold=self.maximum_grouping_distance,
            )

            # if express mode is on, we will put all the phases in its own group
            if self.express_mode:
                for i, phase in enumerate(grouped_results):
                    grouped_results[phase]["group_id"] = i

            for phase, new_result in new_results.items():
                new_phases = [*node.data.current_phases, phase]

                group_id = grouped_results[phase]["group_id"]
                fom = grouped_results[phase]["fom"]

                is_best_result_in_group = phase == max(
                    [
                        phase_
                        for phase_ in grouped_results
                        if grouped_results[phase_]["group_id"] == group_id
                    ],
                    key=lambda x: grouped_results[x]["fom"],
                )

                if new_result is not None:
                    searched_phases = [
                        p for p in new_phases if p not in self.pinned_phases
                    ]
                    sorted_searched_phases = sorted(
                        searched_phases,
                        key=lambda phase: new_result.peak_data[
                            new_result.peak_data["phase"] == phase.path.stem
                        ]["intensity"].sum(),
                        reverse=True,
                    )
                    # make sure the newly added phase has the lowest peak intensity
                    is_low_weight_fraction = (
                        sorted_searched_phases[-1] != searched_phases[-1]
                    )
                else:
                    is_low_weight_fraction = False

                if new_result is not None:
                    peak_matcher = PeakMatcher(
                        new_result.peak_data[["2theta", "intensity"]].values,
                        self.peak_obs,
                    )
                    isolated_missing_peaks = peak_matcher.get_isolated_peaks(
                        peak_type="missing"
                    ).tolist()
                    isolated_extra_peaks = peak_matcher.get_isolated_peaks(
                        peak_type="extra"
                    ).tolist()
                else:
                    isolated_missing_peaks = [[]]
                    isolated_extra_peaks = [[]]

                if new_result is None:
                    status = "error"

                elif (
                    node.data.current_result is not None
                    and (
                        # if the new result is worse than the current result from Rwp perspective
                        node.data.current_result.lst_data.rpb
                        - new_result.lst_data.rpb
                    )
                    < self.rpb_threshold
                ) or (  # or if removing one phase does not improve the result (indication of overfitting)
                    len(
                        remove_unnecessary_phases(
                            new_result,
                            [p.path for p in new_phases],
                            self.rpb_threshold,
                        )
                    )
                    != len(new_phases)
                ):
                    status = "no_improvement"
                elif is_low_weight_fraction:
                    status = "low_weight_fraction"
                elif not is_best_result_in_group:
                    status = "similar_structure"
                elif len(new_phases) >= self.max_phases:
                    status = "max_depth"
                else:
                    status = "pending"

                self.create_node(
                    data=SearchNodeData(
                        current_result=new_result,
                        current_phases=new_phases,
                        status=status,
                        group_id=group_id,
                        isolated_extra_peaks=isolated_extra_peaks,
                        isolated_missing_peaks=isolated_missing_peaks,
                        fom=fom,
                        lattice_strain=grouped_results[phase]["lattice_strain"],
                    ),
                    parent=nid,
                )
        except Exception:
            node.data.status = "error"
            raise

        node.data.status = "expanded"

        return self.get_expandable_children(nid)

    def get_expandable_children(self, nid: str) -> list[str]:
        """
        Get the expandable children of a node.

        The expandable children are the children that have not been expanded yet, which is marked as "pending".

        Args:
            nid: the node id

        Returns
        -------
            a list of node ids that are expandable
        """
        if not self.contains(nid):
            raise ValueError(f"Node with id {nid} does not exist.")

        return [
            child.identifier
            for child in self.children(nid)
            if self.get_node(child.identifier).data.status == "pending"
        ]

    def expand_root(self) -> list[str]:
        """Expand the root node."""
        return self.expand_node(self.root)

    def get_all_possible_nodes_at_same_level(self, node: Node) -> tuple[Node, ...]:
        """
        Get all possible phases that can be added to the current phase combination at this level.

        Args:
            node: the node in the search tree

        Returns
        -------
            a list of selected node
        """
        if node.data.status not in {
            "expanded",
            "max_depth",
            "similar_structure",
        }:
            raise ValueError(f"Node with id {node.identifier} is not expanded.")
        if node.data.group_id == -1:
            raise ValueError("The group id is not available at this node.")

        nodes_at_same_level = self.children(self.ancestor(node.identifier))

        phases_at_same_level = [
            node_at_same_level
            for node_at_same_level in nodes_at_same_level
            if node_at_same_level.data.group_id == node.data.group_id
            and node_at_same_level.data.status
            in {"similar_structure", "expanded", "max_depth"}
        ]

        phases_at_same_level = sorted(
            phases_at_same_level, key=lambda x: x.data.fom, reverse=True
        )

        return tuple(phases_at_same_level)

    def get_phase_combinations(
        self, node: Node
    ) -> tuple[
        list[tuple[RefinementPhase, ...]],
        list[tuple[float, ...]],
        list[tuple[float, ...]],
    ]:
        """
        Get all the phase combinations at this node.

        Args:
            node: the node that will be used to get the phase combinations

        Returns
        -------
            a tuple of the phase combinations
        """
        if node.data.status not in {"expanded", "max_depth"}:
            raise ValueError(f"Node with id {node.identifier} is not expanded.")

        # set up the default value for the current_phases
        parent_node = node

        all_possible_nodes = []

        while self.level(parent_node.identifier) != 0:
            all_possible_nodes.append(
                self.get_all_possible_nodes_at_same_level(parent_node)
            )
            parent_node = self.get_node(self.ancestor(parent_node.identifier))

        all_possible_nodes.append([parent_node])

        all_possible_nodes = all_possible_nodes[::-1]

        foms = [
            tuple([node.data.fom or 0 for node in possible_nodes])
            for possible_nodes in all_possible_nodes
        ]
        phases = [
            (pinned_phase,)
            for pinned_phase in all_possible_nodes[0][
                0
            ].data.current_phases  # root node
        ] + [
            tuple([node.data.current_phases[-1] for node in possible_nodes])
            for possible_nodes in all_possible_nodes[1:]
        ]
        lattice_strains = [
            tuple([node.data.lattice_strain or 0 for node in possible_nodes])
            for possible_nodes in all_possible_nodes
        ]

        return phases, foms, lattice_strains

    def score_phases(
        self,
        all_phases_result: dict[RefinementPhase, RefinementResult],
        current_result: RefinementResult | None = None,
    ) -> tuple[list[RefinementPhase], dict[RefinementPhase, list[float]], float]:
        """
        Get the best matched phases.

        This is a naive search-match method based on the peak matching score. It will return the best matched phases,
        all phases' scores, and the score's threshold.

        The threshold is determined by finding the inflection point of the percentile of the scores.

        Args:
            all_phases_result: the result of all the phases
            current_result: the current result

        Returns
        -------
            a tuple containing the best matched phases, all phases' scores, and the score's threshold
        """
        if current_result is None:
            missing_peaks = self.peak_obs
        else:
            current_peak_calc = current_result.peak_data[["2theta", "intensity"]].values
            missing_peaks = PeakMatcher(current_peak_calc, self.peak_obs).missing

        if len(missing_peaks) == 0:
            return [], {}, 0

        peak_calcs = [
            refinement_result.peak_data[
                refinement_result.peak_data["phase"] == phase.path.stem
            ][["2theta", "intensity"]].values
            for phase, refinement_result in all_phases_result.items()
        ]

        if self.record_peak_matcher_scores:
            peak_matchers = dict(
                zip_longest(
                    all_phases_result.keys(),
                    batch_peak_matching(
                        peak_calcs, missing_peaks, return_type="PeakMatcher"
                    ),
                    fillvalue=None,
                )
            )

            scores = {
                k: v.score() if v is not None else 0 for k, v in peak_matchers.items()
            }

            raw_scores = {}

            for phase, peak_matcher in peak_matchers.items():
                if peak_matcher is not None:
                    raw_scores[phase] = [
                        peak_matcher.score(
                            matched_coeff=1,
                            wrong_intensity_coeff=0,
                            missing_coeff=0,
                            extra_coeff=0,
                        ),
                        peak_matcher.score(
                            matched_coeff=0,
                            wrong_intensity_coeff=1,
                            missing_coeff=0,
                            extra_coeff=0,
                        ),
                        peak_matcher.score(
                            matched_coeff=0,
                            wrong_intensity_coeff=0,
                            missing_coeff=1,
                            extra_coeff=0,
                        ),
                        peak_matcher.score(
                            matched_coeff=0,
                            wrong_intensity_coeff=0,
                            missing_coeff=0,
                            extra_coeff=1,
                        ),
                    ]
        else:
            scores = dict(
                zip_longest(
                    all_phases_result.keys(),
                    batch_peak_matching(peak_calcs, missing_peaks, return_type="score"),
                    fillvalue=0,
                )
            )
            raw_scores = {}

        peak_matcher_score_threshold, _ = find_optimal_score_threshold(
            list(scores.values())
        )
        peak_matcher_score_threshold = max(peak_matcher_score_threshold, 0)

        filtered_scores = {
            phase: score
            for phase, score in scores.items()
            if score >= peak_matcher_score_threshold
        }

        return (
            sorted(filtered_scores, key=lambda x: filtered_scores[x], reverse=True),
            raw_scores,
            peak_matcher_score_threshold,
        )

    def refine_phases(
        self,
        phases: list[RefinementPhase],
        pinned_phases: list[RefinementPhase] | None = None,
    ) -> dict[RefinementPhase, RefinementResult | None]:
        """
        Get the result of all the phases.

        Args:
            phases: the phases
            pinned_phases: the pinned phases thta will be included in all the refinement

        Returns
        -------
            a dictionary containing the phase and its result
        """
        if pinned_phases is None:
            pinned_phases = []

        return dict(
            zip_longest(
                phases,
                self._batch_refine(
                    all_references=[[*pinned_phases, phase] for phase in phases],
                ),
                fillvalue=None,
            )
        )

    def _batch_refine(
        self,
        all_references: list[list[RefinementPhase]],
    ) -> list[RefinementResult]:
        return batch_refinement(
            self.pattern_path,
            all_references,
            wavelength=self.wavelength,
            instrument_profile=self.instrument_profile,
            phase_params=self.phase_params,
            refinement_params=self.refinement_params,
        )

    def _clone(self, identifier=None, with_tree=False, deep=False):
        return self.__class__(
            identifier=identifier,
            tree=self if with_tree else None,
            deep=deep,
            max_phases=self.max_phases,
            pattern_path=self.pattern_path,
            all_phases_result=self.all_phases_result,
            peak_obs=self.peak_obs,
            rpb_threshold=self.rpb_threshold,
            refine_params=self.refinement_params,
            phase_params=self.phase_params,
            intensity_threshold=self.intensity_threshold,
            wavelength=self.wavelength,
            instrument_profile=self.instrument_profile,
            maximum_grouping_distance=self.maximum_grouping_distance,
            pinned_phases=self.pinned_phases,
            express_mode=self.express_mode,
        )

    @classmethod
    def from_search_tree(
        cls, root_nid: str, search_tree: BaseSearchTree
    ) -> BaseSearchTree:
        """
        Create a new search tree from an existing search tree.

        Args:
            root_nid: the node id that will be used as the root node for the new search tree
            search_tree: the search tree that will be used to create the new search tree

        Returns
        -------
            the new search tree
        """
        root_node = search_tree.get_node(root_nid)
        if root_node is None:
            raise ValueError(f"Node with id {root_nid} does not exist.")

        new_search_tree = cls(
            max_phases=search_tree.max_phases,
            pattern_path=search_tree.pattern_path,
            all_phases_result=search_tree.all_phases_result,
            peak_obs=search_tree.peak_obs,
            rpb_threshold=search_tree.rpb_threshold,
            refine_params=search_tree.refinement_params,
            phase_params=search_tree.phase_params,
            intensity_threshold=search_tree.intensity_threshold,
            wavelength=search_tree.wavelength,
            instrument_profile=search_tree.instrument_profile,
            express_mode=search_tree.express_mode,
            maximum_grouping_distance=search_tree.maximum_grouping_distance,
            pinned_phases=search_tree.pinned_phases,
            record_peak_matcher_scores=search_tree.record_peak_matcher_scores,
        )
        new_search_tree.add_node(root_node)

        return new_search_tree

    def add_subtree(self, anchor_nid: str, search_tree: BaseSearchTree):
        """
        Add a subtree to the search tree.

        Args:
            anchor_nid: the node id that the subtree will be added to
            search_tree: the search tree that will be added to the search tree

        Returns
        -------
            the merged search tree
        """
        # update the data from the search tree
        if (
            search_tree.get_node(search_tree.root).data.current_phases
            != self.get_node(anchor_nid).data.current_phases
        ):
            raise ValueError(
                "The root node of the subtree must have the same current_phases as the anchor node."
            )

        self.merge(nid=anchor_nid, new_tree=search_tree, deep=False)
        self.update_node(anchor_nid, data=search_tree.get_node(search_tree.root).data)


class SearchTree(BaseSearchTree):
    """
    A class for the search tree.

    Args:
        pattern_path: the path to the pattern
        cif_paths: the paths to the CIF files
        pinned_phases: the phases that will be included in all the refinement
        refine_params: the refinement parameters, it will be passed to the refinement function.
        phase_params: the phase parameters, it will be passed to the refinement function.
        instrument_profile: the name/path of the instrument file, it will be passed to the refinement function.
        maximum_grouping_distance: the maximum grouping distance, default to 0.1
        max_phases: the maximum number of phases, note that the pinned phases are COUNTED as well
        rpb_threshold: the minimium Rpb improvement for the search tree to continue to expand one node.
    """

    def __init__(
        self,
        pattern_path: Path | str,
        cif_paths: list[RefinementPhase | Path | str],
        pinned_phases: list[RefinementPhase | Path | str] | None = None,
        refine_params: dict[str, ...] | None = None,
        phase_params: dict[str, ...] | None = None,
        wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float = "Cu",
        instrument_profile: str | Path = "Aeris-fds-Pixcel1d-Medipix3",
        express_mode: bool = True,
        maximum_grouping_distance: float = 0.1,
        max_phases: float = 5,
        rpb_threshold: float = 4,
        record_peak_matcher_scores: bool = False,
        *args,
        **kwargs,
    ):
        pattern_path = Path(pattern_path)

        # remove duplicates
        self.cif_paths = list(
            {RefinementPhase.make(cif_path) for cif_path in cif_paths}
        )
        self.pinned_phases = list(
            {RefinementPhase.make(pinned_phase) for pinned_phase in pinned_phases}
            if pinned_phases is not None
            else []
        )

        if len(self.pinned_phases) >= max_phases:
            raise ValueError(
                "The number of pinned phases must be less than the max_phases, "
                "as the pinned phases are counted in the max_phases."
            )

        super().__init__(
            pattern_path,
            None,
            None,
            refine_params,
            phase_params,
            0.0,
            wavelength,
            instrument_profile,
            express_mode,
            maximum_grouping_distance,
            max_phases,
            rpb_threshold,
            self.pinned_phases,
            record_peak_matcher_scores,
            *args,
            **kwargs,
        )

        # side effect: sets self.peak_obs and self.refinement_params["wmax"] in the function
        # also update the initial guess of b1 in self.refinement_params
        self._detect_peak_in_pattern()

        self.intensity_threshold = min(
            find_optimal_intensity_threshold(self.peak_obs[:, 1]),
            0.1 * np.max(self.peak_obs[:, 1]),
        )
        logger.info(
            f"The intensity threshold is automatically set "
            f"to {self.intensity_threshold / self.peak_obs[:, 1].max() * 100:.2f} % of maximum peak intensity."
        )

        root_node = self._create_root_node()
        self.add_node(root_node)

        all_phases_result = self._get_all_cleaned_phases_result()

        if self.express_mode:
            logger.info("Express mode is enabled. Grouping phases before starting.")
            phases_grouped = group_phases(
                all_phases_result,
                distance_threshold=self.maximum_grouping_distance,
            )
            phase_group_mapping = {}

            for phase in phases_grouped:
                group_id = phases_grouped[phase]["group_id"]
                phase_group_mapping.setdefault(group_id, []).append(
                    {
                        "phase": phase,
                        "fom": phases_grouped[phase]["fom"],
                        "lattice_strain": phases_grouped[phase]["lattice_strain"],
                    }
                )

            for group in phase_group_mapping:  # noqa: PLC0206
                phase_group_mapping[group] = sorted(
                    phase_group_mapping[group],
                    key=lambda x: x["fom"],
                    reverse=True,
                )
            logger.info(
                f"Phases are grouped into {len(phase_group_mapping)} groups. In "
                f"express mode, only the best phase in each group will be considered during the search."
            )
            self.phases_grouped = phases_grouped
            self.all_phases_result = {
                phase_group_mapping[group][0]["phase"]: all_phases_result[
                    phase_group_mapping[group][0]["phase"]
                ]
                for group in phase_group_mapping
            }
        else:
            self.phases_grouped = {}
            self.all_phases_result = all_phases_result

    def _detect_peak_in_pattern(self) -> pd.DataFrame:
        logger.info("Detecting peaks in the pattern.")
        if self.refinement_params.get("wmax", None) is not None:
            warnings.warn(
                f"The wmax ({self.refinement_params['wmax']}) in refinement_params "
                f"will be ignored. The wmax will be automatically adjusted."
            )
        peak_list = detect_peaks(
            self.pattern_path,
            wavelength=self.wavelength,
            instrument_profile=self.instrument_profile,
            wmin=self.refinement_params.get("wmin", None),
            wmax=None,
        )
        if len(peak_list) == 0:
            raise ValueError("No peaks are detected in the pattern.")
        optimal_wmax = get_optimal_max_two_theta(peak_list)
        logger.info(f"The wmax is automatically adjusted to {optimal_wmax}.")
        self.refinement_params["wmax"] = optimal_wmax

        peak_list_array = peak_list[["2theta", "intensity"]].values

        self.peak_obs = peak_list_array[
            np.where(peak_list_array[:, 0] < self.refinement_params["wmax"])
        ]

        # estimate the mean b1 value from the pattern
        estimated_b1 = np.mean(peak_list["b1"].dropna().values)
        initial, lower, upper = parse_refinement_param(self.phase_params["b1"])
        if not isinstance(initial, Number) and estimated_b1 is not None:
            if lower is not None and estimated_b1 < lower:
                estimated_b1 = lower + 0.1 * abs(upper - lower)
            if upper is not None and estimated_b1 > upper:
                estimated_b1 = upper - 0.1 * abs(upper - lower)
            self.phase_params["b1"] = (
                f"{estimated_b1:.6f}"
                + (f"_{lower}" if lower is not None else "")
                + (f"^{upper}" if upper is not None else "")
            )
            logger.info(
                f"The initial value of b1 is automatically set to {self.refinement_params['b1']}."
            )

        return peak_list

    def _create_root_node(self) -> Node:
        logger.info("Creating the root node.")
        return Node(
            data=SearchNodeData(
                current_result=(
                    self._batch_refine([self.pinned_phases])[0]
                    if self.pinned_phases
                    else None
                ),
                current_phases=self.pinned_phases,
            ),
        )

    def _get_all_cleaned_phases_result(self) -> dict[RefinementPhase, RefinementResult]:
        logger.info("Refining all the phases in the dataset.")
        pinned_phases_set = set(self.pinned_phases)
        cif_paths = [
            cif_path for cif_path in self.cif_paths if cif_path not in pinned_phases_set
        ]
        all_phases_result = self.refine_phases(
            cif_paths,
            pinned_phases=self.pinned_phases,
        )

        # adjust the initial value of eps1 based on the weighted average of all the phases
        if not isinstance(self.refinement_params.get("eps1", 0), Number):
            weighted_eps1 = 0
            rwp_sum = 0

            for result in all_phases_result.values():
                if result is not None:
                    weighted_eps1 += (
                        1
                        / (result.lst_data.rwp + 1e-1)
                        * get_number(result.lst_data.EPS1)
                    )
                    rwp_sum += result.lst_data.rwp
            weighted_eps1 /= rwp_sum
            _, eps1_lower, eps1_upper = parse_refinement_param(
                self.refinement_params["eps1"]
            )
            self.refinement_params["eps1"] = (
                f"{weighted_eps1:.6f}"
                + (f"_{eps1_lower}" if eps1_lower is not None else "")
                + (f"^{eps1_upper}" if eps1_upper is not None else "")
            )
            logger.info(
                f"The initial value of eps1 is automatically set to {self.refinement_params['eps1']}."
            )

        # adjust the initial value of eps2 based on the weighted average of all the phases
        if not isinstance(self.refinement_params.get("eps2", 0), Number):
            weighted_eps2 = 0
            rwp_sum = 0

            for result in all_phases_result.values():
                if result is not None:
                    weighted_eps2 += (
                        1
                        / (result.lst_data.rwp + 1e-1)
                        * get_number(result.lst_data.EPS2)
                    )
                    rwp_sum += result.lst_data.rwp
            weighted_eps2 /= rwp_sum
            _, eps2_lower, eps2_upper = parse_refinement_param(
                self.refinement_params["eps2"]
            )
            self.refinement_params["eps2"] = (
                f"{weighted_eps2:.6f}"
                + (f"_{eps2_lower}" if eps2_lower is not None else "")
                + (f"^{eps2_upper}" if eps2_upper is not None else "")
            )
            logger.info(
                f"The initial value of eps2 is automatically set to {self.refinement_params['eps2']}."
            )

        # adjust the initial value of k1 and b1 for each phase based on the refinement result
        all_phases_result_updated = {}
        for phase, result in all_phases_result.items():
            if result is not None:
                k1 = get_number(result.lst_data.phases_results[phase.path.stem].k1)
                b1 = get_number(result.lst_data.phases_results[phase.path.stem].B1)

                k1_initial, k1_lower, k1_upper = parse_refinement_param(
                    phase.params.get("k1", self.phase_params["k1"])
                )
                k1 = k1 or k1_initial
                phase.params["k1"] = (
                    f"{k1:.6f}"
                    + (f"_{k1_lower}" if k1_lower is not None else "")
                    + (f"^{k1_upper}" if k1_upper is not None else "")
                )

                b1_initial, b1_lower, b1_upper = parse_refinement_param(
                    phase.params.get("b1", self.phase_params["b1"])
                )
                b1 = b1 or b1_initial
                phase.params["b1"] = (
                    f"{b1:.6f}"
                    + (f"_{b1_lower}" if b1_lower is not None else "")
                    + (f"^{b1_upper}" if b1_upper is not None else "")
                )

            all_phases_result_updated[phase] = result

        # clean up cif paths (if no result, remove from list)
        all_phases_result = {
            phase: result
            for phase, result in all_phases_result.items()
            if result is not None
        }

        logger.info(
            f"Finished refining {len(cif_paths)} phases, "
            f"with {len(cif_paths) - len(all_phases_result)} phases removed."
        )

        return all_phases_result

    def get_search_results(self) -> list[SearchResult]:
        """
        Get the search results.

        The search results are the results of the nodes that have been expanded and have no expandable children.

        Returns
        -------
            a dictionary containing the phase combinations and their results
        """
        results = []

        for node in self.nodes.values():
            if node.data.current_result is None:
                continue
            if node.data.status in {"expanded", "max_depth"} and all(
                child.data.status not in {"expanded", "max_depth"}
                for child in self.children(node.identifier)
            ):
                phases, foms, lattice_strains = self.get_phase_combinations(node)

                # if express mode is on, we will expand the phases based on the grouping result
                # to include all the similar phases
                if self.express_mode:
                    for i, phases_ in enumerate(phases):
                        new_phases_ = []
                        new_foms_ = []
                        new_lattice_strains_ = []
                        for j in range(len(phases_)):
                            phase = phases_[j]
                            group_id = self.phases_grouped[phase]["group_id"]
                            all_phases_in_group = [
                                p
                                for p in self.phases_grouped
                                if self.phases_grouped[p]["group_id"] == group_id
                            ]
                            new_phases_.extend(all_phases_in_group)
                            new_foms_.extend(
                                [
                                    self.phases_grouped[p]["fom"]
                                    for p in all_phases_in_group
                                ]
                            )
                            new_lattice_strains_.extend(
                                [
                                    self.phases_grouped[p]["lattice_strain"]
                                    for p in all_phases_in_group
                                ]
                            )
                        phases[i] = tuple(new_phases_)
                        foms[i] = tuple(new_foms_)
                        lattice_strains[i] = tuple(new_lattice_strains_)

                results.append(
                    SearchResult(
                        refinement_result=node.data.current_result,
                        phases=tuple(phases),
                        foms=tuple(foms),
                        lattice_strains=tuple(lattice_strains),
                        missing_peaks=node.data.isolated_missing_peaks,
                        extra_peaks=node.data.isolated_extra_peaks,
                    )
                )
        return get_natural_break_results(results)

    def show(
        self,
        nid=None,
        level=Tree.ROOT,
        idhidden=False,
        filter=None,
        key=None,
        reverse=False,
        line_type="ascii-ex",
        data_property="pretty_output",
        stdout=False,
        sorting=True,
    ):
        """
        Show the search tree.

        Args:
            nid: the node id
            level: the level of the tree
            idhidden: whether the node id is hidden
            filter: the filter function
            key: the sorting key
            reverse: whether to reverse the sorting
            line_type: the line type
            data_property: the data property
            stdout: whether to print the result
            sorting: whether to sort the result

        Returns
        -------
            the string representation of the search tree
        """
        return super().show(
            nid=nid,
            level=level,
            idhidden=idhidden,
            filter=filter,
            key=key,
            reverse=reverse,
            line_type=line_type,
            data_property=data_property,
            stdout=stdout,
            sorting=sorting,
        )

    def _clone(self, identifier=None, with_tree=False, deep=False):
        raise NotImplementedError(f"{self.__class__.__name__} cannot be cloned.")
