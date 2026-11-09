from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist

from ._kernel_cme import as_2d_float, as_covariates, rbf_kernel


def _all_pairs_median_bandwidth(values: NDArray) -> float:
    """Gaussian bandwidth from all training-response pairs, as in the paper."""

    array = as_2d_float(values)
    if array.shape[0] < 2:
        return 1.0
    distances = pdist(array, metric="euclidean")
    positive = distances[np.isfinite(distances) & (distances > 0.0)]
    if positive.size == 0:
        return 1.0
    return float(
        max(
            np.median(positive),
            np.finfo(np.float64).eps ** 0.25,
        )
    )


def _strict_binary_treatment(
    values: NDArray | int,
    *,
    n_rows: int,
    allow_scalar: bool,
) -> NDArray[np.int64]:
    """Validate binary labels before casting, avoiding fractional truncation."""

    raw = np.asarray(values)
    if raw.ndim > 2 or (raw.ndim == 2 and raw.shape[1] != 1):
        raise ValueError("treatment must be scalar or one-dimensional")
    raw = raw.reshape(-1)
    if allow_scalar and raw.size == 1 and n_rows != 1:
        raw = np.full(n_rows, raw.item())
    if raw.size != n_rows:
        message = "treatment must be scalar 0/1 or align with x" if allow_scalar else (
            "x, treatment, and y must have the same number of rows"
        )
        raise ValueError(message)
    if not np.all(np.isin(raw, [0, 1])):
        raise ValueError("Causal-DRF requires treatment values 0 and 1")
    return raw.astype(np.int64)


@dataclass(slots=True)
class _TreeNode:
    feature: int = -1
    threshold: float = 0.0
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None
    populate_indices: Optional[NDArray[np.int64]] = None

    @property
    def is_leaf(self) -> bool:
        return self.populate_indices is not None


@dataclass(slots=True)
class CausalDRFConfig:
    x_dim: int
    num_treatments: int = 2
    num_trees: int = 200
    num_groups: int = 20
    tree_subsample_fraction: float = 0.8
    honesty_fraction: float = 0.5
    min_leaf_per_arm: int = 4
    max_depth: int = 12
    max_features: Optional[int] = None
    # Retained as an explicit compatibility guard.  The paper's algorithm
    # evaluates every admissible split level, so threshold subsampling is not
    # supported by this implementation.
    num_split_candidates: Optional[int] = None
    alpha_regular: float = 0.2
    num_rff: int = 32
    outcome_bandwidth: Optional[float] = None
    seed: int = 227


@dataclass(slots=True)
class CausalDRFDiagnostics:
    outcome_bandwidth: float = 0.0
    num_trees: int = 0
    num_groups: int = 0
    mean_num_leaves: float = 0.0
    max_observed_depth: int = 0
    leaf_upper_bound_violation_rate: float = 0.0
    max_leaf_size_per_arm: tuple[int, int] = (0, 0)
    f4_upper_bound_note: str = (
        "The paper states the per-arm kappa-to-(2*kappa-1) range as a forest "
        "assumption but leaves the finite-sample stopping/adjustment algorithm "
        "unspecified; violations are reported rather than repaired by changing "
        "the treatment sampling law or CKTE split criterion."
    )
    tree_sampling: str = (
        "iid Bernoulli(1/2) subforest half-samples; uniform without-replacement "
        "tree subsamples; random honest half-splits"
    )
    sampling_adapter: str = (
        "Evaluation-only empirical-law extension using the positive arm-specific "
        "honest-leaf weights induced by the shared CKTE-targeted forest"
    )


class CausalDRF:
    """Causal distributional random forest of Näf et al. (2026).

    Trees share an honest partition and use the paper's CKTE split criterion:
    a balanced squared distance between the two children's treated-minus-control
    outcome embeddings.  Gaussian-outcome kernel distances are approximated with
    random Fourier features, as in the paper's scalable implementation.

    The signed treated-minus-control weights represent the CKTE.  Distributional
    benchmarking additionally needs draws, so ``sample_potential`` uses the
    corresponding *arm-specific positive* honest-leaf forest weights: choose a
    tree uniformly, then an observed outcome uniformly from that query leaf and
    treatment arm.  This is the empirical-distribution adapter naturally induced
    by the forest leaves; it is distinct from the signed CKTE itself.

    The fitted object also retains the paper's B half-sample subforests, so
    group-specific CKTE weights can be used for the proposed test and witness
    confidence band.  Tree subsampling is uniform over observations and honesty
    is a random half split.  Degenerate finite-sample draws are retried solely
    to ensure that both honest halves contain both treatment arms.
    """

    def __init__(self, config: CausalDRFConfig) -> None:
        integer_fields = {
            "x_dim": config.x_dim,
            "num_treatments": config.num_treatments,
            "num_trees": config.num_trees,
            "num_groups": config.num_groups,
            "min_leaf_per_arm": config.min_leaf_per_arm,
            "max_depth": config.max_depth,
            "num_rff": config.num_rff,
            "seed": config.seed,
        }
        for name, value in integer_fields.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
        real_fields = {
            "tree_subsample_fraction": config.tree_subsample_fraction,
            "honesty_fraction": config.honesty_fraction,
            "alpha_regular": config.alpha_regular,
        }
        if config.outcome_bandwidth is not None:
            real_fields["outcome_bandwidth"] = config.outcome_bandwidth
        for name, real_value in real_fields.items():
            if isinstance(real_value, (bool, np.bool_)) or not isinstance(
                real_value,
                (int, float, np.integer, np.floating),
            ):
                raise TypeError(f"{name} must be a real number")
            if not np.isfinite(real_value):
                raise ValueError(f"{name} must be finite")
        if config.x_dim < 1:
            raise ValueError("x_dim must be positive")
        if config.num_treatments != 2:
            raise ValueError("Näf et al. (2026) formulate Causal-DRF for binary treatment")
        if config.num_trees < 1 or config.num_groups < 1:
            raise ValueError("num_trees and num_groups must be positive")
        if config.num_groups > config.num_trees:
            raise ValueError("num_groups cannot exceed num_trees")
        if config.num_trees % config.num_groups:
            raise ValueError("num_trees must be divisible by num_groups so N = B * L")
        if not 0.0 < config.tree_subsample_fraction <= 1.0:
            raise ValueError("tree_subsample_fraction must lie in (0, 1]")
        if not np.isclose(
            config.honesty_fraction,
            0.5,
        ):
            raise ValueError("honesty_fraction must equal 0.5 for the paper's half split")
        if config.min_leaf_per_arm < 1:
            raise ValueError("min_leaf_per_arm must be positive")
        if config.max_depth < 1:
            raise ValueError("max_depth must be positive")
        if config.max_features is not None:
            if isinstance(config.max_features, (bool, np.bool_)) or not isinstance(
                config.max_features, (int, np.integer)
            ):
                raise TypeError("max_features must be an integer or None")
            if not 1 <= config.max_features <= config.x_dim:
                raise ValueError("max_features must lie in [1, x_dim]")
        if config.num_split_candidates is not None:
            raise ValueError(
                "num_split_candidates must be None; Causal-DRF evaluates every valid threshold"
            )
        if not 0.0 <= config.alpha_regular <= 0.2:
            raise ValueError("alpha_regular must lie in [0, 0.2]")
        if config.num_rff < 1:
            raise ValueError("num_rff must be positive")
        if config.seed < 0:
            raise ValueError("seed must be nonnegative")
        if config.outcome_bandwidth is not None and config.outcome_bandwidth <= 0.0:
            raise ValueError("outcome_bandwidth must be positive and finite")
        self.config = config
        self.trees: list[_TreeNode] = []
        self.tree_group_ids = np.empty(0, dtype=np.int64)
        self.half_sample_indices: list[NDArray[np.int64]] = []
        self.x_train = np.empty((0, config.x_dim), dtype=np.float64)
        self.treatment_train = np.empty(0, dtype=np.int64)
        self.y_train = np.empty((0, 1), dtype=np.float64)
        self.outcome_features = np.empty((0, 2 * config.num_rff), dtype=np.float64)
        self.outcome_bandwidth = 1.0
        self._global_arm_indices: dict[int, NDArray[np.int64]] = {}
        self._outcome_gram: Optional[NDArray[np.float64]] = None
        self.diagnostics = CausalDRFDiagnostics()

    def _make_outcome_features(self, rng: np.random.Generator) -> NDArray[np.float64]:
        omega = rng.normal(
            loc=0.0,
            scale=1.0 / self.outcome_bandwidth,
            size=(self.config.num_rff, self.y_train.shape[1]),
        )
        angles = self.y_train @ omega.T
        return np.concatenate([np.cos(angles), np.sin(angles)], axis=1) / np.sqrt(self.config.num_rff)

    def _arm_counts_ok(self, indices: NDArray[np.int64], minimum: int) -> bool:
        if indices.size < 2 * minimum:
            return False
        treatments = self.treatment_train[indices]
        return bool(np.sum(treatments == 0) >= minimum and np.sum(treatments == 1) >= minimum)

    def _best_split_for_feature(
        self,
        build_indices: NDArray[np.int64],
        populate_indices: NDArray[np.int64],
        *,
        feature: int,
        minimum: int,
        rng: np.random.Generator,
    ) -> Optional[tuple[float, float]]:
        """Return the best score/threshold while evaluating every split level.

        Sorting once and using cumulative arm counts and random-feature sums
        avoids the quadratic work that a literal evaluation of every threshold
        would otherwise require.
        """

        build_order = np.argsort(self.x_train[build_indices, feature], kind="mergesort")
        ordered_build = build_indices[build_order]
        build_values = self.x_train[ordered_build, feature]
        split_positions = np.flatnonzero(build_values[:-1] < build_values[1:])
        if split_positions.size == 0:
            return None

        left_build_size = split_positions + 1
        right_build_size = build_indices.size - left_build_size
        build_treated = (self.treatment_train[ordered_build] == 1).astype(np.int64)
        build_treated_prefix = np.cumsum(build_treated)
        left_build_treated = build_treated_prefix[split_positions]
        left_build_control = left_build_size - left_build_treated
        total_build_treated = int(build_treated_prefix[-1])
        total_build_control = int(build_indices.size - total_build_treated)
        right_build_treated = total_build_treated - left_build_treated
        right_build_control = total_build_control - left_build_control

        # Algorithm 2 iterates over every observed feature level ``l`` and
        # sends observations with X_j <= l to the left child.
        thresholds = build_values[split_positions]
        # The CKTE contrast only requires each build child to contain both
        # arms.  The kappa leaf constraint belongs to the honest population
        # sample (F4), not to the split-selection sample.
        valid = (
            (left_build_treated >= 1)
            & (left_build_control >= 1)
            & (right_build_treated >= 1)
            & (right_build_control >= 1)
        )
        if not np.any(valid):
            return None

        populate_order = np.argsort(
            self.x_train[populate_indices, feature],
            kind="mergesort",
        )
        ordered_populate = populate_indices[populate_order]
        populate_values = self.x_train[ordered_populate, feature]
        left_populate_size = np.searchsorted(populate_values, thresholds, side="right")
        right_populate_size = populate_indices.size - left_populate_size
        populate_treated = (self.treatment_train[ordered_populate] == 1).astype(np.int64)
        populate_treated_prefix = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(populate_treated)]
        )
        left_populate_treated = populate_treated_prefix[left_populate_size]
        left_populate_control = left_populate_size - left_populate_treated
        total_populate_treated = int(populate_treated_prefix[-1])
        total_populate_control = int(populate_indices.size - total_populate_treated)
        right_populate_treated = total_populate_treated - left_populate_treated
        right_populate_control = total_populate_control - left_populate_control
        valid &= (
            (left_populate_treated >= minimum)
            & (left_populate_control >= minimum)
            & (right_populate_treated >= minimum)
            & (right_populate_control >= minimum)
        )
        if self.config.alpha_regular > 0.0:
            valid &= (
                np.minimum(left_populate_size, right_populate_size)
                >= self.config.alpha_regular * populate_indices.size
            )
        valid_positions = np.flatnonzero(valid)
        if valid_positions.size == 0:
            return None

        features = self.outcome_features[ordered_build]
        treated_mask = build_treated[:, None]
        control_mask = 1 - treated_mask
        treated_prefix = np.cumsum(features * treated_mask, axis=0)
        control_prefix = np.cumsum(features * control_mask, axis=0)
        candidate_splits = split_positions[valid_positions]
        left_treated_sum = treated_prefix[candidate_splits]
        left_control_sum = control_prefix[candidate_splits]
        right_treated_sum = treated_prefix[-1] - left_treated_sum
        right_control_sum = control_prefix[-1] - left_control_sum

        left_contrast = (
            left_treated_sum / left_build_treated[valid_positions, None]
            - left_control_sum / left_build_control[valid_positions, None]
        )
        right_contrast = (
            right_treated_sum / right_build_treated[valid_positions, None]
            - right_control_sum / right_build_control[valid_positions, None]
        )
        contrast_difference = left_contrast - right_contrast
        balance = (
            left_build_size[valid_positions]
            * right_build_size[valid_positions]
            / float(build_indices.size * build_indices.size)
        )
        scores = balance * np.einsum(
            "ij,ij->i",
            contrast_difference,
            contrast_difference,
            optimize=True,
        )
        best_score = float(np.max(scores))
        tied = np.flatnonzero(np.isclose(scores, best_score, rtol=1e-12, atol=1e-15))
        chosen = int(tied[0] if tied.size == 1 else rng.choice(tied))
        threshold = float(thresholds[valid_positions[chosen]])
        return best_score, threshold

    def _build_node(
        self,
        build_indices: NDArray[np.int64],
        populate_indices: NDArray[np.int64],
        *,
        depth: int,
        rng: np.random.Generator,
    ) -> _TreeNode:
        minimum = self.config.min_leaf_per_arm
        upper = 2 * minimum - 1
        populate_treatment = self.treatment_train[populate_indices]
        populate_arm_sizes = np.asarray(
            [
                np.sum(populate_treatment == 0),
                np.sum(populate_treatment == 1),
            ],
            dtype=np.int64,
        )
        if np.all(populate_arm_sizes <= upper):
            return _TreeNode(populate_indices=populate_indices)
        if depth >= self.config.max_depth:
            return _TreeNode(populate_indices=populate_indices)
        if not self._arm_counts_ok(populate_indices, 2 * minimum):
            return _TreeNode(populate_indices=populate_indices)

        feature_count = self.config.max_features
        if feature_count is None:
            feature_count = max(1, int(np.sqrt(self.config.x_dim)))
        features = rng.choice(self.config.x_dim, size=feature_count, replace=False)
        best_score = -np.inf
        best_feature = -1
        best_threshold = 0.0

        def consider(candidate_features: NDArray[np.int64]) -> None:
            nonlocal best_score, best_feature, best_threshold
            for feature_value in candidate_features:
                candidate = self._best_split_for_feature(
                    build_indices,
                    populate_indices,
                    feature=int(feature_value),
                    minimum=minimum,
                    rng=rng,
                )
                if candidate is None:
                    continue
                score, threshold = candidate
                if score > best_score:
                    best_score = score
                    best_feature = int(feature_value)
                    best_threshold = threshold

        if self._arm_counts_ok(build_indices, 2):
            consider(features)

        if best_feature < 0:
            return _TreeNode(populate_indices=populate_indices)
        build_left_mask = self.x_train[build_indices, best_feature] <= best_threshold
        populate_left_mask = self.x_train[populate_indices, best_feature] <= best_threshold
        build_left = build_indices[build_left_mask]
        build_right = build_indices[~build_left_mask]
        populate_left = populate_indices[populate_left_mask]
        populate_right = populate_indices[~populate_left_mask]
        return _TreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=self._build_node(build_left, populate_left, depth=depth + 1, rng=rng),
            right=self._build_node(build_right, populate_right, depth=depth + 1, rng=rng),
        )

    def _bernoulli_subset(
        self,
        source_indices: NDArray[np.int64],
        *,
        probability: float,
        minimum_per_arm: int,
        rng: np.random.Generator,
    ) -> NDArray[np.int64]:
        """Draw an iid Bernoulli subset, retrying only degenerate finite samples."""

        if not self._arm_counts_ok(source_indices, minimum_per_arm):
            raise ValueError("source sample does not contain the required observations per arm")
        for _ in range(1_024):
            selected = source_indices[rng.random(source_indices.size) < probability]
            if self._arm_counts_ok(selected, minimum_per_arm):
                return selected[rng.permutation(selected.size)]
        raise RuntimeError("could not draw a viable Bernoulli half-sample after 1024 attempts")

    def _uniform_subsample(
        self,
        source_indices: NDArray[np.int64],
        *,
        fraction: float,
        minimum_per_arm: int,
        rng: np.random.Generator,
    ) -> NDArray[np.int64]:
        """Draw the paper's fixed-size simple-random tree subsample.

        Draws that cannot produce two honest halves containing both arms are
        retried.  Apart from this finite-sample viability conditioning, every
        subset of the requested size is sampled uniformly without replacement.
        """

        if not self._arm_counts_ok(source_indices, minimum_per_arm):
            raise ValueError("half-sample does not contain the required observations per arm")
        requested_size = int(round(fraction * source_indices.size))
        if requested_size < 2 * minimum_per_arm:
            raise ValueError(
                "tree_subsample_fraction gives too few observations for honest leaves"
            )
        requested_size = min(requested_size, source_indices.size)
        for _ in range(1_024):
            selected = rng.choice(
                source_indices,
                size=requested_size,
                replace=False,
            )
            if self._arm_counts_ok(selected, minimum_per_arm):
                return selected
        raise RuntimeError(
            "could not draw a viable uniform tree subsample after 1024 attempts"
        )

    def _honest_split(
        self,
        indices: NDArray[np.int64],
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        minimum = self.config.min_leaf_per_arm
        if not self._arm_counts_ok(indices, 2 * minimum):
            raise ValueError("tree subsample needs at least 2 * min_leaf_per_arm in each arm")
        midpoint = indices.size // 2
        for _ in range(1_024):
            shuffled = indices[rng.permutation(indices.size)]
            build = shuffled[:midpoint]
            populate = shuffled[midpoint:]
            if self._arm_counts_ok(build, minimum) and self._arm_counts_ok(
                populate,
                minimum,
            ):
                return build, populate
        raise RuntimeError("could not draw viable random honest halves after 1024 attempts")

    def fit(self, x: NDArray, treatment: NDArray, y: NDArray) -> "CausalDRF":
        x_array = as_covariates(x, x_dim=self.config.x_dim)
        y_array = as_2d_float(y)
        treatment_array = _strict_binary_treatment(
            treatment,
            n_rows=x_array.shape[0],
            allow_scalar=False,
        )
        if (
            x_array.shape[0] != treatment_array.shape[0]
            or x_array.shape[0] != y_array.shape[0]
        ):
            raise ValueError("x, treatment, and y must have the same number of rows")
        if np.unique(treatment_array).size != 2:
            raise ValueError("Causal-DRF requires both binary treatment arms")
        self.x_train = x_array
        self.treatment_train = treatment_array
        self.y_train = y_array
        required = 2 * self.config.min_leaf_per_arm
        for arm in (0, 1):
            count = int(np.sum(self.treatment_train == arm))
            if count < required:
                raise ValueError(
                    f"treatment arm {arm} has {count} observations; at least {required} are required for honesty"
                )
        self._global_arm_indices = {
            arm: np.flatnonzero(self.treatment_train == arm).astype(np.int64) for arm in (0, 1)
        }
        if self.config.outcome_bandwidth is None:
            self.outcome_bandwidth = _all_pairs_median_bandwidth(self.y_train)
        else:
            self.outcome_bandwidth = float(self.config.outcome_bandwidth)
        rng = np.random.default_rng(self.config.seed)
        self.outcome_features = self._make_outcome_features(rng)
        self._outcome_gram = None

        groups: list[NDArray[np.int64]] = []
        all_indices = np.arange(self.x_train.shape[0], dtype=np.int64)
        for _ in range(self.config.num_groups):
            groups.append(
                self._bernoulli_subset(
                    all_indices,
                    probability=0.5,
                    minimum_per_arm=required,
                    rng=rng,
                )
            )
        self.half_sample_indices = [group.copy() for group in groups]

        self.trees = []
        tree_group_ids: list[int] = []
        leaf_counts: list[int] = []
        depths: list[int] = []
        leaf_arm_sizes: list[tuple[int, int]] = []
        upper = 2 * self.config.min_leaf_per_arm - 1
        for tree_index in range(self.config.num_trees):
            group_id = tree_index % len(groups)
            group = groups[group_id]
            tree_indices = self._uniform_subsample(
                group,
                fraction=self.config.tree_subsample_fraction,
                minimum_per_arm=required,
                rng=rng,
            )
            build_indices, populate_indices = self._honest_split(tree_indices, rng)
            tree = self._build_node(build_indices, populate_indices, depth=0, rng=rng)
            candidate_leaf_sizes = self._leaf_arm_sizes(tree)
            self.trees.append(tree)
            tree_group_ids.append(group_id)
            num_leaves, maximum_depth = self._tree_shape(tree)
            leaf_counts.append(num_leaves)
            depths.append(maximum_depth)
            leaf_arm_sizes.extend(candidate_leaf_sizes)
        self.tree_group_ids = np.asarray(tree_group_ids, dtype=np.int64)

        leaf_sizes = np.asarray(leaf_arm_sizes, dtype=np.int64)
        upper_violations = np.any(leaf_sizes > upper, axis=1)
        self.diagnostics = CausalDRFDiagnostics(
            outcome_bandwidth=self.outcome_bandwidth,
            num_trees=len(self.trees),
            num_groups=len(groups),
            mean_num_leaves=float(np.mean(leaf_counts)),
            max_observed_depth=max(depths, default=0),
            leaf_upper_bound_violation_rate=float(np.mean(upper_violations)),
            max_leaf_size_per_arm=(
                int(np.max(leaf_sizes[:, 0])),
                int(np.max(leaf_sizes[:, 1])),
            ),
        )
        return self

    def _leaf_arm_sizes(self, node: _TreeNode) -> list[tuple[int, int]]:
        if node.is_leaf:
            if node.populate_indices is None:
                raise RuntimeError("malformed Causal-DRF leaf")
            treatment = self.treatment_train[node.populate_indices]
            return [
                (
                    int(np.sum(treatment == 0)),
                    int(np.sum(treatment == 1)),
                )
            ]
        if node.left is None or node.right is None:
            raise RuntimeError("malformed Causal-DRF tree")
        return self._leaf_arm_sizes(node.left) + self._leaf_arm_sizes(node.right)

    def _tree_shape(self, node: _TreeNode, depth: int = 0) -> tuple[int, int]:
        if node.is_leaf:
            return 1, depth
        if node.left is None or node.right is None:
            raise RuntimeError("malformed Causal-DRF tree")
        left_leaves, left_depth = self._tree_shape(node.left, depth + 1)
        right_leaves, right_depth = self._tree_shape(node.right, depth + 1)
        return left_leaves + right_leaves, max(left_depth, right_depth)

    def _check_fitted(self) -> None:
        if not self.trees:
            raise RuntimeError("CausalDRF must be fit before prediction")

    def _leaf(self, row: NDArray[np.float64], node: _TreeNode) -> NDArray[np.int64]:
        current = node
        while not current.is_leaf:
            if current.left is None or current.right is None:
                raise RuntimeError("malformed Causal-DRF tree")
            current = current.left if row[current.feature] <= current.threshold else current.right
        if current.populate_indices is None:
            raise RuntimeError("malformed Causal-DRF leaf")
        return current.populate_indices

    def _arm_leaf(self, row: NDArray[np.float64], tree: _TreeNode, arm: int) -> NDArray[np.int64]:
        leaf = self._leaf(row, tree)
        arm_indices = leaf[self.treatment_train[leaf] == arm]
        if arm_indices.size == 0:
            # The construction enforces this condition; this fallback only
            # protects prediction if a manually modified tree is supplied.
            return self._global_arm_indices[arm]
        return arm_indices

    def group_arm_weights(self, x: NDArray) -> NDArray[np.float64]:
        """Paper half-sample weights, shape ``(n_query, B, 2, n_train)``."""

        self._check_fitted()
        query = as_covariates(x, x_dim=self.config.x_dim)
        group_ids = self.tree_group_ids
        if group_ids.shape != (len(self.trees),):
            if self.config.num_groups == 1:
                group_ids = np.zeros(len(self.trees), dtype=np.int64)
            else:
                raise RuntimeError("fitted forest is missing its subforest group assignments")
        trees_per_group = np.bincount(
            group_ids,
            minlength=self.config.num_groups,
        ).astype(np.int64)
        if np.any(trees_per_group == 0):
            raise RuntimeError("each Causal-DRF half-sample group must contain a tree")
        weights = np.zeros(
            (
                query.shape[0],
                self.config.num_groups,
                2,
                self.x_train.shape[0],
            ),
            dtype=np.float64,
        )
        for row_index, row in enumerate(query):
            for tree_index, tree in enumerate(self.trees):
                group_id = int(group_ids[tree_index])
                tree_scale = 1.0 / int(trees_per_group[group_id])
                for arm in (0, 1):
                    indices = self._arm_leaf(row, tree, arm)
                    weights[row_index, group_id, arm, indices] += (
                        tree_scale / indices.size
                    )
        return weights

    def arm_weights(self, x: NDArray) -> NDArray[np.float64]:
        """Positive arm weights averaged over the B paper subforests."""

        self._check_fitted()
        query = as_covariates(x, x_dim=self.config.x_dim)
        weights = np.zeros(
            (query.shape[0], 2, self.x_train.shape[0]),
            dtype=np.float64,
        )
        tree_scale = 1.0 / len(self.trees)
        for row_index, row in enumerate(query):
            for tree in self.trees:
                for arm in (0, 1):
                    indices = self._arm_leaf(row, tree, arm)
                    weights[row_index, arm, indices] += tree_scale / indices.size
        return weights

    def group_signed_weights(self, x: NDArray) -> NDArray[np.float64]:
        """Treated-minus-control CKTE weights for every half-sample group."""

        weights = self.group_arm_weights(x)
        return weights[:, :, 1] - weights[:, :, 0]

    def signed_weights(self, x: NDArray) -> NDArray[np.float64]:
        """Point-estimate CKTE weights from Eq. (3) of the supplied manuscript."""

        weights = self.arm_weights(x)
        return weights[:, 1] - weights[:, 0]

    def predict_potential_outcomes(self, x: NDArray) -> NDArray[np.float32]:
        weights = self.arm_weights(x)
        means = np.einsum("qtn,nd->qtd", weights, self.y_train, optimize=True)
        if means.shape[2] != 1:
            raise ValueError("predict_potential_outcomes currently requires scalar outcomes")
        return means[:, :, 0].astype(np.float32)

    def sample_potential(
        self,
        x: NDArray,
        treatment: NDArray | int,
        n_per_x: int = 1,
        seed: Optional[int] = None,
    ) -> NDArray[np.float32]:
        """Draw from an evaluation-only empirical conditional-law extension.

        The paper proposes a CKTE embedding, witness, test, and confidence band;
        it does not propose potential-outcome CDF or quantile estimation.  This
        adapter samples from the positive arm-specific forest law induced by
        the same honest leaves and is used only to place Causal-DRF in the
        repository's common distributional metrics.
        """

        self._check_fitted()
        if isinstance(n_per_x, (bool, np.bool_)) or not isinstance(n_per_x, (int, np.integer)):
            raise TypeError("n_per_x must be an integer")
        if n_per_x < 1:
            raise ValueError("n_per_x must be positive")
        query = as_covariates(x, x_dim=self.config.x_dim)
        treatment_array = _strict_binary_treatment(
            treatment,
            n_rows=query.shape[0],
            allow_scalar=True,
        )
        if seed is not None:
            if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
                raise TypeError("seed must be an integer or None")
            if seed < 0:
                raise ValueError("seed must be nonnegative")
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        samples = np.empty((query.shape[0], n_per_x, self.y_train.shape[1]), dtype=np.float64)
        for row_index, row in enumerate(query):
            leaves = [self._arm_leaf(row, tree, int(treatment_array[row_index])) for tree in self.trees]
            selected_trees = rng.integers(0, len(self.trees), size=n_per_x)
            for sample_index, tree_index in enumerate(selected_trees):
                leaf_indices = leaves[int(tree_index)]
                chosen = int(rng.choice(leaf_indices))
                samples[row_index, sample_index] = self.y_train[chosen]
        return samples.astype(np.float32)

    def predict_ckte_mmd2(self, x: NDArray) -> NDArray[np.float32]:
        """Squared RKHS norm of the forest's signed CKTE embedding."""

        signed = self.signed_weights(x)
        if self._outcome_gram is None:
            self._outcome_gram = rbf_kernel(self.y_train, self.y_train, self.outcome_bandwidth)
        values = np.einsum("qi,ij,qj->q", signed, self._outcome_gram, signed, optimize=True)
        return np.maximum(values, 0.0).astype(np.float32)

    def witness(
        self,
        x: NDArray,
        outcome: NDArray,
        *,
        paired: bool = False,
    ) -> NDArray[np.float32]:
        """Evaluate the treated-minus-control witness on an outcome grid.

        By default the result has shape ``(n_query, n_outcome)``. Set
        ``paired=True`` to evaluate row-matched ``(x_i, y_i)`` pairs.
        """

        signed = self.signed_weights(x)
        outcome_array = as_2d_float(outcome)
        values = signed @ rbf_kernel(self.y_train, outcome_array, self.outcome_bandwidth)
        if paired:
            if outcome_array.shape[0] != signed.shape[0]:
                raise ValueError("paired witness evaluation requires equal row counts")
            return np.diag(values).astype(np.float32)
        return values.astype(np.float32)

    def resampling_radius2(
        self,
        x: NDArray,
        *,
        alpha: float = 0.05,
    ) -> NDArray[np.float32]:
        """Return the paper's q_(n,alpha) from B half-sample CKTE draws."""

        if isinstance(alpha, (bool, np.bool_)) or not isinstance(
            alpha,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("alpha must be a real number")
        if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.config.num_groups < 2:
            raise ValueError("at least two half-sample groups are required for uncertainty")
        group_signed = self.group_signed_weights(x)
        point_signed = group_signed.mean(axis=1)
        deviations = group_signed - point_signed[:, None, :]
        if self._outcome_gram is None:
            self._outcome_gram = rbf_kernel(
                self.y_train,
                self.y_train,
                self.outcome_bandwidth,
            )
        squared_norms = np.einsum(
            "qbi,ij,qbj->qb",
            deviations,
            self._outcome_gram,
            deviations,
            optimize=True,
        )
        squared_norms = np.maximum(squared_norms, 0.0)
        try:
            quantiles = np.quantile(
                squared_norms,
                1.0 - float(alpha),
                axis=1,
                method="higher",
            )
        except TypeError:  # NumPy < 1.22 compatibility.
            quantiles = cast(Callable[..., NDArray], np.quantile)(
                squared_norms,
                1.0 - float(alpha),
                axis=1,
                interpolation="higher",
            )
        return np.asarray(quantiles, dtype=np.float32)

    def test_distributional_effect(
        self,
        x: NDArray,
        *,
        alpha: float = 0.05,
    ) -> NDArray[np.bool_]:
        """Paper test of equality of the two conditional outcome laws."""

        return self.predict_ckte_mmd2(x) > self.resampling_radius2(x, alpha=alpha)

    def witness_confidence_band(
        self,
        x: NDArray,
        outcome: NDArray,
        *,
        alpha: float = 0.05,
        paired: bool = False,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Paper simultaneous-in-outcome witness band at each fixed query x."""

        estimate = self.witness(x, outcome, paired=paired)
        radius = np.sqrt(self.resampling_radius2(x, alpha=alpha))
        if estimate.ndim == 2:
            radius = radius[:, None]
        lower = estimate - radius
        upper = estimate + radius
        return (
            estimate.astype(np.float32),
            lower.astype(np.float32),
            upper.astype(np.float32),
        )
