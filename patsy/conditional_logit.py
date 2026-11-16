# This file is part of Patsy
# Copyright (C) 2024
# See file LICENSE.txt for license information.

# Conditional (McFadden) logistic regression for discrete-choice data, built
# on top of Patsy formulas. Each observation is one alternative inside a
# choice group; the model uses an intercept-free utility specification
# ``u = X @ beta`` and conditions each group's likelihood on the set of
# available alternatives.

# These are made available in the patsy.* namespace:
__all__ = [
    "conditional_logit",
    "predict_conditional_logit",
    "ConditionalLogitResult",
    "ConditionalLogitPrediction",
    "EXCLUSION_FEWER_THAN_TWO_AVAILABLE",
    "EXCLUSION_CHOSEN_UNAVAILABLE",
    "EXCLUSION_NO_CHOICE",
    "EXCLUSION_MULTIPLE_CHOICES",
    "EXCLUSION_MISSING_FEATURE",
    "UNPREDICTABLE_NO_AVAILABLE_ALTERNATIVES",
    "UNPREDICTABLE_MISSING_FEATURE",
    "UNPREDICTABLE_UNKNOWN_CATEGORY",
]

import numpy as np

from patsy import PatsyError
from patsy.build import _eval_factor, build_design_matrices
from patsy.categorical import _CategoricalBox, categorical_to_int
from patsy.desc import INTERCEPT, ModelDesc
from patsy.eval import EvalEnvironment, EvalFactor, ast_names
from patsy.highlevel import incr_dbuilder
from patsy.missing import NAAction
from patsy.util import have_pandas

if have_pandas:
    import pandas


# Group-level exclusion reason codes (training)
EXCLUSION_FEWER_THAN_TWO_AVAILABLE = "fewer_than_two_available"
EXCLUSION_CHOSEN_UNAVAILABLE = "chosen_unavailable"
EXCLUSION_NO_CHOICE = "no_choice"
EXCLUSION_MULTIPLE_CHOICES = "multiple_choices"
EXCLUSION_MISSING_FEATURE = "missing_feature"

# Group-level reason codes (prediction)
UNPREDICTABLE_NO_AVAILABLE_ALTERNATIVES = "no_available_alternatives"
UNPREDICTABLE_MISSING_FEATURE = "missing_feature"
UNPREDICTABLE_UNKNOWN_CATEGORY = "unknown_category"


def _n_rows(data):
    if have_pandas and isinstance(data, pandas.DataFrame):
        return data.shape[0]
    if isinstance(data, dict):
        n = None
        for key, value in data.items():
            this_n = len(value)
            if n is None:
                n = this_n
            elif this_n != n:
                raise PatsyError(
                    "all variables in 'data' must have the same length; "
                    "variable %r has length %s, expected %s" % (key, this_n, n)
                )
        if n is None:
            raise PatsyError("'data' is empty")
        return n
    raise PatsyError("'data' must be a dict-like mapping or a pandas.DataFrame")


def _resolve_values(data, values, name):
    """Resolve a column-name reference, returning an array-like of length n."""
    if isinstance(values, str):
        if hasattr(data, "__contains__") and values in data:
            return data[values]
        raise PatsyError("%s=%r does not name a column in the data" % (name, values))
    if values is None:
        return None
    return values


def _check_length(values, n, name):
    if len(values) != n:
        raise PatsyError(
            "%s has length %s, but the data has %s rows" % (name, len(values), n)
        )


def _truthy_mask(values, n, name):
    """Boolean mask treating None/NaN as False (never selected/available)."""
    arr = np.asarray(values)
    _check_length(arr, n, name)
    if arr.ndim != 1:
        raise PatsyError("%s must be one-dimensional" % (name,))
    kind = arr.dtype.kind
    if kind == "b":
        return arr.astype(bool, copy=True)
    if kind in "iu":
        return arr.astype(bool)
    if kind == "f":
        with np.errstate(invalid="ignore"):
            return (~np.isnan(arr)) & (arr != 0)
    out = np.zeros(n, dtype=bool)
    for i, value in enumerate(arr):
        if _is_na_scalar(value):
            # Missing never means selected/available.
            continue
        out[i] = bool(value)
    return out


def _safe_scalar_nan(value):
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _is_na_scalar(value):
    """Missingness for a single scalar, including pandas' pd.NA/pd.NaT."""
    if value is None:
        return True
    if have_pandas:
        try:
            return bool(pandas.isna(value))
        except (TypeError, ValueError):
            return False
    return _safe_scalar_nan(value)


def _missing_mask(values):
    """Per-row missingness mask for a raw data variable."""
    arr = np.asarray(values)
    n = arr.shape[0]
    if arr.ndim > 1:
        if arr.dtype.kind in "fc":
            return np.any(np.isnan(arr), axis=1)
        flat = arr.reshape(n, -1)
        out = np.zeros(n, dtype=bool)
        for i in range(n):
            for value in flat[i]:
                if _is_na_scalar(value):
                    out[i] = True
                    break
        return out
    if arr.dtype.kind in "fc":
        return np.isnan(arr)
    if arr.dtype.kind in "biubool":
        return np.zeros(n, dtype=bool)
    out = np.zeros(n, dtype=bool)
    for i, value in enumerate(arr):
        if _is_na_scalar(value):
            out[i] = True
    return out


def _group_positions(labels):
    """Insertion-ordered mapping of group label -> physical row positions."""
    groups = {}
    for i, label in enumerate(labels):
        try:
            groups.setdefault(label, []).append(i)
        except TypeError as exc:
            raise PatsyError("group labels must be hashable values") from exc
    return groups


def _parse_rhs_termlist(formula_like):
    if isinstance(formula_like, str):
        model_desc = ModelDesc.from_formula(formula_like)
    elif isinstance(formula_like, ModelDesc):
        model_desc = formula_like
    else:
        raise PatsyError(
            "the formula must be a formula string or a ModelDesc; "
            "a fitted DesignInfo cannot be used to fit a new model"
        )
    if model_desc.lhs_termlist:
        raise PatsyError(
            "conditional logit formulas are intercept-free predictor-only "
            "formulas; do not put a response on the left-hand side of '~'"
        )
    rhs = model_desc.rhs_termlist
    if INTERCEPT in rhs:
        raise PatsyError(
            "conditional logit requires an intercept-free formula; "
            "remove the intercept, e.g. use '~ 0 + ...' instead of '~ ...'"
        )
    return rhs


def _factor_data_names(termlist, data_keys):
    names = set()
    for term in termlist:
        for factor in term.factors:
            if isinstance(factor, EvalFactor):
                for name in ast_names(factor.code):
                    if name in data_keys:
                        names.add(name)
    return names


def _data_keys(data):
    if have_pandas and isinstance(data, pandas.DataFrame):
        return set(data.columns)
    return set(data.keys())


def _take_rows(data, positions):
    """Position-based row subsetting; never relies on (possibly duplicate)
    pandas indexes."""
    positions = np.asarray(positions)
    if have_pandas and isinstance(data, pandas.DataFrame):
        return data.iloc[positions]
    out = {}
    for key, value in data.items():
        if have_pandas and isinstance(value, (pandas.Series, pandas.DataFrame)):
            out[key] = value.iloc[positions]
        else:
            arr = np.asarray(value)
            out[key] = arr[positions] if arr.ndim == 1 else arr[positions, :]
    return out


def _check_lambda(lambda_penalty):
    if not isinstance(lambda_penalty, (int, float, np.number)):
        raise ValueError("lambda_penalty must be a positive number")
    if not np.isfinite(lambda_penalty) or lambda_penalty <= 0:
        raise ValueError("lambda_penalty must be a positive, finite number")


def _fit_beta(X, group_local_rows, group_chosen_local, n_params, lambda_penalty):
    def objective(beta):
        utilities = X.dot(beta)
        nll = 0.0
        grad = np.zeros(n_params)
        for rows, chosen_local in zip(group_local_rows, group_chosen_local):
            u = utilities[rows]
            u_max = u.max()
            exp_u = np.exp(u - u_max)
            denom = exp_u.sum()
            nll += -u[chosen_local] + u_max + np.log(denom)
            probs = exp_u / denom
            residual = probs.copy()
            residual[chosen_local] -= 1.0
            grad += X[rows].T.dot(residual)
        penalty = 0.5 * lambda_penalty * beta.dot(beta)
        grad += lambda_penalty * beta
        return nll + penalty, grad

    beta0 = np.zeros(n_params)
    if n_params == 0:
        objective_value, _ = objective(beta0)
        return beta0, objective_value, True, "no parameters to optimize"
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError(
            "Fitting a conditional logit model requires scipy. "
            "Install scipy to use this functionality."
        ) from exc
    result = minimize(
        objective,
        beta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000},
    )
    return result.x, float(result.fun), bool(result.success), str(result.message)


class ConditionalLogitResult(object):
    """A fitted conditional (McFadden) logit model.

    Attributes:

    .. attribute:: coef

       1-d ndarray of fitted coefficients, aligned with
       :attr:`coef_names` and the columns of :attr:`design_info`.

    .. attribute:: coef_names

       List of design-matrix column names.

    .. attribute:: design_info

       The fitted :class:`patsy.DesignInfo`. New choice sets are transformed
       with this exact object, so stateful transforms (``center``,
       ``standardize``) and categorical levels are reused, never re-learned.

    .. attribute:: lambda_penalty

       The ridge penalty used when fitting.

    .. attribute:: neg_log_likelihood

       Sum of per-group negative log-likelihoods at the fitted coefficients,
       without the ridge penalty.

    .. attribute:: penalized_objective

       The minimized value: negative log-likelihood plus
       ``lambda_penalty * ||beta||^2 / 2``.

    .. attribute:: converged

       Whether the optimizer reported convergence.

    .. attribute:: group_info

       Dict keyed by group label, in order of first appearance, with each
       value describing the group's physical rows, availability, chosen row
       (when identifiable), usage status and exclusion reason.

    .. attribute:: row_info

       Dict of length-n ndarrays, aligned by physical row: ``groups``,
       ``available``, ``chosen``, ``used_in_fit``, ``excluded``,
       ``exclusion_reason``.
    """

    def __init__(
        self,
        coef,
        coef_names,
        design_info,
        lambda_penalty,
        neg_log_likelihood,
        penalized_objective,
        converged,
        message,
        group_info,
        row_info,
        raw_var_names,
    ):
        self.coef = coef
        self.coef_names = list(coef_names)
        self.design_info = design_info
        self.lambda_penalty = lambda_penalty
        self.neg_log_likelihood = neg_log_likelihood
        self.penalized_objective = penalized_objective
        self.converged = converged
        self.message = message
        self.group_info = group_info
        self.row_info = row_info
        self.used_group_labels = [
            label for label, info in group_info.items() if info["used"]
        ]
        self.excluded_group_labels = [
            label for label, info in group_info.items() if not info["used"]
        ]
        self.used_row_positions = np.flatnonzero(row_info["used_in_fit"]).astype(int)
        self.excluded_row_positions = np.flatnonzero(row_info["excluded"]).astype(int)
        self.n_groups = len(group_info)
        self._raw_var_names = raw_var_names

    @property
    def coef_dict(self):
        """Coefficients as an ordered dict keyed by column name."""
        return dict(zip(self.coef_names, self.coef))

    def predict(self, data, groups, available=None):
        """Predict choice probabilities and choices for new choice sets.

        See :func:`predict_conditional_logit`; this method calls it with this
        fitted result. The fitted :class:`DesignInfo` is reused, so nothing is
        re-fitted.
        """
        return predict_conditional_logit(self, data, groups, available)

    def row_frame(self):
        """Return the physical-row-aligned :attr:`row_info` as a DataFrame."""
        if not have_pandas:
            raise PatsyError("pandas is not installed")
        frame = pandas.DataFrame(self.row_info)
        frame.insert(0, "row_position", np.arange(frame.shape[0]))
        return frame


class ConditionalLogitPrediction(object):
    """Predictions on new choice sets, aligned by physical row.

    Attributes:

    .. attribute:: probabilities

       Length-n ndarray. Within a predictable group, the probabilities of the
       available alternatives sum to 1; unavailable alternatives get 0. Rows
       of unpredictable groups carry NaN (except unavailable rows, which stay
       0).

    .. attribute:: chosen

       Length-n boolean ndarray marking the selected alternative of each
       predictable group. Ties are resolved in favour of the alternative with
       the smallest physical row number. All rows of unpredictable groups and
       all unavailable rows are False.

    .. attribute:: chosen_positions

       Physical row positions of the selected alternatives, ordered by each
       group's first appearance.

    .. attribute:: group_info

       Dict keyed by group label with per-group availability, predictability
       status, reason (when unpredictable), chosen position and a
       ``probabilities`` mapping of physical row position to probability.

    .. attribute:: row_info

       Length-n ndarrays ``groups``, ``available``, ``predictable`` and
       ``unpredictable_reason``, aligned by physical row.
    """

    def __init__(self, probabilities, chosen, chosen_positions, group_info, row_info):
        self.probabilities = probabilities
        self.chosen = chosen
        self.chosen_positions = np.asarray(chosen_positions, dtype=int)
        self.group_info = group_info
        self.row_info = row_info
        self.predictable_group_labels = [
            label for label, info in group_info.items() if info["predictable"]
        ]
        self.unpredictable_group_labels = [
            label for label, info in group_info.items() if not info["predictable"]
        ]

    def row_frame(self):
        if not have_pandas:
            raise PatsyError("pandas is not installed")
        frame = pandas.DataFrame(self.row_info)
        frame.insert(0, "row_position", np.arange(frame.shape[0]))
        frame["probability"] = self.probabilities
        frame["chosen"] = self.chosen
        return frame


def conditional_logit(
    formula_like,
    data,
    groups,
    chosen,
    lambda_penalty,
    available=None,
    eval_env=0,
):
    """Fit a conditional logit (discrete choice) model with a Patsy formula.

    :arg formula_like: An intercept-free formula string or :class:`ModelDesc`,
      e.g. ``"~ 0 + price + C(brand) + price:C(type) + standardize(size)"``.
      Numeric terms, :func:`C` categorical terms, interactions and stateful
      transforms such as :func:`center` / :func:`standardize` are all
      supported. Intercept terms and left-hand-side responses are rejected.
    :arg data: A dict-like mapping or :class:`pandas.DataFrame` holding the
      formula variables, one alternative per physical row.
    :arg groups: Array-like of group labels, length equal to the number of
      physical rows. Repeated labels identify alternatives of the same choice
      set; labels need not be contiguous, and duplicate pandas indexes are
      handled positionally.
    :arg chosen: Boolean array-like (or a data column name) marking the
      actually chosen alternative. Missing values count as not chosen.
    :arg lambda_penalty: Positive, finite ridge strength. The fitted objective
      is the sum over groups of the negative log-likelihood plus
      ``lambda_penalty * ||beta||^2 / 2``.
    :arg available: Optional boolean array-like (or column name) marking
      usable alternatives; missing values mean unavailable. Defaults to every
      alternative being available.
    :arg eval_env: Patsy :class:`EvalEnvironment` or capture depth.

    Training groups must have at least two available alternatives and exactly
    one chosen available alternative. Invalid groups, and groups whose
    available alternatives have missing formula features, are excluded *as
    whole groups*. Exclusion happens before any formula feature is learned, so
    stateful-transform statistics and categorical levels are computed only
    from retained rows. Missingness is checked for variables looked up in
    `data`; formula variables resolved through `eval_env` instead are not
    visible to the pre-check and may raise during design-matrix construction.

    :returns: A :class:`ConditionalLogitResult` with coefficients and full
      usage/exclusion details.
    """
    eval_env = EvalEnvironment.capture(eval_env, reference=1)
    _check_lambda(lambda_penalty)
    n = _n_rows(data)

    groups_ref = _resolve_values(data, groups, "groups")
    chosen_ref = _resolve_values(data, chosen, "chosen")
    labels = np.asarray(groups_ref)
    _check_length(labels, n, "groups")
    if labels.ndim != 1:
        raise PatsyError("'groups' must be one-dimensional")
    chosen_mask = _truthy_mask(chosen_ref, n, "chosen")
    if available is None:
        available_mask = np.ones(n, dtype=bool)
    else:
        available_ref = _resolve_values(data, available, "available")
        available_mask = _truthy_mask(available_ref, n, "available")

    rhs = _parse_rhs_termlist(formula_like)
    data_keys = _data_keys(data)
    raw_var_names = _factor_data_names(rhs, data_keys)

    raw_missing = np.zeros(n, dtype=bool)
    for name in raw_var_names:
        raw_missing |= _missing_mask(data[name])

    ordered_groups = _group_positions(labels)
    group_info = {}
    for label, positions in ordered_groups.items():
        positions = np.asarray(positions, dtype=int)
        available_positions = positions[available_mask[positions]]
        chosen_positions = positions[chosen_mask[positions]]
        chosen_available = positions[chosen_mask[positions] & available_mask[positions]]
        reason = None
        if len(available_positions) < 2:
            reason = EXCLUSION_FEWER_THAN_TWO_AVAILABLE
        elif np.any(~available_mask[chosen_positions]):
            reason = EXCLUSION_CHOSEN_UNAVAILABLE
        elif len(chosen_available) == 0:
            reason = EXCLUSION_NO_CHOICE
        elif len(chosen_available) > 1:
            reason = EXCLUSION_MULTIPLE_CHOICES
        elif np.any(raw_missing[available_positions]):
            # Whole-group exclusion: a missing feature on any available
            # alternative makes the group unusable.
            reason = EXCLUSION_MISSING_FEATURE
        group_info[label] = {
            "row_positions": positions,
            "available_positions": available_positions,
            "chosen_positions": chosen_positions,
            "chosen_position": (
                int(chosen_available[0]) if len(chosen_available) == 1 else None
            ),
            "used": reason is None,
            "exclusion_reason": reason,
        }

    used_labels = [label for label, info in group_info.items() if info["used"]]
    if not used_labels:
        raise PatsyError(
            "no valid choice groups remain after exclusion; see "
            "ConditionalLogitResult.group_info for reasons"
        )

    # Exclude first, learn features second: the DesignInfo (stateful transform
    # statistics, categorical levels, ...) is built only from retained rows.
    learn_positions = np.sort(
        np.concatenate(
            [group_info[label]["available_positions"] for label in used_labels]
        )
    )
    learn_data = _take_rows(data, learn_positions)
    design_info = incr_dbuilder(
        ModelDesc([], rhs),
        lambda: iter([learn_data]),
        eval_env,
    )
    if len(design_info.column_names) == 0:
        # An intercept-free formula with no actual factors has no data
        # dependence; patsy itself cannot infer the row count from a dict.
        X = np.zeros((len(learn_positions), 0), dtype=float)
    else:
        design_matrix = build_design_matrices(
            [design_info], learn_data, NA_action=NAAction("raise")
        )[0]
        X = np.asarray(design_matrix, dtype=float)
    n_params = X.shape[1]

    local_index = np.full(n, -1, dtype=int)
    local_index[learn_positions] = np.arange(len(learn_positions))
    group_local_rows = []
    group_chosen_local = []
    for label in used_labels:
        info = group_info[label]
        local_rows = local_index[info["available_positions"]]
        chosen_x_index = local_index[info["chosen_position"]]
        chosen_within_group = int(np.where(local_rows == chosen_x_index)[0][0])
        group_local_rows.append(local_rows)
        group_chosen_local.append(chosen_within_group)

    beta, penalized_objective, converged, message = _fit_beta(
        X, group_local_rows, group_chosen_local, n_params, lambda_penalty
    )
    utilities = X.dot(beta)
    neg_log_likelihood = 0.0
    for local_rows, chosen_local in zip(group_local_rows, group_chosen_local):
        u = utilities[local_rows]
        u_max = u.max()
        neg_log_likelihood += -u[chosen_local] + u_max + np.log(np.exp(u - u_max).sum())
    neg_log_likelihood = float(neg_log_likelihood)

    used_in_fit = np.zeros(n, dtype=bool)
    used_in_fit[learn_positions] = True
    reason_by_row = np.full(n, None, dtype=object)
    excluded_rows = np.zeros(n, dtype=bool)
    for label, info in group_info.items():
        if not info["used"]:
            reason_by_row[info["row_positions"]] = info["exclusion_reason"]
            excluded_rows[info["row_positions"]] = True
    row_info = {
        "groups": labels,
        "available": available_mask,
        "chosen": chosen_mask,
        "used_in_fit": used_in_fit,
        "excluded": excluded_rows,
        "exclusion_reason": reason_by_row,
    }

    return ConditionalLogitResult(
        coef=beta,
        coef_names=design_info.column_names,
        design_info=design_info,
        lambda_penalty=float(lambda_penalty),
        neg_log_likelihood=neg_log_likelihood,
        penalized_objective=penalized_objective,
        converged=converged,
        message=message,
        group_info=group_info,
        row_info=row_info,
        raw_var_names=raw_var_names,
    )


def _factor_row_problems(design_info, data, n):
    """Return (missing rows, unknown-category rows) masks by reusing the
    fitted factor states. No DesignInfo is learned here."""
    missing = np.zeros(n, dtype=bool)
    unknown = np.zeros(n, dtype=bool)
    na_action = NAAction()
    for factor_info in design_info.factor_infos.values():
        if factor_info.type == "numerical":
            _, is_na = _eval_factor(factor_info, data, na_action)
            missing |= np.asarray(is_na)
        else:
            raw = factor_info.factor.eval(factor_info.state, data)
            try:
                codes = categorical_to_int(raw, factor_info.categories, na_action)
                missing |= np.asarray(codes == -1)
            except PatsyError:
                # At least one unknown level. Locate the offending rows
                # explicitly so one bad row does not abort the whole call.
                values = raw.data if isinstance(raw, _CategoricalBox) else raw
                values = np.asarray(values)
                levels = set(factor_info.categories)
                for i, value in enumerate(values):
                    if _is_na_scalar(value) or na_action.is_categorical_NA(value):
                        missing[i] = True
                    elif value not in levels:
                        unknown[i] = True
    return missing, unknown


def predict_conditional_logit(result, data, groups, available=None):
    """Predict choice probabilities and choices using a fitted model.

    :arg result: A fitted :class:`ConditionalLogitResult`.
    :arg data: New alternatives in the same dict-like / DataFrame form used
      for fitting. They are transformed with the fitted
      :class:`DesignInfo`, which is reused as-is -- prediction never refits
      stateful transforms or categorical levels.
    :arg groups: Group labels for the new rows.
    :arg available: Optional availability mask; missing means unavailable.
    :returns: A :class:`ConditionalLogitPrediction`.

    Within each predictable group, softmax probabilities are computed over
    the available alternatives and the alternatives of unavailable rows get
    probability 0. A group is entirely unpredictable when it has no available
    alternative, when any available alternative has a missing feature, or when
    it contains a categorical level unseen during training; its rows then
    carry NaN probabilities and no choice. Probability ties are resolved by
    selecting the alternative with the smallest physical row number. All
    outputs are aligned to physical input rows, so duplicate indexes are
    safe.
    """
    if not isinstance(result, ConditionalLogitResult):
        raise PatsyError(
            "'result' must be a ConditionalLogitResult returned by conditional_logit()"
        )
    design_info = result.design_info
    n = _n_rows(data)
    groups_ref = _resolve_values(data, groups, "groups")
    labels = np.asarray(groups_ref)
    _check_length(labels, n, "groups")
    if labels.ndim != 1:
        raise PatsyError("'groups' must be one-dimensional")
    if available is None:
        available_mask = np.ones(n, dtype=bool)
    else:
        available_ref = _resolve_values(data, available, "available")
        available_mask = _truthy_mask(available_ref, n, "available")

    # Raw-variable guard (also catches missingness inside stateful
    # transforms) plus factor-level checks using the fitted states.
    raw_missing = np.zeros(n, dtype=bool)
    for name in result._raw_var_names:
        if hasattr(data, "__contains__") and name in data:
            raw_missing |= _missing_mask(data[name])
    factor_missing, factor_unknown = _factor_row_problems(design_info, data, n)
    missing_rows = raw_missing | factor_missing
    unknown_rows = factor_unknown

    ordered_groups = _group_positions(labels)
    build_positions = []
    group_info = {}
    for label, positions in ordered_groups.items():
        positions = np.asarray(positions, dtype=int)
        available_positions = positions[available_mask[positions]]
        reason = None
        if len(available_positions) == 0:
            reason = UNPREDICTABLE_NO_AVAILABLE_ALTERNATIVES
        elif np.any(missing_rows[available_positions]):
            reason = UNPREDICTABLE_MISSING_FEATURE
        elif np.any(unknown_rows[available_positions]):
            reason = UNPREDICTABLE_UNKNOWN_CATEGORY
        predictable = reason is None
        if predictable:
            build_positions.extend(available_positions.tolist())
        group_info[label] = {
            "row_positions": positions,
            "available_positions": available_positions,
            "predictable": predictable,
            "reason": reason,
            "chosen_position": None,
            "probabilities": {},
        }

    probabilities = np.full(n, np.nan, dtype=float)
    # Unavailable rows are always exactly 0, even inside unpredictable groups.
    probabilities[~available_mask] = 0.0
    chosen_mask = np.zeros(n, dtype=bool)
    chosen_positions = []

    if build_positions:
        build_positions = np.sort(np.asarray(build_positions, dtype=int))
        new_data = _take_rows(data, build_positions)
        if len(design_info.column_names) == 0:
            X = np.zeros((len(build_positions), 0), dtype=float)
        else:
            # NA_action="raise" is a safety net: every row here was vetted
            # above.
            design_matrix = build_design_matrices(
                [design_info], new_data, NA_action=NAAction("raise")
            )[0]
            X = np.asarray(design_matrix, dtype=float)
        local_index = np.full(n, -1, dtype=int)
        local_index[build_positions] = np.arange(len(build_positions))
        utilities_all = X.dot(result.coef)

        for label, info in group_info.items():
            if not info["predictable"]:
                continue
            available_positions = info["available_positions"]
            local_rows = local_index[available_positions]
            u = utilities_all[local_rows]
            u_max = u.max()
            exp_u = np.exp(u - u_max)
            probs = exp_u / exp_u.sum()
            probabilities[available_positions] = probs
            # np.argmax returns the first maximum; available positions are
            # stored in ascending physical order, so this is the physical-row
            # tie-break the contract requires.
            chosen_local = int(np.argmax(probs))
            chosen_position = int(available_positions[chosen_local])
            chosen_mask[chosen_position] = True
            chosen_positions.append(chosen_position)
            info["chosen_position"] = chosen_position
            info["probabilities"] = dict(zip(available_positions.tolist(), probs))

    reason_by_row = np.full(n, None, dtype=object)
    predictable_rows = np.zeros(n, dtype=bool)
    for label, info in group_info.items():
        if info["predictable"]:
            predictable_rows[info["row_positions"]] = True
        else:
            reason_by_row[info["row_positions"]] = info["reason"]
    row_info = {
        "groups": labels,
        "available": available_mask,
        "predictable": predictable_rows,
        "unpredictable_reason": reason_by_row,
    }

    return ConditionalLogitPrediction(
        probabilities=probabilities,
        chosen=chosen_mask,
        chosen_positions=chosen_positions,
        group_info=group_info,
        row_info=row_info,
    )
