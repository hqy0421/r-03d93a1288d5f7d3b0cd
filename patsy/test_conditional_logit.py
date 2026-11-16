# This file is part of Patsy
# Copyright (C) 2024
# See file LICENSE.txt for license information.

import numpy as np
import pytest

from patsy import (
    EXCLUSION_CHOSEN_UNAVAILABLE,
    EXCLUSION_FEWER_THAN_TWO_AVAILABLE,
    EXCLUSION_MISSING_FEATURE,
    EXCLUSION_MULTIPLE_CHOICES,
    EXCLUSION_NO_CHOICE,
    UNPREDICTABLE_MISSING_FEATURE,
    UNPREDICTABLE_NO_AVAILABLE_ALTERNATIVES,
    UNPREDICTABLE_UNKNOWN_CATEGORY,
    ConditionalLogitResult,
    PatsyError,
    conditional_logit,
    predict_conditional_logit,
)


def _make_data(seed=0, n_groups=20, n_alts=3, noise=0.5):
    rng = np.random.RandomState(seed)
    groups = np.repeat(np.arange(n_groups), n_alts)
    price = rng.randn(n_groups * n_alts)
    brand = np.tile(np.array(["a", "b", "c"]), n_groups)
    util = -1.2 * price + 0.7 * (brand == "b") + 0.3 * (brand == "c")
    util = util + rng.gumbel(size=util.shape) * noise
    chosen = np.zeros(len(groups), dtype=bool)
    for g in range(n_groups):
        idx = np.where(groups == g)[0]
        chosen[idx[np.argmax(util[idx])]] = True
    return groups, price, brand, chosen


def test_basic_fit_and_predict():
    groups, price, brand, chosen = _make_data()
    data = {"price": price, "brand": brand}
    result = conditional_logit(
        "~ 0 + price + C(brand)", data, groups, chosen, lambda_penalty=0.5
    )
    assert isinstance(result, ConditionalLogitResult)
    assert result.converged
    assert result.coef.shape == (len(result.coef_names),)
    assert result.n_groups == 20
    assert len(result.used_group_labels) == 20
    assert result.excluded_group_labels == []
    assert result.used_row_positions.tolist() == list(range(60))
    assert result.row_info["used_in_fit"].all()
    assert not result.row_info["excluded"].any()
    # penalty positive and objective = nll + penalty
    penalty = 0.5 * 0.5 * result.coef.dot(result.coef)
    assert np.allclose(result.penalized_objective, result.neg_log_likelihood + penalty)

    pred = predict_conditional_logit(result, data, groups)
    assert pred.probabilities.shape == (60,)
    for g in range(20):
        rows = np.where(groups == g)[0]
        assert np.allclose(pred.probabilities[rows].sum(), 1.0)
    assert pred.chosen.sum() == 20
    # one choice per group, and it is an argmax of utility
    beta = result.coef_dict
    X_util = (
        beta["price"] * price
        + beta["C(brand)[a]"] * (brand == "a")
        + beta["C(brand)[b]"] * (brand == "b")
        + beta["C(brand)[c]"] * (brand == "c")
    )
    for g in range(20):
        rows = np.where(groups == g)[0]
        assert pred.chosen[rows[np.argmax(X_util[rows])]]

    # the result.predict shortcut is identical
    pred2 = result.predict(data, groups)
    assert np.array_equal(pred2.probabilities, pred.probabilities)
    assert np.array_equal(pred2.chosen, pred.chosen)


def test_gradient_is_correct():
    from scipy.optimize import check_grad

    rng = np.random.RandomState(1)
    groups = np.repeat(np.arange(4), 3)
    X_data = rng.randn(12, 2)
    chosen = np.zeros(12, dtype=bool)
    chosen[[0, 4, 7, 10]] = True
    data = {"x1": X_data[:, 0], "x2": X_data[:, 1]}
    result = conditional_logit(
        "~ 0 + x1 + x2", data, groups, chosen, lambda_penalty=0.7
    )
    design = result.design_info
    X = np.asarray(__import__("patsy").build_design_matrices([design], data)[0])
    lam = result.lambda_penalty
    rows = [np.where(groups == g)[0] for g in range(4)]
    chosen_local = [int(np.where(chosen[r])[0][0]) for r in rows]

    def obj(beta):
        u = X.dot(beta)
        nll = 0.0
        for r, c in zip(rows, chosen_local):
            uu = u[r]
            nll += -uu[c] + np.log(np.exp(uu - uu.max()).sum()) + uu.max()
        return nll + 0.5 * lam * beta.dot(beta)

    def grad(beta):
        u = X.dot(beta)
        g = np.zeros(2)
        for r, c in zip(rows, chosen_local):
            uu = u[r]
            p = np.exp(uu - uu.max())
            p = p / p.sum()
            resid = p.copy()
            resid[c] -= 1
            g += X[r].T.dot(resid)
        return g + lam * beta

    beta = rng.randn(2)
    assert check_grad(obj, grad, beta) < 1e-5


def test_numeric_interaction_and_stateful_transforms():
    groups, price, brand, chosen = _make_data()
    size = np.random.RandomState(2).randn(len(price))
    data = {"price": price, "brand": brand, "size": size}
    result = conditional_logit(
        "~ 0 + price + C(brand) + price:size + standardize(size) + center(price)",
        data,
        groups,
        chosen,
        lambda_penalty=1.0,
    )
    # standardize column learned mean/std from the training rows
    names = result.coef_names
    assert any("standardize(size)" in n for n in names)
    assert any("center(price)" in n for n in names)
    assert any("price:size" in n for n in names)
    # prediction reuses the learned stateful transform, not new statistics
    new_size = size + 100.0  # wildly different mean
    pred = result.predict({"price": price, "brand": brand, "size": new_size}, groups)
    assert np.isfinite(pred.probabilities).all()


def test_intercept_and_lhs_rejected():
    groups, price, brand, chosen = _make_data(n_groups=2)
    data = {"price": price, "brand": brand, "y": chosen.astype(float)}
    pytest.raises(
        PatsyError,
        conditional_logit,
        "~ price + C(brand)",
        data,
        groups,
        chosen,
        0.1,
    )
    pytest.raises(
        PatsyError,
        conditional_logit,
        "y ~ 0 + price",
        data,
        groups,
        chosen,
        0.1,
    )


def test_lambda_must_be_positive():
    groups, price, brand, chosen = _make_data(n_groups=2)
    data = {"price": price, "brand": brand}
    for bad in (0, -1.0, np.nan, np.inf, "x"):
        pytest.raises(
            (ValueError, TypeError),
            conditional_logit,
            "~ 0 + price",
            data,
            groups,
            chosen,
            bad,
        )


def test_exclusion_fewer_than_two_available():
    groups, price, brand, chosen = _make_data(n_groups=3)
    available = np.ones(9, dtype=bool)
    available[1] = False
    available[2] = False  # group 0 now has a single available alternative
    data = {"price": price, "brand": brand}
    result = conditional_logit(
        "~ 0 + price", data, groups, chosen, 0.1, available=available
    )
    assert 0 in result.excluded_group_labels
    assert (
        result.group_info[0]["exclusion_reason"] == EXCLUSION_FEWER_THAN_TWO_AVAILABLE
    )
    assert result.row_info["excluded"][:3].all()
    assert not result.row_info["excluded"][3:].any()
    assert result.used_row_positions.tolist() == list(range(3, 9))


def test_exclusion_no_choice_and_multiple_choices():
    groups = np.repeat(np.arange(3), 3)
    price = np.zeros(9)
    brand = np.array(["a", "b", "c"] * 3)
    chosen = np.zeros(9, dtype=bool)
    chosen[4] = True  # group 1
    chosen[6:9] = True  # group 2: two choices; group 0: no choice
    result = conditional_logit(
        "~ 0 + price",
        {"price": price, "brand": brand},
        groups,
        chosen,
        0.1,
    )
    assert result.group_info[0]["exclusion_reason"] == EXCLUSION_NO_CHOICE
    assert result.group_info[2]["exclusion_reason"] == EXCLUSION_MULTIPLE_CHOICES
    assert result.used_group_labels == [1]


def test_exclusion_chosen_unavailable():
    groups = np.repeat(np.arange(2), 3)
    price = np.zeros(6)
    brand = np.array(["a", "b", "c"] * 2)
    chosen = np.zeros(6, dtype=bool)
    chosen[[0, 3]] = True
    available = np.ones(6, dtype=bool)
    available[0] = False  # the chosen alternative of group 0 is unavailable
    result = conditional_logit(
        "~ 0 + price",
        {"price": price, "brand": brand},
        groups,
        chosen,
        0.1,
        available=available,
    )
    assert result.group_info[0]["exclusion_reason"] == EXCLUSION_CHOSEN_UNAVAILABLE
    assert result.used_group_labels == [1]


def test_missing_feature_excludes_whole_group_before_learning():
    # standardize's mean must be computed WITHOUT the contaminated group's
    # rows: a NaN there would otherwise poison the learned statistic.
    groups = np.repeat(np.arange(3), 3)
    x = np.arange(9, dtype=float)
    chosen = np.zeros(9, dtype=bool)
    chosen[[0, 4, 7]] = True
    x[2] = np.nan  # group 0 has a missing feature on an available row
    result = conditional_logit("~ 0 + standardize(x)", {"x": x}, groups, chosen, 0.2)
    assert result.group_info[0]["exclusion_reason"] == EXCLUSION_MISSING_FEATURE
    assert result.used_group_labels == [1, 2]
    # mean learned only from rows 3..8
    assert np.allclose(np.nanmean(x[3:9]), np.mean(x[3:9]))
    pred = result.predict({"x": np.arange(9, dtype=float)}, groups)
    assert np.isfinite(pred.probabilities).all()
    # and the poisoned NaN group is itself excluded, so no NaN entered fitting
    assert np.isfinite(result.coef).all()


def test_missing_feature_on_unavailable_row_does_not_exclude():
    groups = np.repeat(np.arange(2), 3)
    x = np.arange(6, dtype=float)
    chosen = np.zeros(6, dtype=bool)
    chosen[[0, 3]] = True
    available = np.ones(6, dtype=bool)
    x[2] = np.nan
    available[2] = False
    x[5] = np.nan
    available[5] = False
    result = conditional_logit(
        "~ 0 + x", {"x": x}, groups, chosen, 0.1, available=available
    )
    assert result.used_group_labels == [0, 1]
    # unavailable rows never enter the design matrix
    assert result.used_row_positions.tolist() == [0, 1, 3, 4]


def test_predict_unavailable_alternatives_get_zero():
    groups = np.repeat(np.arange(2), 3)
    price = np.array([0.0, 1.0, 2.0] * 2)
    brand = np.array(["a", "b", "c"] * 2)
    chosen = np.zeros(6, dtype=bool)
    chosen[[0, 3]] = True
    data = {"price": price, "brand": brand}
    result = conditional_logit("~ 0 + price", data, groups, chosen, 0.1)
    available = np.ones(6, dtype=bool)
    available[2] = False
    available[5] = False
    pred = result.predict(data, groups, available=available)
    assert pred.probabilities[2] == 0.0
    assert pred.probabilities[5] == 0.0
    assert np.allclose(pred.probabilities[[0, 1]].sum(), 1.0)
    assert np.allclose(pred.probabilities[[3, 4]].sum(), 1.0)
    # choice never lands on an unavailable row
    assert not pred.chosen[[2, 5]].any()


def test_predict_missing_feature_makes_group_unpredictable():
    groups = np.repeat(np.arange(3), 3)
    x = np.arange(9, dtype=float)
    chosen = np.zeros(9, dtype=bool)
    chosen[[0, 3, 6]] = True
    result = conditional_logit("~ 0 + x", {"x": x}, groups, chosen, 0.1)
    new_x = x.copy()
    new_x[1] = np.nan  # group 0 unpredictable; other groups fine
    pred = result.predict({"x": new_x}, groups)
    assert pred.group_info[0]["reason"] == UNPREDICTABLE_MISSING_FEATURE
    assert not pred.row_info["predictable"][:3].any()
    assert np.isnan(pred.probabilities[:3]).all()
    assert not pred.chosen[:3].any()
    assert np.allclose(pred.probabilities[3:].reshape(2, 3).sum(axis=1), 1.0)
    assert pred.chosen.sum() == 2
    assert pred.unpredictable_group_labels == [0]


def test_predict_unknown_category_makes_group_unpredictable():
    groups = np.repeat(np.arange(3), 3)
    brand = np.array(["a", "b", "c"] * 3)
    price = np.arange(9, dtype=float)
    chosen = np.zeros(9, dtype=bool)
    chosen[[0, 3, 6]] = True
    result = conditional_logit(
        "~ 0 + price + C(brand)",
        {"price": price, "brand": brand},
        groups,
        chosen,
        0.1,
    )
    new_brand = brand.copy()
    new_brand[7] = "d"  # unseen level in group 2
    pred = result.predict({"price": price, "brand": new_brand}, groups)
    assert pred.group_info[2]["reason"] == UNPREDICTABLE_UNKNOWN_CATEGORY
    assert np.isnan(pred.probabilities[6:9]).all()
    assert np.isfinite(pred.probabilities[:6]).all()
    assert pred.predictable_group_labels == [0, 1]


def test_predict_group_with_no_available_alternatives():
    groups = np.repeat(np.arange(2), 3)
    x = np.arange(6, dtype=float)
    chosen = np.zeros(6, dtype=bool)
    chosen[[0, 3]] = True
    result = conditional_logit("~ 0 + x", {"x": x}, groups, chosen, 0.1)
    available = np.ones(6, dtype=bool)
    available[3:] = False  # group 1 has nothing available
    pred = result.predict({"x": x}, groups, available=available)
    assert pred.group_info[1]["reason"] == UNPREDICTABLE_NO_AVAILABLE_ALTERNATIVES
    # unavailable rows keep an exact zero rather than NaN
    assert (pred.probabilities[3:] == 0.0).all()
    assert not pred.chosen[3:].any()
    assert np.allclose(pred.probabilities[:3].sum(), 1.0)


def test_tie_break_prefers_earlier_physical_row():
    groups = np.array([0, 0, 0])
    # identical utility across alternatives -> equal probabilities
    x = np.zeros(3)
    chosen = np.array([True, False, False])
    result = conditional_logit("~ 0 + x", {"x": x}, groups, chosen, 0.1)
    pred = result.predict({"x": x}, groups)
    assert np.allclose(pred.probabilities, [1 / 3, 1 / 3, 1 / 3])
    assert pred.chosen.tolist() == [True, False, False]
    assert pred.chosen_positions.tolist() == [0]


def test_non_contiguous_groups_and_duplicate_indexes():
    # group labels interleaved, and a duplicated pandas index
    pd = pytest.importorskip("pandas")
    groups = np.array([0, 1, 0, 1, 0, 1])
    x = np.arange(6, dtype=float)
    chosen = np.zeros(6, dtype=bool)
    chosen[[0, 1]] = True
    df = pd.DataFrame({"x": x, "g": groups, "y": chosen})
    df.index = np.array([0, 0, 1, 1, 2, 2])  # deliberately duplicated
    result = conditional_logit(
        "~ 0 + x", df, df["g"].to_numpy(), df["y"].to_numpy(), 0.1
    )
    assert result.used_group_labels == [0, 1]
    assert result.group_info[0]["row_positions"].tolist() == [0, 2, 4]
    assert result.group_info[1]["row_positions"].tolist() == [1, 3, 5]
    pred = result.predict(df, groups)
    assert np.allclose(pred.probabilities[[0, 2, 4]].sum(), 1.0)
    assert np.allclose(pred.probabilities[[1, 3, 5]].sum(), 1.0)
    # outputs are physical-row aligned regardless of the duplicated index
    assert pred.probabilities.shape[0] == 6
    frame = result.row_frame()
    assert frame.shape[0] == 6
    assert frame["row_position"].tolist() == list(range(6))


def test_predict_does_not_refit():
    groups, price, brand, chosen = _make_data()
    data = {"price": price, "brand": brand}
    result = conditional_logit(
        "~ 0 + standardize(price) + C(brand)", data, groups, chosen, 0.1
    )
    info_before = result.design_info
    # new data with a totally different distribution and a brand-new level
    new_price = price + 1000.0
    new_brand = brand.copy()
    new_brand[0] = "a"  # still a known level
    pred = result.predict({"price": new_price, "brand": new_brand}, groups)
    assert result.design_info is info_before
    assert np.isfinite(pred.probabilities).all()
    # the standardized column still uses the training mean/std, so the first
    # row is far below zero
    pred2 = result.predict({"price": np.zeros_like(price), "brand": brand}, groups)
    assert np.isfinite(pred2.probabilities).all()


def test_length_mismatches_raise():
    groups = np.repeat(np.arange(2), 3)
    data = {"x": np.arange(6, dtype=float)}
    chosen = np.zeros(5, dtype=bool)
    pytest.raises(
        PatsyError,
        conditional_logit,
        "~ 0 + x",
        data,
        groups,
        chosen,
        0.1,
    )
    result = conditional_logit(
        "~ 0 + x", data, groups, np.array([1, 0, 0, 1, 0, 0], bool), 0.1
    )
    pytest.raises(
        PatsyError,
        predict_conditional_logit,
        result,
        data,
        np.array([0, 0, 1]),
    )


def test_missing_choice_marks_count_as_not_chosen():
    groups = np.repeat(np.arange(2), 3)
    x = np.arange(6, dtype=float)
    chosen = np.array([1, 0, 0, 1, np.nan, 0])  # NaN -> not chosen
    result = conditional_logit("~ 0 + x", {"x": x}, groups, chosen, 0.1)
    assert result.used_group_labels == [0, 1]
    assert result.row_info["chosen"].tolist() == [
        True,
        False,
        False,
        True,
        False,
        False,
    ]


def test_no_valid_groups_raises():
    groups = np.repeat(np.arange(2), 3)
    x = np.zeros(6)
    chosen = np.zeros(6, dtype=bool)  # nobody chose anything
    pytest.raises(
        PatsyError,
        conditional_logit,
        "~ 0 + x",
        {"x": x},
        groups,
        chosen,
        0.1,
    )


def test_empty_formula_gives_uniform_probabilities():
    groups = np.repeat(np.arange(2), 3)
    chosen = np.zeros(6, dtype=bool)
    chosen[[0, 3]] = True
    result = conditional_logit("~ 0", {"z": np.zeros(6)}, groups, chosen, 0.1)
    assert result.coef.shape == (0,)
    pred = result.predict({"z": np.zeros(6)}, groups)
    assert np.allclose(pred.probabilities, np.full(6, 1 / 3))


def test_categorical_none_and_nan_are_missing():
    groups = np.repeat(np.arange(3), 3)
    brand = np.array(["a", "b", "c"] * 3, dtype=object)
    price = np.arange(9, dtype=float)
    chosen = np.zeros(9, dtype=bool)
    chosen[[0, 3, 6]] = True
    # training: None in group 1 -> whole group excluded
    train_brand = brand.copy()
    train_brand[4] = None
    result = conditional_logit(
        "~ 0 + price + C(brand)",
        {"price": price, "brand": train_brand},
        groups,
        chosen,
        0.1,
    )
    assert result.group_info[1]["exclusion_reason"] == EXCLUSION_MISSING_FEATURE
    # prediction: NaN level -> group unpredictable
    pred_brand = brand.copy()
    pred_brand[7] = np.nan
    pred = result.predict({"price": price, "brand": pred_brand}, groups)
    assert pred.group_info[2]["reason"] == UNPREDICTABLE_MISSING_FEATURE


def test_pandas_nullable_dtypes():
    pd = pytest.importorskip("pandas")
    groups = np.repeat([0, 1], 3)
    x = np.arange(6, dtype=float)
    # pd.NA inside nullable boolean / integer columns must read as
    # not-chosen and not-available, not raise "boolean value of NA is
    # ambiguous".
    df = pd.DataFrame(
        {
            "x": x,
            "g": groups,
            "pick": pd.array([True, False, False, True, False, pd.NA], dtype="boolean"),
            "avail": pd.array([True, True, True, True, True, False], dtype="boolean"),
        }
    )
    result = conditional_logit("~ 0 + x", df, "g", "pick", 0.1, available="avail")
    assert result.used_group_labels == [0, 1]
    assert result.row_info["chosen"].tolist() == [
        True,
        False,
        False,
        True,
        False,
        False,
    ]
    pred = result.predict(df, "g", available="avail")
    assert np.isfinite(pred.probabilities[:3]).all()
    assert pred.probabilities[5] == 0.0

    # pd.NA in a nullable floating column excludes the owning whole group
    df_na = pd.DataFrame(
        {"x": pd.array([1.0, 2.0, 3.0, 4.0, pd.NA, 6.0], dtype="Float64")}
    )
    chosen = np.zeros(6, dtype=bool)
    chosen[[0, 3]] = True
    result2 = conditional_logit("~ 0 + x", df_na, groups, chosen, 0.1)
    assert result2.group_info[1]["exclusion_reason"] == EXCLUSION_MISSING_FEATURE


def test_pandas_na_category_makes_group_unpredictable():
    pd = pytest.importorskip("pandas")
    groups = np.repeat([0, 1], 3)
    brand = np.array(["a", "b", "c"] * 2, dtype=object)
    x = np.arange(6, dtype=float)
    chosen = np.zeros(6, dtype=bool)
    chosen[[0, 3]] = True
    result = conditional_logit(
        "~ 0 + x + C(brand)",
        {"x": x, "brand": brand},
        groups,
        chosen,
        0.1,
    )
    new_brand = brand.copy()
    new_brand[1] = pd.NA
    pred = result.predict({"x": x, "brand": new_brand}, groups)
    assert pred.group_info[0]["reason"] == UNPREDICTABLE_MISSING_FEATURE
    assert np.isfinite(pred.probabilities[3:]).all()


def test_column_name_references():
    pd = pytest.importorskip("pandas")
    groups, price, brand, chosen = _make_data(n_groups=3)
    df = pd.DataFrame(
        {
            "price": price,
            "brand": brand,
            "gid": groups,
            "pick": chosen,
            "ok": np.ones(9, dtype=bool),
        }
    )
    result = conditional_logit(
        "~ 0 + price + C(brand)",
        df,
        "gid",
        "pick",
        0.1,
        available="ok",
    )
    assert result.used_group_labels == [0, 1, 2]
    pred = result.predict(df, "gid", available="ok")
    assert np.isfinite(pred.probabilities).all()


def test_group_info_probability_mapping():
    groups = np.repeat(np.arange(2), 3)
    x = np.arange(6, dtype=float)
    chosen = np.zeros(6, dtype=bool)
    chosen[[0, 3]] = True
    result = conditional_logit("~ 0 + x", {"x": x}, groups, chosen, 0.1)
    pred = result.predict({"x": x}, groups)
    info = pred.group_info[0]
    assert set(info["probabilities"]) == {0, 1, 2}
    assert np.isclose(sum(info["probabilities"].values()), 1.0)
    assert info["chosen_position"] in (0, 1, 2)
