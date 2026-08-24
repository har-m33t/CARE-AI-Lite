"""The reverse-coding boundary, and the aggregation the pre-registration fixes.

`ritualistic` is scored so a raw 5 is the worst response. If any aggregation in
this package reads `raw` instead of `quality` for that dimension, one dimension
of eleven is inverted and every number downstream still looks plausible. These
tests are the tripwire for that, at each aggregation step separately.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from carelite.eval.rubric.dimensions import REVERSE_CODED, to_quality
from carelite.stats.measures import (
    FOUR_HABITS_COMPOSITE,
    MEASURES,
    NURSE_COMPOSITE,
    Measure,
    attach_quality,
    cell_means,
    measure,
    measure_by_generation,
    paired_matrix,
    quality_lookup,
)
from carelite.types import RUBRIC_DIMENSIONS, Condition
from tests.unit.stats.conftest import constant_scores, make_long

# ---------------------------------------------------------------------------
# The lookup table is the rubric lane's function, not a copy of its constant
# ---------------------------------------------------------------------------


def test_quality_lookup_agrees_with_to_quality_on_every_cell() -> None:
    table = quality_lookup()
    assert len(table) == len(RUBRIC_DIMENSIONS) * 5
    for (dimension, raw), value in table.items():
        assert value == to_quality(dimension, raw)


def test_only_ritualistic_is_flipped() -> None:
    """Guards against a future dimension being reverse-coded without this noticing."""
    assert frozenset({"ritualistic"}) == REVERSE_CODED
    table = quality_lookup()
    for dimension in RUBRIC_DIMENSIONS:
        for raw in range(1, 6):
            expected = 6 - raw if dimension == "ritualistic" else raw
            assert table[(dimension, raw)] == expected


# ---------------------------------------------------------------------------
# attach_quality
# ---------------------------------------------------------------------------


def test_attach_quality_inverts_ritualistic_and_passes_the_rest_through() -> None:
    long = make_long(scores={("SC-000", "A", 0): {"ritualistic": 5, "naturalness": 5, "name": 2}})
    scored = attach_quality(long).set_index("dimension")["quality"]
    assert scored["ritualistic"] == 1.0  # raw 5 is the WORST ritual score
    assert scored["naturalness"] == 5.0
    assert scored["name"] == 2.0


def test_attach_quality_is_idempotent() -> None:
    long = make_long(scores={("SC-000", "A", 0): {"ritualistic": 4}})
    once = attach_quality(long)
    twice = attach_quality(once)
    assert twice["quality"].tolist() == once["quality"].tolist() == [2.0]


def test_attach_quality_overwrites_a_wrong_quality_column() -> None:
    """A caller cannot poison the transform by pre-setting the column."""
    long = make_long(scores={("SC-000", "A", 0): {"ritualistic": 5}})
    long["quality"] = 99.0
    assert attach_quality(long)["quality"].tolist() == [1.0]


def test_attach_quality_keeps_missing_scores_missing() -> None:
    long = make_long(scores={("SC-000", "A", 0): {"name": 3}})
    long.loc[0, "raw"] = np.nan
    assert math.isnan(attach_quality(long)["quality"].iloc[0])


def test_attach_quality_rejects_an_out_of_range_score() -> None:
    long = make_long(scores={("SC-000", "A", 0): {"name": 3}})
    long.loc[0, "raw"] = 7
    with pytest.raises(ValueError, match="outside the 1-5 rubric scale"):
        attach_quality(long)


def test_attach_quality_needs_the_columns_it_reads() -> None:
    with pytest.raises(KeyError, match="missing"):
        attach_quality(pd.DataFrame({"generation_id": ["g"]}))


# ---------------------------------------------------------------------------
# Composites
# ---------------------------------------------------------------------------


def test_the_nurse_composite_is_the_five_dimensions_section_3_names() -> None:
    assert NURSE_COMPOSITE.dimensions == ("name", "understand", "respect", "support", "explore")
    assert FOUR_HABITS_COMPOSITE.dimensions == ("ib", "epp", "de", "ie")


def test_every_dimension_is_available_as_a_measure_of_its_own() -> None:
    for dimension in RUBRIC_DIMENSIONS:
        assert MEASURES[dimension].dimensions == (dimension,)


def test_a_measure_cannot_name_a_dimension_that_does_not_exist() -> None:
    with pytest.raises(ValueError, match="not rubric dimensions"):
        Measure(key="bogus", label="bogus", dimensions=("empathy_vibes",))


def test_measure_lookup_reports_a_typo_usefully() -> None:
    with pytest.raises(KeyError, match="not a measure"):
        measure("nurse")


def test_a_composite_is_the_mean_of_its_quality_scores() -> None:
    long = make_long(
        scores={
            ("SC-000", "A", 0): {
                "name": 1,
                "understand": 2,
                "respect": 3,
                "support": 4,
                "explore": 5,
            }
        }
    )
    row = measure_by_generation(long, NURSE_COMPOSITE).iloc[0]
    assert row["value"] == pytest.approx(3.0)
    assert row["n_dimensions"] == 5


def test_a_missing_dimension_is_dropped_from_the_composite_not_the_generation() -> None:
    """Pre-registration §10: missing for that dimension only, never imputed."""
    long = make_long(
        scores={("SC-000", "A", 0): {"name": 4, "understand": 4, "respect": 4, "support": 4}}
    )
    row = measure_by_generation(long, NURSE_COMPOSITE).iloc[0]
    assert row["value"] == pytest.approx(4.0)
    assert row["n_dimensions"] == 4  # visible, not silently equal to five


def test_a_ritualistic_measure_is_aggregated_on_the_quality_scale() -> None:
    """The single most consequential assertion in this file.

    Two generations, raw ritualistic 5 (worst) and 1 (best). Their mean on the
    quality scale is 3.0 either way, so the giveaway is the individual values:
    the worst raw score must come out as the lowest quality value.
    """
    long = make_long(
        scores={
            ("SC-000", "A", 0): {"ritualistic": 5},
            ("SC-000", "B", 0): {"ritualistic": 1},
        }
    )
    values = measure_by_generation(long, measure("ritualistic")).set_index("condition")["value"]
    assert values["A"] == pytest.approx(1.0)
    assert values["B"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Cell means: the §3 aggregation
# ---------------------------------------------------------------------------


def test_cell_means_average_the_three_samples_in_a_cell() -> None:
    long = make_long(
        scores={
            ("SC-000", "A", 0): {"name": 1},
            ("SC-000", "A", 1): {"name": 2},
            ("SC-000", "A", 2): {"name": 3},
        }
    )
    cells = cell_means(long, measure("name"))
    assert len(cells) == 1
    assert cells.iloc[0]["value"] == pytest.approx(2.0)
    assert cells.iloc[0]["n_samples"] == 3


def test_cell_means_keep_rater_types_apart() -> None:
    """§3 aggregates 'per rater type'; pooling a judge and a human would be a bug."""
    judge = make_long(scores={("SC-000", "A", 0): {"name": 5}}, rater_type="llm_judge")
    human = make_long(scores={("SC-000", "A", 0): {"name": 1}}, rater_type="human", rater_id="R1")
    cells = cell_means(pd.concat([judge, human]), measure("name"))
    assert len(cells) == 2
    assert set(cells["value"]) == {1.0, 5.0}


def test_cell_means_of_ritualistic_are_on_the_quality_scale() -> None:
    long = make_long(
        scores={("SC-000", "A", s): {"ritualistic": 5} for s in range(3)},
    )
    assert cell_means(long, measure("ritualistic")).iloc[0]["value"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def test_paired_matrix_drops_scenarios_missing_a_condition() -> None:
    scores = {
        ("SC-000", "A", 0): {"name": 3},
        ("SC-000", "B", 0): {"name": 4},
        ("SC-001", "A", 0): {"name": 3},  # no B cell
    }
    matrix = paired_matrix(cell_means(make_long(scores=scores), measure("name")), ["A", "B"])
    assert list(matrix.index) == ["SC-000"]
    assert list(matrix.columns) == ["A", "B"]


def test_paired_matrix_returns_columns_in_the_requested_order() -> None:
    scores = {
        ("SC-000", "A", 0): {"name": 3},
        ("SC-000", "B", 0): {"name": 4},
    }
    cells = cell_means(make_long(scores=scores), measure("name"))
    assert list(paired_matrix(cells, [Condition.B, Condition.A]).columns) == ["B", "A"]


def test_paired_matrix_is_empty_when_a_condition_is_wholly_absent() -> None:
    long = make_long(scores={("SC-000", "A", 0): {"name": 3}})
    assert paired_matrix(cell_means(long, measure("name")), ["A", "D"]).empty


def test_paired_matrix_can_restrict_to_one_rater_type() -> None:
    judge = make_long(scores={("SC-000", "A", 0): {"name": 5}, ("SC-000", "B", 0): {"name": 5}})
    human = make_long(
        scores={("SC-000", "A", 0): {"name": 1}, ("SC-000", "B", 0): {"name": 1}},
        rater_type="human",
        rater_id="R1",
    )
    cells = cell_means(pd.concat([judge, human]), measure("name"))
    matrix = paired_matrix(cells, ["A", "B"], rater_type="human")
    assert matrix.loc["SC-000", "A"] == pytest.approx(1.0)


def test_empty_input_produces_an_empty_frame_not_an_error() -> None:
    empty = make_long(scores={})
    assert measure_by_generation(empty, NURSE_COMPOSITE).empty
    assert cell_means(empty, NURSE_COMPOSITE).empty


def test_full_pipeline_preserves_polarity_end_to_end(
    nurse_dimensions: tuple[str, ...],
) -> None:
    """A degraded condition scoring worst on raw ritualistic must rank lowest.

    Condition D is written as maximally ritualistic (raw 5) and Condition B as
    minimally so (raw 1). Through the whole path — attach_quality, per-generation
    measure, cell mean, paired matrix — B must come out above D. If any step read
    `raw`, this assertion reverses.
    """
    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(5):
        scenario = f"SC-{i:03d}"
        for sample in range(3):
            scores[(scenario, "B", sample)] = {
                **constant_scores(nurse_dimensions, 4),
                "ritualistic": 1,
            }
            scores[(scenario, "D", sample)] = {
                **constant_scores(nurse_dimensions, 2),
                "ritualistic": 5,
            }
    matrix = paired_matrix(cell_means(make_long(scores=scores), measure("ritualistic")), ["B", "D"])
    assert (matrix["B"] > matrix["D"]).all()
    assert matrix["B"].iloc[0] == pytest.approx(5.0)
    assert matrix["D"].iloc[0] == pytest.approx(1.0)
