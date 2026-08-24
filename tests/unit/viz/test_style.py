"""Shared figure furniture: palette, save-to-disk, and the provenance footer
every figure in this lane is required to carry (fleet brief: "n, the test
used, and whether it is pre-specified or exploratory in the figure itself")."""

from __future__ import annotations

from pathlib import Path

import matplotlib

from carelite.types import Condition
from carelite.viz.style import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_MARKERS,
    CONDITION_ORDER,
    OKABE_ITO,
    add_provenance_footer,
    new_figure,
    save_figure,
)


def test_backend_is_non_interactive() -> None:
    # Importing carelite.viz.style must never risk opening a window, in tests
    # or in `make reproduce` on a headless CI runner.
    assert matplotlib.get_backend().lower() == "agg"


def test_every_condition_has_a_colour_label_and_marker() -> None:
    for cond in Condition:
        assert str(cond) in CONDITION_COLORS
        assert str(cond) in CONDITION_LABELS
        assert str(cond) in CONDITION_MARKERS
    assert set(CONDITION_ORDER) == {str(c) for c in Condition}


def test_condition_d_is_visually_distinct_from_the_qualitative_palette() -> None:
    # Rule 4: D is a floor marker, not a competing condition. It must not share
    # a hue with any of the five real conditions, and its marker must differ
    # from at least one of them so a reader can't mistake it for "just another
    # condition, slightly faded" even in greyscale.
    other_colors = {CONDITION_COLORS[c] for c in CONDITION_ORDER if c != "D"}
    assert CONDITION_COLORS["D"] not in other_colors
    assert CONDITION_MARKERS["D"] != CONDITION_MARKERS["A"]


def test_palette_is_the_okabe_ito_set() -> None:
    # Every condition colour must come from the colourblind-safe palette, not
    # an ad hoc hex value introduced later.
    palette_values = set(OKABE_ITO.values())
    for color in CONDITION_COLORS.values():
        assert color in palette_values


def test_save_figure_writes_png_and_pdf(tmp_path: Path) -> None:
    fig = new_figure((4, 3))
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    paths = save_figure(fig, "smoke_test_figure", tmp_path)
    assert len(paths) == 2
    png, pdf = paths
    assert png.suffix == ".png"
    assert pdf.suffix == ".pdf"
    assert png.exists() and png.stat().st_size > 0
    assert pdf.exists() and pdf.stat().st_size > 0


def test_save_figure_creates_missing_output_dir(tmp_path: Path) -> None:
    fig = new_figure((3, 3))
    fig.add_subplot(111)
    nested = tmp_path / "does" / "not" / "exist"
    paths = save_figure(fig, "x", nested)
    assert nested.exists()
    assert all(p.exists() for p in paths)


def test_provenance_footer_confirmatory_shows_prespecified_badge() -> None:
    fig = new_figure((5, 4))
    fig.add_subplot(111)
    add_provenance_footer(fig, n=60, test="Wilcoxon signed-rank", prespec=True)
    texts = [t.get_text() for t in fig.texts]
    joined = "\n".join(texts)
    assert "n = 60" in joined
    assert "Wilcoxon signed-rank" in joined
    assert "PRE-SPECIFIED" in joined


def test_provenance_footer_exploratory_shows_exploratory_badge() -> None:
    fig = new_figure((5, 4))
    fig.add_subplot(111)
    add_provenance_footer(fig, n=12, test="descriptive", prespec=False)
    joined = "\n".join(t.get_text() for t in fig.texts)
    assert "EXPLORATORY" in joined
    assert "PRE-SPECIFIED" not in joined


def test_provenance_footer_mixed_status_does_not_claim_either_badge() -> None:
    # Figures 1, 2, 7, 8 mix confirmatory and exploratory content panel-by-panel
    # or point-by-point; the footer must not falsely claim a single status.
    fig = new_figure((5, 4))
    fig.add_subplot(111)
    add_provenance_footer(fig, n="12-60", test="mixed", prespec=None)
    joined = "\n".join(t.get_text() for t in fig.texts)
    assert "PRE-SPECIFIED" not in joined
    assert joined.count("EXPLORATORY") == 0
    assert "per element" in joined
