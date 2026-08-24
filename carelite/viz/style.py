"""Shared look-and-feel for every figure in `carelite.viz`.

Everything here exists so that no individual figure module has to reinvent a
colourblind-safe palette, a print-legible font stack, or the "n / test /
pre-specified" footer the fleet rules require on every figure. Import this
module (or a module that imports it) before building any figure; it forces the
`Agg` backend so figure code never accidentally tries to open a window,
including under `pytest` on a headless CI runner.

Palette: Okabe & Ito (2008), the standard colourblind-safe qualitative
palette — distinguishable under deuteranopia, protanopia and tritanopia, and
still reads correctly printed in greyscale because the hues were chosen for
distinct luminance too. Condition D (the negative control) and pre-specified
vs. exploratory status are additionally encoded by marker shape, hatch, or
line style, not colour alone, because "distinguishable at a glance" has to
survive a black-and-white printout.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: never touch a display, in tests or in `make reproduce`

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

__all__ = [
    "CONDITION_COLORS",
    "CONDITION_LABELS",
    "CONDITION_MARKERS",
    "CONDITION_ORDER",
    "EXPLORATORY_LABEL",
    "OKABE_ITO",
    "PRESPEC_LABEL",
    "add_provenance_footer",
    "apply_style",
    "new_figure",
    "save_figure",
]

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

#: Okabe & Ito 2008, "Color Universal Design". Black is reserved for text/axes.
OKABE_ITO: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#999999",
}

#: Canonical display order for the six experimental conditions (`carelite.types.Condition`).
CONDITION_ORDER: tuple[str, ...] = ("A", "A2", "B", "C", "LC", "D")

CONDITION_LABELS: dict[str, str] = {
    "A": "A — bare model",
    "A2": "A2 — bare model, cross-model",
    "B": "B — framework-prompted",
    "C": "C — framework + retrieval",
    "LC": "LC — long-context",
    "D": "D — negative control",
}

#: D is deliberately NOT a qualitative-palette hue: it is a degraded control, not a
#: competing condition (see the viz lane brief, rule 4), so it is rendered in a
#: neutral grey with a distinct marker rather than earning a place among the six
#: "real" comparison colours a reader's eye would otherwise rank it against.
CONDITION_COLORS: dict[str, str] = {
    "A": OKABE_ITO["blue"],
    "A2": OKABE_ITO["sky_blue"],
    "B": OKABE_ITO["orange"],
    "C": OKABE_ITO["bluish_green"],
    "LC": OKABE_ITO["reddish_purple"],
    "D": OKABE_ITO["grey"],
}

#: Marker shape is a second, colour-independent channel — required for D (a plain
#: grey dot would read as "just another condition, slightly faded") and useful
#: everywhere else so the figure still works reproduced in greyscale.
CONDITION_MARKERS: dict[str, str] = {
    "A": "o",
    "A2": "s",
    "B": "o",
    "C": "o",
    "LC": "D",
    "D": "x",
}

#: Pre-specified/exploratory status is a second channel too: filled vs. hollow
#: markers, and solid vs. dashed reference lines, not colour alone.
PRESPEC_LABEL = "PRE-SPECIFIED"
EXPLORATORY_LABEL = "EXPLORATORY"

_FOOTER_FACE = "#F0F0F0"
_FOOTER_EDGE_PRESPEC = OKABE_ITO["black"]
_FOOTER_EDGE_EXPLORATORY = OKABE_ITO["vermillion"]

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

_STYLE_APPLIED = False


def apply_style() -> None:
    """Set print-legible, non-interactive rcParams. Idempotent."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.titlesize": 12,
            "axes.edgecolor": OKABE_ITO["black"],
            "axes.labelcolor": OKABE_ITO["black"],
            "text.color": OKABE_ITO["black"],
            "xtick.color": OKABE_ITO["black"],
            "ytick.color": OKABE_ITO["black"],
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "savefig.dpi": 300,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    _STYLE_APPLIED = True


apply_style()

# ---------------------------------------------------------------------------
# Figure lifecycle
# ---------------------------------------------------------------------------


def new_figure(figsize: tuple[float, float]) -> Figure:
    """A `Figure` with an Agg canvas already attached — never touches pyplot's
    global figure registry, so tests can create many figures without leaking
    state or triggering "too many open figures" warnings."""
    apply_style()
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig


def save_figure(fig: Figure, name: str, output_dir: Path) -> list[Path]:
    """Save `fig` as both PNG and PDF under `output_dir/name.{png,pdf}`.

    Returns the two paths written, in that order. `output_dir` is created if
    it does not exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    return [png_path, pdf_path]


def add_provenance_footer(
    fig: Figure,
    *,
    n: int | str,
    test: str,
    prespec: bool | None,
    extra: str | None = None,
) -> None:
    """Stamp n, the test used, and pre-specified/exploratory status onto `fig`.

    Every figure this lane produces carries this footer directly on the
    canvas — never only in a caption a reader might separate from the image.

    `prespec=None` means the figure mixes pre-specified and exploratory
    content panel-by-panel or point-by-point (figures 1, 2, 7, 8 all do this);
    in that case the footer states the mixed-status convention instead of a
    single badge, and the per-element encoding (marker fill, hatch) carries
    the actual distinction — see each figure's own legend.
    """
    if prespec is None:
        status_text = "pre-specified vs. exploratory encoded per element — see legend"
        edge = OKABE_ITO["black"]
    else:
        status_text = PRESPEC_LABEL if prespec else EXPLORATORY_LABEL
        edge = _FOOTER_EDGE_PRESPEC if prespec else _FOOTER_EDGE_EXPLORATORY

    line = f"n = {n}    |    test: {test}    |    {status_text}"
    if extra:
        line = f"{line}\n{extra}"

    linestyle = "solid" if prespec else ("dashed" if prespec is False else "dotted")
    fig.text(
        0.01,
        0.01,
        line,
        fontsize=6.8,
        family="monospace",
        color=OKABE_ITO["black"],
        va="bottom",
        ha="left",
        linespacing=1.4,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": _FOOTER_FACE,
            "edgecolor": edge,
            "linewidth": 1.1,
            "linestyle": linestyle,
        },
    )
