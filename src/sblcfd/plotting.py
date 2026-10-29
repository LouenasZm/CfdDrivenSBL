"""Figures for the three questions an NSBL run answers.

* Which terms are kept?  -> :func:`plot_relevance`
* What did the ARD priors do to each coefficient?  -> :func:`plot_marginals`
* How does the sparse model predict, and how sure is it?  -> :func:`plot_predictions`

All functions need matplotlib (``pip install 'sbl-cfd-driven[plot]'``) and come in
a light and a dark rendition.  Colours follow a fixed role assignment: the model
under study is blue, the model it is compared with is orange, reference data and
pruned terms are neutral.  The three hues are separable under the common forms of
colour-vision deficiency, and every figure carries a legend so that identity never
rests on colour alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager

import numpy as np

from .mixture import GaussianMixture
from .nsbl import NSBLResult

__all__ = ["plot_marginals", "plot_predictions", "plot_relevance", "save_figure", "style"]

_TOKENS = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ("#2a78d6", "#eb6834", "#1baf7a"),
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ("#3987e5", "#d95926", "#199e70"),
    },
}


@contextmanager
def style(mode: str = "light"):
    """Context manager applying the figure style; yields the colour tokens."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if mode not in _TOKENS:
        raise ValueError("mode must be 'light' or 'dark'")
    t = _TOKENS[mode]
    # A generic family is resolved when the figure is drawn, i.e. possibly outside this
    # context and with another font than the one the layout was computed for.  Naming a
    # concrete font pins it to the text objects.
    installed = {font.name for font in font_manager.fontManager.ttflist}
    family = next(
        (f for f in ("Helvetica Neue", "Helvetica", "Arial") if f in installed), "DejaVu Sans"
    )
    rc = {
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "font.family": family,
        "font.size": 9,
        "text.color": t["ink"],
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": t["ink"],
        "axes.labelcolor": t["ink_secondary"],
        "axes.labelsize": 9,
        "axes.edgecolor": t["axis"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": t["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "xtick.labelcolor": t["muted"],
        "ytick.labelcolor": t["muted"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "lines.linewidth": 1.5,  # 2 px at 96 dpi
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": t["ink_secondary"],
    }
    with plt.rc_context(rc):
        yield t


def save_figure(fig, path, dpi: int = 200) -> None:
    """Save a figure produced by this module with its own background, tightly cropped."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())


def _rounded_bar(ax, y: float, value: float, height: float, radius_pt: float, **kwargs):
    """Horizontal bar growing from zero: square at the baseline, rounded at the data end."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    fig = ax.figure
    box = ax.get_position()
    width_in = box.width * fig.get_figwidth()
    height_in = box.height * fig.get_figheight()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    rx = radius_pt / 72.0 * (x1 - x0) / width_in
    ry = radius_pt / 72.0 * abs(y1 - y0) / height_in
    rx, ry = min(rx, value), min(ry, height / 2)
    lo, hi = y - height / 2, y + height / 2
    k = 0.5523  # cubic Bezier approximation of a quarter circle
    vertices = [
        (0.0, lo),
        (value - rx, lo),
        (value - rx + k * rx, lo), (value, lo + ry - k * ry), (value, lo + ry),
        (value, hi - ry),
        (value, hi - ry + k * ry), (value - rx + k * rx, hi), (value - rx, hi),
        (0.0, hi),
        (0.0, lo),
    ]  # fmt: skip
    codes = [Path.MOVETO, Path.LINETO] + [Path.CURVE4] * 3 + [Path.LINETO] + [Path.CURVE4] * 3
    codes += [Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(vertices, codes), lw=0, **kwargs))


def plot_relevance(result: NSBLResult, *, mode: str = "light", title: str | None = None):
    """Relevance indicator of every questionable term, kept terms highlighted.

    One bar per term.  Kept terms are drawn in the accent colour and labelled with
    their evidence gain (the log-evidence lost if that term alone is pruned); pruned
    terms have a relevance of zero and show as empty rows.

    Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    names = [result.names[i] for i in result.ard_index]
    n = len(names)
    with style(mode) as t:
        fig, ax = plt.subplots(figsize=(7.2, 1.3 + 0.24 * n))
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(n - 0.5, -1.3)  # the band above the first row holds the threshold label
        ax.set_yticks(range(n), names)
        ax.grid(axis="y", visible=False)
        ax.axvline(result.relevance_tol, color=t["axis"], lw=0.8, zorder=1)
        ax.text(
            result.relevance_tol + 0.012, -0.9, f"threshold {result.relevance_tol:g}",
            ha="left", va="center", fontsize=7.5, color=t["muted"],
        )  # fmt: skip
        for label, kept in zip(ax.get_yticklabels(), result.relevant):
            label.set_color(t["ink"] if kept else t["muted"])
        ax.set_xlabel(
            "relevance indicator  (share of the posterior precision that comes from the data)"
        )
        n_kept = int(result.relevant.sum())
        ax.set_title(title or f"{n_kept} of {n} candidate terms kept", pad=22)
        ax.annotate(
            "kept terms in blue; gain = log-evidence lost if that term alone is pruned",
            xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
            ha="left", va="bottom", fontsize=8, color=t["ink_secondary"],
        )  # fmt: skip
        fig.tight_layout()
        fig.canvas.draw()  # the corner radius is expressed in points: the layout must be final
        for row, (gamma, kept, gain) in enumerate(
            zip(result.relevance, result.relevant, result.evidence_gain)
        ):
            if gamma <= 1e-3:
                continue
            colour = t["series"][0] if kept else t["axis"]
            _rounded_bar(ax, row, float(gamma), 0.62, 3.0, color=colour, zorder=2)
            if kept:
                inside = gamma > 0.82
                ax.text(
                    gamma - 0.012 if inside else gamma + 0.012, row, f"gain {gain:.1f}",
                    ha="right" if inside else "left", va="center", fontsize=7.5,
                    color="#ffffff" if inside else t["ink_secondary"], zorder=3,
                )  # fmt: skip
    return fig


def plot_marginals(
    result: NSBLResult,
    before: GaussianMixture,
    *,
    index: Sequence[int] | None = None,
    truths: Sequence[float] | None = None,
    mode: str = "light",
    ncols: int = 3,
    n_grid: int = 400,
):
    """Marginal densities before and after the ARD priors are applied.

    ``before`` is the mixture of ``likelihood x known prior`` (flat priors on the
    questionable parameters); "after" is the NSBL posterior.  A pruned parameter
    collapses onto zero, a kept one keeps its shape.

    Parameters
    ----------
    result : NSBLResult
    before : GaussianMixture
        Typically ``model.mixture_``.
    index : sequence of int, optional
        Parameters to show; defaults to all of them.
    truths : sequence of float, optional
        Known true values (synthetic studies), one per parameter of the model.
    mode : {"light", "dark"}
    ncols : int
    n_grid : int

    Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    lower, upper, fold = result._bounds
    index = list(range(len(result.parameters))) if index is None else list(index)
    before = before.normalized()
    centre, spread = before.mean(), before.std()
    nrows = int(np.ceil(len(index) / ncols))
    with style(mode) as t:
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(2.5 * ncols, 1.9 * nrows + 0.5), squeeze=False
        )
        for ax in axes.ravel()[len(index) :]:
            ax.set_visible(False)
        handles = None
        for ax, i in zip(axes.ravel(), index):
            # the region where the likelihood has mass, within the admissible range
            lo = max(lower[i], centre[i] - 4 * spread[i])
            hi = min(upper[i], centre[i] + 4 * spread[i])
            grid = np.linspace(lo, hi, n_grid)
            curves = []
            for mixture in (before, result.posterior):
                pdf = mixture.marginal_pdf(i, grid)
                if i in fold:  # mirror image of the kernels that leaked across zero
                    pdf = pdf + mixture.marginal_pdf(i, -grid)
                curves.append(pdf / max(pdf.max(), 1e-300))  # each scaled to its own mode
            (h0,) = ax.plot(grid, curves[0], color=t["muted"], lw=1.5)
            ax.fill_between(grid, curves[1], color=t["series"][0], alpha=0.12, lw=0)
            (h1,) = ax.plot(grid, curves[1], color=t["series"][0], lw=1.5)
            handles = [h0, h1]
            if truths is not None:
                h2 = ax.axvline(truths[i], color=t["ink"], lw=0.9, ymax=0.94)
                handles.append(h2)
            ax.set_title(result.names[i], fontsize=8.5, pad=4)
            ax.set_yticks([])
            ax.set_ylim(0.0, 1.12)
            ax.set_xlim(lo, hi)
            ax.grid(axis="y", visible=False)
            ax.spines["left"].set_visible(False)
        labels = ["flat prior (likelihood only)", "with the learnt ARD prior"]
        if truths is not None:
            labels.append("true value")
        fig.legend(
            handles, labels, loc="upper center", ncols=len(labels),
            bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.6,
        )  # fmt: skip
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.42 / fig.get_figheight()))
    return fig


def plot_predictions(
    panels: Mapping[str, Mapping[str, np.ndarray]],
    *,
    xlabel: str = "x",
    ylabel: str = "",
    model_label: str = "sparse model: posterior mean and 95 % band",
    baseline_label: str = "baseline model",
    reference_label: str = "reference",
    mode: str = "light",
    ncols: int = 3,
    marker_every: int = 3,
):
    """Posterior predictions against reference data, one panel per flow case.

    Parameters
    ----------
    panels : mapping
        ``{title: {"x": abscissa, "draws": (n_draws, n_x) predictions,
        "reference": (n_x,), "baseline": (n_x,)}}``; ``reference`` and ``baseline``
        are optional.  The predictions are those of the solver (or its emulator)
        for posterior draws of the coefficients.
    xlabel, ylabel : str
    model_label, baseline_label, reference_label : str
        Legend entries.
    mode : {"light", "dark"}
    ncols : int
    marker_every : int
        Spacing of the reference markers.

    Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    titles = list(panels)
    nrows = int(np.ceil(len(titles) / ncols))
    with style(mode) as t:
        blue, orange = t["series"][0], t["series"][1]
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.1 * ncols, 2.35 * nrows + 0.55), squeeze=False, sharey=True
        )
        for ax in axes.ravel()[len(titles) :]:
            ax.set_visible(False)
        has_baseline = has_reference = False
        for ax, title in zip(axes.ravel(), titles):
            panel = panels[title]
            x = np.asarray(panel["x"])
            draws = np.atleast_2d(panel["draws"])
            low, high = np.quantile(draws, [0.025, 0.975], axis=0)
            if "baseline" in panel:
                has_baseline = True
                ax.plot(x, panel["baseline"], color=orange, lw=1.5, zorder=2)
            ax.fill_between(x, low, high, color=blue, alpha=0.22, lw=0, zorder=3)
            ax.plot(x, draws.mean(axis=0), color=blue, lw=1.5, zorder=4)
            if "reference" in panel:
                has_reference = True
                ax.plot(
                    x[::marker_every], np.asarray(panel["reference"])[::marker_every],
                    ls="none", marker="o", ms=4.2, mfc=t["ink_secondary"],
                    mec=t["surface"], mew=1.0, zorder=5,
                )  # fmt: skip
            ax.set_title(title, pad=5)
            ax.set_xlabel(xlabel)
        for row in axes:
            row[0].set_ylabel(ylabel)
        handles = [(Line2D([], [], color=blue, lw=1.5), Patch(color=blue, alpha=0.22, lw=0))]
        labels = [model_label]
        if has_baseline:
            handles.append(Line2D([], [], color=orange, lw=1.5))
            labels.append(baseline_label)
        if has_reference:
            handles.append(
                Line2D([], [], ls="none", marker="o", ms=4.2, mfc=t["ink_secondary"],
                       mec=t["surface"], mew=1.0)
            )  # fmt: skip
            labels.append(reference_label)
        from matplotlib.legend_handler import HandlerTuple

        fig.legend(
            handles, labels, loc="upper center", ncols=len(labels), bbox_to_anchor=(0.5, 1.0),
            handlelength=1.8, columnspacing=1.8, handler_map={tuple: HandlerTuple(ndivide=1)},
        )  # fmt: skip
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.42 / fig.get_figheight()))
    return fig
