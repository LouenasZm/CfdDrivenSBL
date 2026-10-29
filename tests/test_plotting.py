"""Smoke tests of the figures: they must build in both themes without touching a display."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from sblcfd import NSBL, questionable  # noqa: E402
from sblcfd.plotting import (  # noqa: E402
    plot_marginals,
    plot_predictions,
    plot_relevance,
    save_figure,
    style,
)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    samples = np.c_[
        rng.normal(0.5, 0.05, 1500),  # clearly needed
        rng.normal(0.0, 0.3, 1500),  # consistent with zero
        np.abs(rng.normal(0.0, 0.2, 1500)),  # sign-constrained, piled up against zero
    ]
    parameters = questionable(["kept", "pruned", "RE[exp:eta1^2*eta3^2]"], [0.0, -1.0, 0.0], 1.0)
    model = NSBL(lambda x: np.zeros(x.shape[0]), parameters)
    model.set_samples(samples)
    model.fit_mixture("kde")
    return model, model.optimize(n_starts=4, seed=0)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_figures_build_and_save(fitted, mode, tmp_path):
    import matplotlib.pyplot as plt

    model, result = fitted
    np.testing.assert_array_equal(result.relevant, [True, False, False])

    figure = plot_relevance(result, mode=mode)
    labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
    assert labels == result.names
    save_figure(figure, tmp_path / "relevance.png")

    figure = plot_marginals(result, model.mixture_, truths=[0.5, 0.0, 0.0], mode=mode)
    assert sum(ax.get_visible() for ax in figure.axes) == 3
    save_figure(figure, tmp_path / "marginals.png")

    x = np.linspace(0.0, 1.0, 30)
    draws = np.sin(3 * x)[None, :] + 0.05 * np.random.default_rng(1).standard_normal((40, 30))
    panels = {
        "training": {
            "x": x,
            "draws": draws,
            "reference": np.sin(3 * x),
            "baseline": 0.8 * np.sin(3 * x),
        },
        "unseen": {"x": x, "draws": draws},
    }
    figure = plot_predictions(panels, xlabel="x", ylabel="y", mode=mode, ncols=2)
    assert len(figure.legends[0].get_texts()) == 3
    save_figure(figure, tmp_path / "predictions.png")

    assert all(
        (tmp_path / name).stat().st_size > 0
        for name in ("relevance.png", "marginals.png", "predictions.png")
    )
    plt.close("all")


def test_style_rejects_unknown_mode():
    with pytest.raises(ValueError):
        with style("sepia"):
            pass
