"""Matplotlib style helpers for UMass Lowell branded figures."""

from __future__ import annotations

import os
from pathlib import Path

from .colors import COLOR_SEQUENCE, PALETTE, axis_colors, get_color

WEB_FONT_FAMILIES = {
    "headline": ["Anek Latin", "Barlow", "Arial", "DejaVu Sans"],
    "body": ["Barlow", "Anek Latin", "Arial", "DejaVu Sans"],
    "mono": ["Space Mono", "Consolas", "DejaVu Sans Mono"],
}


def register_fonts(font_dirs: list[str | Path] | None = None) -> list[Path]:
    """Register local .ttf/.otf fonts with Matplotlib.

    The UMass Lowell brand fonts are open source, but this package does not
    download or bundle them. Put font files in a directory and pass it here, or
    set UML_FONT_DIR to that directory before importing your plotting script.
    """
    import matplotlib.font_manager as fm

    dirs: list[Path] = []
    env_dir = os.environ.get("UML_FONT_DIR")
    if env_dir:
        dirs.append(Path(env_dir))
    if font_dirs:
        dirs.extend(Path(path) for path in font_dirs)

    registered: list[Path] = []
    for font_dir in dirs:
        if not font_dir.exists():
            continue
        for suffix in ("*.ttf", "*.otf"):
            for font_file in font_dir.rglob(suffix):
                fm.fontManager.addfont(str(font_file))
                registered.append(font_file)
    return registered


def _available_fonts() -> set[str]:
    import matplotlib.font_manager as fm

    return {font.name for font in fm.fontManager.ttflist}


def _available_stack(preferred: list[str], installed: set[str]) -> list[str]:
    stack = [font for font in preferred if font in installed]
    return stack or ["DejaVu Sans"]


def _rc_params() -> dict:
    installed = _available_fonts()
    body_fonts = _available_stack(WEB_FONT_FAMILIES["body"], installed)
    mono_fonts = _available_stack(WEB_FONT_FAMILIES["mono"], installed)

    return {
        "font.family": body_fonts,
        "font.sans-serif": body_fonts,
        "font.monospace": mono_fonts,
        "font.size": 11,
        "axes.prop_cycle": _cycler(),
        "axes.facecolor": PALETTE["background"],
        "axes.edgecolor": PALETTE["black"],
        "axes.labelcolor": PALETTE["black"],
        "axes.labelsize": 12,
        "axes.labelweight": "regular",
        "axes.titlecolor": PALETTE["dark_blue"],
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.7,
        "grid.alpha": 0.75,
        "grid.linestyle": "-",
        "figure.facecolor": PALETTE["background"],
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": PALETTE["background"],
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "patch.edgecolor": "none",
        "text.color": PALETTE["black"],
        "xtick.color": PALETTE["black"],
        "ytick.color": PALETTE["black"],
    }


def _cycler():
    from cycler import cycler

    return cycler(color=COLOR_SEQUENCE)


def apply(font_dirs: list[str | Path] | None = None) -> None:
    """Apply the UMass Lowell Matplotlib rcParams."""
    import matplotlib as mpl

    register_fonts(font_dirs)
    mpl.rcParams.update(_rc_params())


def use(font_dirs: list[str | Path] | None = None) -> None:
    """Alias for apply()."""
    apply(font_dirs)


def style_axes(ax=None):
    """Apply finishing touches to an Axes and return it."""
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["black"])
    ax.spines["bottom"].set_color(PALETTE["black"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.7, alpha=0.75)
    return ax


def gradient(color: str = "uml_blue", n: int = 6, start: float = 0.18) -> list[str]:
    """Return n tints from white toward a brand color."""
    import matplotlib.colors as mcolors

    rgb = mcolors.to_rgb(get_color(color) if color in PALETTE else color)
    white = (1.0, 1.0, 1.0)
    if n <= 1:
        return [mcolors.to_hex(rgb)]
    values = []
    for i in range(n):
        t = start + (1.0 - start) * (i / (n - 1))
        mixed = tuple((1.0 - t) * white[j] + t * rgb[j] for j in range(3))
        values.append(mcolors.to_hex(mixed))
    return values


def brand_cmap(name: str = "uml_blue", n: int = 256):
    """Return a Matplotlib colormap based on the UML palette."""
    from matplotlib.colors import LinearSegmentedColormap

    if name == "sequential":
        colors = [PALETTE["light_blue"], PALETTE["uml_blue"], PALETTE["dark_blue"]]
    elif name == "diverging":
        colors = [PALETTE["gray"], PALETTE["white"], PALETTE["uml_blue"]]
    elif name == "axis":
        colors = axis_colors()
    else:
        colors = gradient(name, n=6)
    return LinearSegmentedColormap.from_list(f"uml_{name}", colors, N=n)
