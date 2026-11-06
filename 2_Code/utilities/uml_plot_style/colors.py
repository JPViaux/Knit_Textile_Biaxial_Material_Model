"""UMass Lowell brand colors for plotting.

Source: UMass Lowell brand guide color page.
"""

from __future__ import annotations

from itertools import cycle, islice

PRIMARY_COLORS = {
    "uml_blue": "#1257D1",
    "black": "#000000",
}

SECONDARY_COLORS = {
    "dark_blue": "#00396E",
    "light_blue": "#5ADBFF",
    "gray": "#878A8F",
    "blue": "#00B5F1",
}

ACCENT_COLORS = {
    "green": "#3BD5AE",
    "aqua": "#62DAFC",
    "yellow": "#FFD140",
    "orange": "#FB471F",
    "fern": "#027669",
    "river": "#4295A9",
    "gold": "#D0AF22",
    "maroon": "#9E3124",
}

PALETTE = {
    **PRIMARY_COLORS,
    **SECONDARY_COLORS,
    **ACCENT_COLORS,
    "white": "#FFFFFF",
    "grid": "#D7D9DE",
    "background": "#FFFFFF",
}

# High contrast order for repeated data series.
COLOR_SEQUENCE = [
    PALETTE["uml_blue"],
    PALETTE["gray"],
    PALETTE["dark_blue"],
    PALETTE["light_blue"],
    PALETTE["river"],
    PALETTE["fern"],
    PALETTE["blue"],
    PALETTE["green"],
    PALETTE["gold"],
    PALETTE["maroon"],
]

# Useful for strain/stress components and course/wale/z style plots.
AXIS_COLORS = [
    PALETTE["uml_blue"],
    PALETTE["gray"],
    PALETTE["dark_blue"],
    PALETTE["light_blue"],
    PALETTE["river"],
    PALETTE["fern"],
]


def get_color(name: str) -> str:
    """Return a brand color by name."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return PALETTE[key]
    except KeyError as exc:
        valid = ", ".join(sorted(PALETTE))
        raise KeyError(f"Unknown UML color '{name}'. Valid names: {valid}") from exc


def color_cycle(n: int | None = None) -> list[str]:
    """Return the default brand color sequence, repeated to length n if needed."""
    if n is None:
        return COLOR_SEQUENCE.copy()
    return list(islice(cycle(COLOR_SEQUENCE), n))


def axis_colors(n: int | None = None) -> list[str]:
    """Return the axis/component color sequence, repeated to length n if needed."""
    if n is None:
        return AXIS_COLORS.copy()
    return list(islice(cycle(AXIS_COLORS), n))
