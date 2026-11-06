"""UMass Lowell plotting style helpers.

Import this package before creating Matplotlib figures:

    import uml_plot_style as uml
    uml.apply()
"""

from .colors import (
    ACCENT_COLORS,
    AXIS_COLORS,
    COLOR_SEQUENCE,
    PALETTE,
    PRIMARY_COLORS,
    SECONDARY_COLORS,
    axis_colors,
    color_cycle,
    get_color,
)
from .style import apply, brand_cmap, gradient, register_fonts, style_axes, use

__all__ = [
    "ACCENT_COLORS",
    "AXIS_COLORS",
    "COLOR_SEQUENCE",
    "PALETTE",
    "PRIMARY_COLORS",
    "SECONDARY_COLORS",
    "apply",
    "axis_colors",
    "brand_cmap",
    "color_cycle",
    "get_color",
    "gradient",
    "register_fonts",
    "style_axes",
    "use",
]
