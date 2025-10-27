# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Color palettes and similar for chuchichaestli visualizations."""

from enum import Enum


class Color(Enum):
    """An assortment of colors and palettes."""

    # Shades
    WHITE = "#DDDEE1"  # rba(221, 222, 225)
    GRAY = "#98989D"  # rba(152, 152, 157)
    GREY = "#98989D"  # rba(152, 152, 157)
    DARKISH = "#666769"  # rba(102, 103, 105)
    DARK = "#3D3E41"  # rba( 61,  62,  65)
    DARKER = "#333437"  # rba( 51,  52,  55)
    DARKEST = "#212225"  # rba( 33,  34,  37)
    BLACK = "#090F0F"  # rba(  9,  15,  15)
    TEXTCOLOR = "#DDDEE1"  # rba(221, 222, 225)
    # Primary colors
    RED = "#FF6767"
    PINK = "#FF375F"  # rba(255,  55,  95)
    ORANGE = "#FF9F0A"  # rba(255, 159,  10)
    YELLOW = "#FFD60A"  # rba(155, 214,  10)
    PURPLE = "#603DD0"  # rba( 96,  61, 208)
    GREEN = "#32D74B"  # rba( 50, 215,  75)
    CYAN = "#5BC1AE"
    BLUE = "#6767FF"
    BROWN = "#D88C4E"  # rba(172, 142, 104)
    # Other
    GOLDEN = "#FEB125"  # rba(256, 177,  37)
    PURPLEBLUE = "#7D7DE1"  # rba(125, 125, 225)
    TURQUOISE = "#00D1A4"  # rba( 10, 210, 165)
    MARGUERITE = "#756BB1"
    # Variants
    CYANLIGHT = "#A0DED2"
    CYANDARK = "#24A38B"


def list_color_names() -> list[str]:
    """List the names of all colors."""
    return [c.name for c in Color]


def get_color(color_name: str) -> str:
    """Retrieve hex-color from default color palette.

    Args:
        color_name: Color name to be fetched.
    """
    sanatized_color_name = color_name.replace("_", "").replace("-", "").strip().upper()
    return Color[sanatized_color_name].value


def color_variant(hex_color: str, shift: int = 10) -> str:
    """Takes a color in hex code and produces a lighter or darker shift variant.

    Args:
        hex_color (str): formatted as '#' + rgb hex string of length 6, or color name.
        shift (int): decimal shift of the rgb hex string

    Returns:
        variant (str): formatted as '#' + rgb hex string of length 6
    """
    if not hex_color.startswith("#"):
        hex_color = get_color(hex_color)
    if len(hex_color) != 7:
        raise ValueError(
            f"Passed {hex_color} to color_variant(), needs to be in hex format."
        )
    rgb_hex = [hex_color[x : x + 2] for x in [1, 3, 5]]
    new_rgb_int = [int(hex_value, 16) + shift for hex_value in rgb_hex]
    # limit to interval 0 and 255
    new_rgb_int = [min([255, max([0, i])]) for i in new_rgb_int]
    # hex() produces "0x88", we want the last two digits
    return "#" + "".join([hex(i)[2:] if i else "00" for i in new_rgb_int])
