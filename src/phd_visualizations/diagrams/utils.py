import math


# Diagram generation auxiliary functions
def round_to_nonzero_decimal(n):
    if n == 0:
        return 0
    sgn = -1 if n < 0 else 1
    scale = int(-math.floor(math.log10(abs(n))))
    if scale <= 0:
        scale = 1
    factor = 10 ** scale
    return sgn * math.floor(abs(n) * factor) / factor


def convert_to_float_if_possible(value):
    try:
        converted_value = float(value)
        return converted_value
    except ValueError:
        return value
    
    
def get_y(x, xmin, xmax, ymin, ymax):
    return ((ymax - ymin) / (xmax - xmin)) * (x - xmin) + ymin


def generate_boundary_circle(id, size_icon, size_boundary, max_value, pos_x, pos_y) -> str:
    x = pos_x + size_icon / 2
    y = pos_y + size_icon / 2

    return f"""
    <g id="boundary-{id}">
        <ellipse cx="{x}" cy="{y}" rx="{size_boundary / 2}" ry="{size_boundary / 2}" fill-opacity="0" fill="rgb(255, 255, 255)" stroke="#ececec" stroke-dasharray="3 3" pointer-events="all"/>
        <g fill="#ECECEC" font-family="Helvetica" font-size="10px">
        <text x="{x + size_boundary / 2}" y="{y}">{max_value:.0f}</text></g></g>
    """