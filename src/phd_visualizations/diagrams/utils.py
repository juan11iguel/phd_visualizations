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