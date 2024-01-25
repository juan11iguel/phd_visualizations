import math
from lxml import etree
# import os
# import logging
import base64
# from loguru import logger
# from pathlib import Path
# from typing import Literal

""" Global variables """

nsmap = {
    'sodipodi': 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd',
    'cc': 'http://web.resource.org/cc/',
    'svg': 'http://www.w3.org/2000/svg',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'xlink': 'http://www.w3.org/1999/xlink',
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'inkscape': 'http://www.inkscape.org/namespaces/inkscape'
}

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


def change_text(diagram, object_id, new_text):
    obj = diagram.xpath(f'//svg:g[@id="cell-{object_id}"]', namespaces=nsmap)

    for child in obj[0]:
        if child.tag.endswith('g'):
            for child2 in child:
                if child2.tag.endswith('text'):
                    child2.text = new_text
                    break

    return diagram


def get_y(x, xmin, xmax, ymin, ymax):
    return ((ymax - ymin) / (xmax - xmin)) * (x - xmin) + ymin


def adjust_icon(id, size, tag, value, unit, include_boundary=True, max_size=None, max_value=None):
    if unit == 'degree_celsius': unit = '⁰C'

    for child in tag[0]:
        # Adjust icon size
        if 'image' in child.tag:
            pos_x = child.get("x");
            pos_y = child.get("y")
            current_size = float(child.get("width"))
            delta_size = size - current_size

            child.set("width", str(size))
            child.set("height", str(size))

            pos_x = float(pos_x) - delta_size / 2
            pos_y = float(pos_y) - delta_size / 2

            child.set("x", str(pos_x))
            child.set("y", str(pos_y))

            # Add template-id property to be used later
            child.set("template-id", f'icon-{id}')

        # Add text
        if child.tag.endswith('g'):
            for child2 in child:
                if 'text' in child2.tag:
                    if isinstance(value, str):
                        child2.text = f'{value} {unit}'
                    elif isinstance(value, int):
                        child2.text = f'{value} {unit}'
                    else:
                        child2.text = f'{round_to_nonzero_decimal(value)} {unit}'

    # Add boundary circle
    if include_boundary:
        tag[0][0].addprevious(etree.fromstring(generate_boundary_circle(id, size, max_size, max_value, pos_x, pos_y)))
    return tag, pos_x, pos_y


def generate_boundary_circle(id, size_icon, size_boundary, max_value, pos_x, pos_y):
    x = pos_x + size_icon / 2
    y = pos_y + size_icon / 2

    return f"""
    <g id="boundary-{id}">
        <ellipse cx="{x}" cy="{y}" rx="{size_boundary / 2}" ry="{size_boundary / 2}" fill-opacity="0" fill="rgb(255, 255, 255)" stroke="#ececec" stroke-dasharray="3 3" pointer-events="all"/>
        <g fill="#ECECEC" font-family="Helvetica" font-size="10px">
        <text x="{x + size_boundary / 2}" y="{y}">{max_value:.0f}</text></g></g>
    """


def get_level(value, min_value, max_value):
    span = max_value - min_value
    if value < min_value + span / 3:
        level = 1
    elif value < min_value + 2 * span / 3:
        level = 2
    else:
        level = 3
    return level


def change_color_text(diagram, text_color, object_id):
    obj = diagram.xpath(f'//svg:g[@id="cell-{object_id}"]', namespaces=nsmap)

    for child in obj[0]:
        # print(child.tag)
        if child.tag.endswith('g'):
            # In multiline text, the color is set in the group tag
            child.set('fill', text_color)
            for child_ in child:
                # print(child_.tag)
                if 'text' in child_.tag:
                    child_.set('fill', text_color)

    return diagram

def update_image(diagram, image_path, object_id):
    binary_fc = open(image_path, 'rb').read()  # fc aka file_content
    base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')

    ext = image_path.split('.')[-1]
    if ext == 'svg': ext = 'svg+xml'
    dataurl = f'data:image/{ext};base64,{base64_utf8_str}'

    obj = diagram.xpath(f'//svg:g[@id="cell-{object_id}"]', namespaces=nsmap)

    # print(obj[0].attrib)

    for child in obj[0]:
        if 'image' in child.tag:
            child.set('{http://www.w3.org/1999/xlink}href', dataurl)

    return diagram