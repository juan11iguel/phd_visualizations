import base64
from lxml import etree
from copy import deepcopy
from typing import Literal
from loguru import logger
import warnings
# import os
# import logging
# from pathlib import Path
from . import utils

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


def find_object(object_id: str, diagram: etree._ElementTree, group: bool = False) -> etree.ElementBase:
    
    if not group:
        object = diagram.xpath(f'//svg:g[@id="cell-{object_id}"]', namespaces=nsmap)
    else:
        object = diagram.xpath(f'//svg:g[starts-with(@id, "cell-{object_id}")]', namespaces=nsmap)
    
    if not object:
        raise ValueError(f'Object {object_id} not found in diagram')
    else:
        return object

def change_text(object_id, diagram, new_text: str, not_inplace: bool = False) -> etree.ElementBase | None:
    
    centinela = False
    
    def update_text(child):
        if child.tag.endswith('g'):
           for child_ in child:
                update_text(child_)
        elif child.tag.endswith('text'):
            logger.debug(f'Changing text of {object_id}/{child.get("id")} to {new_text}')
            child.text = new_text
            
            nonlocal centinela
            centinela = True
            
            return # Exit on first match
        # Recursive call
        elif len(child) > 0:
            for child_ in child:
                update_text(child_)
        
    if not_inplace:
        diagram = copy.deepcopy(diagram)
    object = find_object(object_id, diagram, group=False)
    for child in object:        
        update_text(child)
    
    if not centinela:
        logger.error(f'Could not find any text object in {object_id}')
    
    return diagram if not_inplace else None

def change_icon(object_id:str, diagram: etree._ElementTree, size: int, text: str = None, 
                max_size: int = None, max_value: int | float = None, include_boundary:bool=False,
                not_inplace:bool = False, group:bool=False) -> etree.ElementBase | None:
    
    if include_boundary:
        assert max_size is not None and max_value is not None, "max_size and max_value must be provided if include_boundary is True"
    
    centinela = False
    
    def change_size(child):
        if "image" in child.tag:
            logger.debug(f'Changing size of {object_id}/{child.get("id")} to {size}')
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

            # Add template-id property to be used later ??
            child.set("template-id", f'icon-{object_id}')
            
            # Add boundary circle
            if include_boundary:
                object[0][0].addprevious(etree.fromstring(utils.generate_boundary_circle(object_id, size, max_size, max_value, pos_x, pos_y)))
            
            nonlocal centinela
            centinela = True
            
            return # Exit on first match
        
        # Recursive call
        elif len(child) > 0:
            for child_ in child:
                change_size(child_)
            
    if not_inplace:
        diagram = copy.deepcopy(diagram)
            
    object = find_object(object_id, diagram, group=group)
            
    if text is not None:
        change_text(object_id, diagram, text, not_inplace=False)
    
    # Change icon size and add boundary circle
    for child in object:        
        change_size(child)
    if not centinela:
        logger.error(f'Could not find any icon object in {object_id}')
        
        
    return diagram if not_inplace else None


def adjust_icon(id, size, tag, value, unit, include_boundary=True, max_size=None, max_value=None):
    
    warnings.warn("This function is deprecated. Use change_icon instead", DeprecationWarning)
    
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
                        child2.text = f'{utils.round_to_nonzero_decimal(value)} {unit}'

    # Add boundary circle
    if include_boundary:
        tag[0][0].addprevious(etree.fromstring(utils.generate_boundary_circle(id, size, max_size, max_value, pos_x, pos_y)))
        
    return tag, pos_x, pos_y


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

    
def change_bg_color(object: etree._Element, color: str, not_inplace:bool = False, tag_key: Literal['rect', 'path'] = 'rect'):
    
    centinela = False
    
    def change_color(child):
        
        if tag_key in child.tag:
            logger.debug(f'Changing fill color of {child.get("id")} to {color}')
            child.set('fill', color)
            
            nonlocal centinela
            centinela = True
            
            return
        
        elif len(child) > 0:
                for child_ in child:
                    change_color(child_)
            
    if not_inplace:
        object = deepcopy(object)
            
    for child in object:
        change_color(child)
        
    if not centinela:
        logger.error('Could not find any path object to change its fill color')
        
    return object if not_inplace else None
    
def change_line_width(object_id:str, diagram: etree._ElementTree, width: int, not_inplace:bool = False, tag_key: Literal['rect', 'path'] = 'path', group:bool=False, stop_on_first_change=False) -> etree.ElementBase | None:
    
    centinela = False
    
    def change_width(child):
        
        if tag_key in child.tag:
            logger.debug(f'Changing width of {object_id}/{child.get("id")} to {width}')
            child.set('stroke-width', str(width))
            
            nonlocal centinela
            centinela = True
            
            if stop_on_first_change:
                return
        
        elif len(child) > 0:
                for child_ in child:
                    change_width(child_)
            
    if not_inplace:
        diagram = deepcopy(diagram)
            
    object = find_object(object_id, diagram, group=group)
            
    for child in object:
        # if centinela and stop_on_first_change:
        #     break
        
        change_width(child)
        
    if not centinela:
        logger.error(f'Could not find any {tag_key} object to change its width in {object_id}')
        
    return diagram if not_inplace else None


def change_line_color(object_id:str, diagram: etree._ElementTree, color: str, not_inplace:bool = False, tag_key: Literal['rect', 'path'] = 'path', group:bool=False, stop_on_first_change=False) -> etree.ElementBase | None:
    
    centinela = False
    
    def change_property(child):
        
        if tag_key in child.tag:
            logger.debug(f'Changing color of {object_id}/{child.get("id")} to {color}')
            child.set('stroke', str(color))
            
            nonlocal centinela
            centinela = True
            
            if stop_on_first_change:
                return
        
        elif len(child) > 0:
                for child_ in child:
                    change_property(child_)
            
    if not_inplace:
        diagram = deepcopy(diagram)
            
    object = find_object(object_id, diagram, group=group)
            
    for child in object:
        # if centinela and stop_on_first_change:
        #     break
        
        change_property(child)
        
    if not centinela:
        logger.error(f'Could not find any {tag_key} object to change its width in {object_id}')
        
    return diagram if not_inplace else None