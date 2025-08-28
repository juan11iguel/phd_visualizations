from typing import Literal, Optional
from pathlib import Path
import pandas as pd
import copy
from loguru import logger
import numpy as np
from plotly.colors import hex_to_rgb
from PIL import Image

def validate_hex_color(hex_str: str) -> bool:
    if not isinstance(hex_str, str):
        return False
    if len(hex_str) != 7:
        return False
    if hex_str[0] != '#':
        return False
    try:
        int(hex_str[1:], 16)
    except ValueError:
        return False
    return True

def hex_to_rgb_str(hex_str: str) -> str:
    """
    Convert a hex color string to an RGB string.
    :param hex_str: Hex color string in the format '#RRGGBB'.
    :return: RGB string in the format 'rgb(r, g, b)'.
    """
    return f'rgb{hex_to_rgb(hex_str)}'

def hex_to_rgba_str(hex_str: str, alpha: float) -> str:
    if not validate_hex_color(hex_str):
        raise ValueError(f"Invalid hex color: {hex_str}")

    rgb_col = hex_to_rgb(hex_str)
    return f"rgba({rgb_col[0]}, {rgb_col[1]}, {rgb_col[2]}, {alpha})"

def rgb_str_to_rgba_str(rgb_str: str, alpha: float) -> str:
    """
    Convert an RGB string to an RGBA string with the specified alpha value.
    :param rgb_str: RGB string in the format 'rgb(r, g, b)'.
    :param alpha: Alpha value to set in the RGBA string.
    :return: RGBA string in the format 'rgba(r, g, b, alpha)'.
    """
    if not rgb_str.startswith('rgb(') or not rgb_str.endswith(')'):
        raise ValueError(f"Invalid RGB string format: {rgb_str}")
    
    rgb_values = rgb_str[4:-1].split(',')
    return f"rgba({rgb_values[0]}, {rgb_values[1]}, {rgb_values[2]}, {alpha})"

def rgba_str_to_rgb_str(rgba_str: str) -> str:
    """
    Convert an RGBA string to an RGB string by removing the alpha value.
    :param rgba_str: RGBA string in the format 'rgba(r, g, b, a)'.
    :return: RGB string in the format 'rgb(r, g, b)'.
    """
    if not rgba_str.startswith('rgba(') or not rgba_str.endswith(')'):
        raise ValueError(f"Invalid RGBA string format: {rgba_str}")
    
    rgb_values = rgba_str[5:-1].split(',')
    return f"rgb({rgb_values[0]}, {rgb_values[1]}, {rgb_values[2]})"

def any_str_to_rgb_str(color: str) -> str:
    """
    Convert a color string to an RGB string.
    :param color: Color string, can be a hex string, RGBA string.
    :return: RGB string in the format 'rgb(r, g, b)'.
    """
    if validate_hex_color(color):
        return hex_to_rgb_str(color)
    elif color.startswith('rgba(') and color.endswith(')'):
        return rgba_str_to_rgb_str(color)
    else:
        raise ValueError(f"Invalid color format: {color}")

class ColorChooser:
    """
    Terrible name I know, could better be RandomColorPicker or something like that.
    """
    def __init__(self, color_options):
        self.color_options = color_options
        self.last_choice = None

    def choose(self):
        available_options = [option for option in self.color_options if option != self.last_choice]
        self.last_choice = np.random.choice(available_options)
        return self.last_choice

Operators = {
    '<=': lambda x, y: x <= y,
    '>=': lambda x, y: x >= y,
    '<': lambda x, y: x < y,
    '>': lambda x, y: x > y
}

def rename_signal_ids_to_var_ids(df: pd.DataFrame, vars_config: dict) -> pd.DataFrame:
    """
    Rename signal ids to var ids in a dataframe.
    :param df: Dataframe to rename signal ids to var ids in.
    :param vars_config: Dictionary with variables configuration, should contain var_id
    and signal_id for each variable.
    :return: None
    """

    # Simple solution when there are no duplicates
    # var_ids, signal_ids = zip(*[(var_info['var_id'], var_info['signal_id']) for var_info in vars_config.values()])
    # return df.rename(columns=dict(zip(signal_ids, var_ids)))

    # Create a dictionary to keep track of the signal_id duplicates
    duplicates = {}

    for var_info in vars_config.values():
        if 'signal_id' not in var_info:
            logger.warning(f"Signal id not found in variable {var_info['var_id']}, skipping")
            continue
        signal_id = var_info['signal_id']
        var_id = var_info['var_id']

        if signal_id in df.columns:
            if signal_id in duplicates:
                # If the signal_id is already in the duplicates dictionary, create a new column
                idx = len(duplicates[signal_id])
                df[f"{signal_id}_{idx}"] = df[signal_id]
                duplicates[signal_id].append(var_id)
            else:
                # If the signal_id is not in the duplicates dictionary, add it
                duplicates[signal_id] = [var_id]
        else:
            logger.warning(f"Signal id {signal_id} not found in dataframe columns.")

    # Assign each var_id to one of the copied signal_id{some_idx} columns
    for signal_id, var_ids in duplicates.items():
        for idx, var_id in enumerate(var_ids):
            signal_name = f"{signal_id}_{idx}" if idx > 0 else signal_id

            df.rename(columns={signal_name: var_id}, inplace=True)

    return df


def tuple_to_string(input_tuple):
    return ', '.join(str(i) for i in input_tuple)


def stack_images_vertically(image_path_top: Path, image_path_bottom: Path, output_path: Path):
    # Load the images
    top_image = Image.open(image_path_top)
    bottom_image = Image.open(image_path_bottom)
    
    # Calculate dimensions for the new image
    total_height = top_image.height + bottom_image.height
    max_width = max(top_image.width, bottom_image.width)
    
    # Create a new image with appropriate dimensions
    new_image = Image.new('RGB', (max_width, total_height))
    
    # Paste the top image at the top of the new image
    new_image.paste(top_image, (0, 0))
    
    # Paste the bottom image below the top image
    new_image.paste(bottom_image, (0, top_image.height))
    
    # Save or display the new image
    new_image.save(output_path, )
    # new_image.show()  # Uncomment to display the image
    
def stack_images_horizontally(image_paths: list[Path], output_path: Path) -> None:
    # Load all images
    images = [Image.open(path) for path in image_paths]
    
    # Calculate total width and max height for the new image
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    
    # Create a new blank image with the combined dimensions
    new_image = Image.new('RGB', (total_width, max_height))
    
    # Paste images side by side
    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width
    
    # Save the new image
    new_image.save(output_path)
    
def find_n_best_values_in_list(source_list: list[list[float]], n: int, objective: Literal["minimize", "maximize"] = "minimize") -> tuple[list[int], list[float]]:

    best_idxs = [None] * (n)
    best_fitness_list = [float("inf")] * (n)
    
    if objective == "minimize":
        fitness_list = [np.min(np.array(case)) for case in source_list if len(case) > 0]
    else:
        fitness_list = [np.max(np.array(case)) for case in source_list if len(case) > 0]

    for idx, fitness in enumerate(fitness_list):            
        for i, best_fitness in enumerate(best_fitness_list):
            # print(f"{fitness=} vs {best_fitness=} in position {i}")

            if (objective == "minimize" and fitness < best_fitness) or (objective == "maximize" and fitness > best_fitness):
                # Shift elements to the right from i
                best_fitness_list[i+1:] = best_fitness_list[i:-1]
                best_idxs[i+1:] = best_idxs[i:-1]
                
                # Insert new best fitness and index at position i
                best_fitness_list[i] = fitness
                best_idxs[i] = idx
                break

    logger.info(f"{best_fitness_list=} at {best_idxs=}")
    return best_idxs, best_fitness_list


def compute_axis_range(data, padding_ratio=0.1):
    data = np.asarray(data)
    data = data[~np.isnan(data)]  # Remove NaNs
    if data.size == 0:
        return None  # or a default range like [0, 1]
    data_min, data_max = np.min(data), np.max(data)
    span = data_max - data_min
    padding = span * padding_ratio if span > 0 else 1.0  # avoid zero span
    return [data_min - padding, data_max + padding]


def update_plot_config(
    plot_config: dict, 
    show_titles: bool, 
    show_left_axis_titles: bool,
    show_right_axis_titles: bool, 
    showlegends: bool,
    show_main_title: bool = False,
    width: Optional[int] = None,
) -> dict:
    pc = copy.deepcopy(plot_config)
    
    if not show_main_title:
        pc["title"] = ""
        pc["subtitle"] = ""
    if width is not None:
        pc["width"] = width
    for plot in pc["plots"].values():
        if not show_titles:
            plot["title"] = ""
        if not show_left_axis_titles:
            plot["ylabels_left"] = ["" for _ in plot["ylabels_left"]]
        if "ylabels_right" in plot and not show_right_axis_titles:
                plot["ylabels_right"] = ["" for _ in plot["ylabels_right"]]
        if not showlegends:
            plot["showlegend"] = False
            for trace in plot.get("traces_left", []):
                trace["showlegend"] = False
            for trace in plot.get("traces_right", []):
                trace["showlegend"] = False
    return pc