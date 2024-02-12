import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Literal
from collections import OrderedDict

# Colors definition
color_palette = OrderedDict({
    "cool_red": "#E77C8D", "cool_red_rgb": "231, 124, 141",
    "cool_green":  "#5AA9A2", "cool_green_rgb": "90, 169, 162",
    "bg_gray": "#B6B6B6", "bg_gray_rgb": "182, 182, 182",
    "bg_blue": "steelblue", "bg_blue_rgb": "70, 130, 180",
    "bg_orange": "#e66100", "bg_orange_rgb": "230, 97, 0",
    "gray": "#9a9996", "gray_rgb": "154, 153, 150",
    "yellow": "#e5a50a", "yellow_rgb": "229, 165, 10",
    "plotly_blue": "#636EFA", "plotly_blue_rgb": "99, 110, 250",
    "plotly_green": "#00CC96", "plotly_green_rgb": "0, 204, 150",
    "plotly_red": "#EF553B", "plotly_red_rgb": "239, 85, 59",
    "plotly_cyan": "#19D3F3", "plotly_cyan_rgb": "25, 211, 243",
    "plotly_yellow": "#FECB52", "plotly_yellow_rgb": "254, 203, 82",
    "plotly_orange": "#FFA15A", "plotly_orange_rgb": "255, 161, 90",
    "dc_green": "#83b366", "dc_green_rgb": "131, 179, 102",
    "wct_purple": "#9573a6", "wct_purple_rgb": "149, 115, 166",
    "c_blue": "#6c8ebf", "c_blue_rgb": "108, 142, 191",
    "turquesa": "#00A08B", "turquesa_rgb": "0, 160, 139",
})

default_fontsize = 16
plt_colors = px.colors.qualitative.Plotly * 3 # * 3 to have plenty of colors
ArrayLike = np.ndarray | list | pd.DataFrame | pd.Series

# style of user-drawn shapes
newshape_style: dict = dict(
    line_color=color_palette['plotly_green'],
    fillcolor=color_palette['plotly_green'],
    opacity=0.5,
    layer="below"
)


def generate_plotly_config(fig: go.Figure, figure_name: str = 'solhycool_plot',
                           file_format: Literal['png', 'svg', 'jpeg', 'webp'] = 'png',
                           height: int = None, width: int = None, scale: int = 2) -> dict:
    # configuration options for the plotly figure
    return dict(
        toImageButtonOptions={
            'format': file_format,  # one of png, svg, jpeg, webp
            'filename': figure_name,
            'height': height if height else fig.layout.height,
            'width': width if width else fig.layout.width,
            'scale': scale  # Multiply title/legend/axis/canvas sizes by this factor
        },

        modeBarButtonsToAdd=[
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ],

        modeBarButtonsToRemove=[
            'autoScale2d' # Causes problems with plotly_resampler, equivalent to double-clicking on the plot
        ]
    )
