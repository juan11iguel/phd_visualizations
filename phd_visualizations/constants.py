import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Literal
from collections import OrderedDict

# Colors definition
color_palette = OrderedDict({
    "cool_red": "#E77C8D",
    "cool_green":  "#5AA9A2",
    "bg_gray": "#B6B6B6",
    "bg_blue": "steelblue",
    "bg_orange": "#e66100",
    "bg_red": "#b2182b",
    "gray": "#9a9996",
    "yellow": "#e5a50a",
    "plotly_blue": "#636EFA",
    "plotly_green": "#00CC96",
    "plotly_red": "#EF553B",
    "plotly_cyan": "#19D3F3",
    "plotly_yellow": "#FECB52",
    "plotly_orange": "#FFA15A",
    "dc_green": "#83b366",
    "wct_purple": "#9573a6",
    "c_blue": "#6c8ebf",
    "turquesa": "#00A08B",
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

named_css_colors = [
    "darkslategray", "darkorange", "darkorchid", "firebrick", "teal", "darkgreen",
    "aliceblue", "antiquewhite", "aqua", "aquamarine", "azure",
    "beige", "bisque", "black", "blanchedalmond", "blue",
    "blueviolet", "brown", "burlywood", "cadetblue",
    "chartreuse", "chocolate", "coral", "cornflowerblue",
    "cornsilk", "crimson", "cyan", "darkblue", "darkcyan",
    "darkgoldenrod", "darkgray", "darkgrey",
    "darkkhaki", "darkmagenta", "darkolivegreen",
    "darkred", "darksalmon", "darkseagreen",
    "darkslateblue", "darkslategrey",
    "darkturquoise", "darkviolet", "deeppink", "deepskyblue",
    "dimgray", "dimgrey", "dodgerblue",
    "floralwhite", "forestgreen", "fuchsia", "gainsboro",
    "ghostwhite", "gold", "goldenrod", "gray", "grey", "green",
    "greenyellow", "honeydew", "hotpink", "indianred", "indigo",
    "ivory", "khaki", "lavender", "lavenderblush", "lawngreen",
    "lemonchiffon", "lightblue", "lightcoral", "lightcyan",
    "lightgoldenrodyellow", "lightgray", "lightgrey",
    "lightgreen", "lightpink", "lightsalmon", "lightseagreen",
    "lightskyblue", "lightslategray", "lightslategrey",
    "lightsteelblue", "lightyellow", "lime", "limegreen",
    "linen", "magenta", "maroon", "mediumaquamarine",
    "mediumblue", "mediumorchid", "mediumpurple",
    "mediumseagreen", "mediumslateblue", "mediumspringgreen",
    "mediumturquoise", "mediumvioletred", "midnightblue",
    "mintcream", "mistyrose", "moccasin", "navajowhite", "navy",
    "oldlace", "olive", "olivedrab", "orange", "orangered",
    "orchid", "palegoldenrod", "palegreen", "paleturquoise",
    "palevioletred", "papayawhip", "peachpuff", "peru", "pink",
    "plum", "powderblue", "purple", "red", "rosybrown",
    "royalblue", "saddlebrown", "salmon", "sandybrown",
    "seagreen", "seashell", "sienna", "silver", "skyblue",
    "slateblue", "slategray", "slategrey", "snow", "springgreen",
    "steelblue", "tan", "thistle", "tomato", "turquoise",
    "violet", "wheat", "white", "whitesmoke", "yellow",
    "yellowgreen"
]

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
