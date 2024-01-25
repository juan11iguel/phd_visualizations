from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Literal
from loguru import logger
from .constants import color_palette, plt_colors, default_fontsize, newshape_style

def save_figure(fig: go.Figure, figure_name: str, figure_path: str | Path, formats: List[Literal['eps', 'png', 'svg']],
                width=600, height=800, scale=2) -> None:

    """ Save figures in different formats """

    for fmt in formats:
        if fmt not in ['eps', 'png', 'svg', 'html']:
            raise ValueError(f'Format {fmt} not supported')

        if fmt == 'html':
            fig.write_html(f'{figure_path}/{figure_name}.{fmt}', include_plotlyjs='cdn', full_html=False)
        else:
            fig.write_image(f'{figure_path}/{figure_name}.{fmt}', format=fmt, width=width, height=height, scale=scale)
            # plt.savefig(f'{figure_path}/{figure_name}.{fmt}', format=fmt)

        logger.info(f'Figure saved in {figure_path}/{figure_name}.{fmt}')
