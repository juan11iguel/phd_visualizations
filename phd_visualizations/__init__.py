from pathlib import Path
# import numpy as np
# import pandas as pd
import plotly.graph_objects as go
import typing
from loguru import logger
from .constants import color_palette, plt_colors, default_fontsize, newshape_style

Argument = typing.Literal['eps', 'png', 'svg', 'html']
VALID_FIGURE_FORMATS: typing.Tuple[Argument, ...] = typing.get_args(Argument)


def save_figure(fig: go.Figure, figure_name: str, figure_path: str | Path | list,
                formats: VALID_FIGURE_FORMATS,
                width=600, height=800, scale=2) -> None:

    """ Save figures in different formats """

    if not isinstance(figure_path, list):
        figure_path = [figure_path]

    for path in figure_path:
        for fmt in formats:
            if fmt not in ['eps', 'png', 'svg', 'html']:
                raise ValueError(f'Format {fmt} not supported')

            if fmt == 'html':
                fig.write_html(f'{path}/{figure_name}.{fmt}', include_plotlyjs='cdn', full_html=False)
            else:
                fig.write_image(f'{path}/{figure_name}.{fmt}', format=fmt, width=width, height=height, scale=scale)
                # plt.savefig(f'{path}/{figure_name}.{fmt}', format=fmt)

            logger.info(f'Figure saved in {figure_path}/{figure_name}.{fmt}')
