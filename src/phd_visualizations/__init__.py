from pathlib import Path
# import numpy as np
# import pandas as pd
import plotly.graph_objects as go
import typing
from loguru import logger

Argument = typing.Literal['eps', 'png', 'svg', 'html']
VALID_FIGURE_FORMATS: typing.Tuple[Argument, ...] = typing.get_args(Argument)


def save_figure(fig: go.Figure, figure_name: str, figure_path: str | Path | list,
                formats: VALID_FIGURE_FORMATS,
                width:int=None, height:int=None, scale:float=2) -> None:

    """ Save figures in different formats """

    if not isinstance(figure_path, list):
        figure_path = [figure_path]

    if width is None:
        width = fig.layout.width
    if height is None:
        height = fig.layout.height

    for path in figure_path:
        path = Path(path)
        
        for fmt in formats:
            if fmt not in ['eps', 'png', 'svg', 'html']:
                raise ValueError(f'Format {fmt} not supported')
            
            output_path = (path/figure_name).with_suffix(f".{fmt}")
            
            if fmt == 'html':
                fig.write_html(output_path, include_plotlyjs='cdn', full_html=False)
            else:
                fig.write_image(output_path, format=fmt, width=width, height=height, scale=scale)
                # plt.savefig(f'{path}/{figure_name}.{fmt}', format=fmt)

            logger.info(f'Figure saved in {output_path}')
