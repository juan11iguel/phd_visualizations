from typing import Literal
# import numpy as np

supported_instruments = Literal['pt100', 'pt1000', 'humidity_capacitive', 'vortex_flow_meter', 'paddle_wheel_flow_meter']

def calculate_uncertainty(value: float, instrument: supported_instruments) -> float:
    """
    Calculate uncertainty for a given value and instrument

    Source:
    Wet cooling tower performance prediction in CSP plants: a comparison between
    artificial neural networks and Poppe’s model
    """

    if instrument.lower() == 'pt100':
        return value * 0.005 + 0.03  # 0.5% of reading + 0.03ºC
    elif instrument.lower() == 'pt1000':
        return 0.5  # 0.5ºC
    elif instrument.lower() == 'humidity_capacitive':
        return 3  # 3%
    elif instrument.lower() == 'vortex_flow_meter':
        return 0.65e-2 * value  # 0.65% of reading
    elif instrument.lower() == 'paddle_wheel_flow_meter':
        return 0.5e-2 * 1.95 + 2.5e-2 * value  # 0.5% full scale + 2.5% of reading
    else:
        raise ValueError(f'Instrument {instrument} not supported, supported instruments are {supported_instruments}')