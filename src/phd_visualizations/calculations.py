from enum import Enum
import numpy as np
from typing import Literal

MetricNames = Literal["rmse", "mape", "mae", "r2", "nrmse_mean", "nrmse_range"]
MetricsDict: dict[str, dict[str, str]] = {
    "rmse": {"label": "RMSE", "unit": "idem_squared"},
    "mape": {"label": "MAPE", "unit": "%"},
    "mae": {"label": "MAE", "unit": "idem"},
    "r2": {"label": "R²", "unit": "-"},
    "nrmse_mean": {"label": "NRMSE (mean)", "unit": "-"},
    "nrmse_range": {"label": "NRMSE (range)", "unit": "-"},
}

class SupportedInstruments(Enum):
    pt100 = "pt100" 
    pt1000 = "pt1000"
    humidity_capacitive = "humidity_capacitive"
    vortex_flow_meter = "vortex_flow_meter"
    paddle_wheel_flow_meter = "paddle_wheel_flow_meter"


def calculate_uncertainty(value: float, instrument: SupportedInstruments | str) -> float:
    """
    Calculate uncertainty for a given value and instrument

    Source:
    Wet cooling tower performance prediction in CSP plants: a comparison between
    artificial neural networks and Poppe’s model
    """
    assert isinstance(instrument, str) | isinstance(instrument, SupportedInstruments), "`instrument` needs to be of correct type"
    
    if isinstance(instrument, str):
        instrument = SupportedInstruments[instrument]

    if instrument == SupportedInstruments.pt100:
        return value * 0.005 + 0.03  # 0.5% of reading + 0.03ºC
    elif instrument == SupportedInstruments.pt1000:
        return 0.5  # 0.5ºC
    elif instrument == SupportedInstruments.humidity_capacitive:
        return 3  # 3%
    elif instrument == SupportedInstruments.vortex_flow_meter:
        return 0.65e-2 * value  # 0.65% of reading
    elif instrument == SupportedInstruments.paddle_wheel_flow_meter:
        return 0.5e-2 * 1.95 + 2.5e-2 * value  # 0.5% full scale + 2.5% of reading
    else:
        raise ValueError(f'Instrument {instrument} not supported, supported instruments are {SupportedInstruments.__members__.keys()}')


# From med-performance-evaluation/med_evaluation
# def calculate_uncertainty(data: pd.DataFrame, instrument: str, meas_range: list = None) -> float:
#     """
#     Direct measurement uncertainty. The uncertainty of each direct measure
#     can be broken down into two components:
#
#         ΔX = ΔX_sensor + ΔX_control
#
#     where:
#     - ΔX_sensor is the contribution of the sensor, which depends on its
#       accuracy, calibration and conversion errors. It can be obtained from
#       the datasheet of the measurement devices (transducer, ADC, etc).
#
#     - ΔX_control is the uncertainty attributed to the quality of the
#       control. For the period considered in steady state, the mean value of
#       the signal is calculated and its standard deviation used as the variation.
#
#     Parameters
#     ----------
#     data : TYPE
#         DESCRIPTION.
#     instrument : string
#         Instrument used to perform measurement. Options are: pt100, pt1000,
#         vortex_flow_meter, paddle_flow_meter.
#
#     range : list, optional
#         Mesurement range (minimum and maximum measurable values). The default is None.
#
#     Raises
#     ------
#     ValueError
#         If an unkown instrument is specified.
#
#     Returns
#     -------
#     float
#         Accumulated measurement uncertainty.
#
#     """
#
#     # Measurement uncertainty
#     if instrument.lower() in ['pt100_class_a', 'pt1000_class_a']:
#         meas_uncertainty = 0.15 + 0.002 * data.mean()
#
#     # elif instrument.lower() == 'vortex_flow_meter':
#     #     meas_uncertainty = 0.65e-2*data.mean()
#
#     elif instrument.lower() in ['pressure_transmitter_wika_s10',
#                                 'pressure_transmitter_cerabar_tpmc131']:
#         meas_uncertainty = 0.5e-2 * abs(meas_range[0] - meas_range[1])
#
#
#
#     # elif instrument.lower() == 'paddle_flow_meter':
#     #     meas_uncertainty = 0.5e-2*abs(meas_range[0]-meas_range[1]) + 2.5e-2*data.mean()
#
#     elif instrument.lower() == 'level_meter_wika_lh10':
#         if data.mean() < 0.25:
#             meas_uncertainty = 0.5e-2 * abs(meas_range[0] - meas_range[1])
#         else:
#             meas_uncertainty = 0.25e-2 * abs(meas_range[0] - meas_range[1])
#
#     elif instrument.lower() in ['flow_meter_eh_proline_promag_50p',
#                                 'flow_meter_eh_proline_promag_p300']:
#         meas_uncertainty = 0.5e-2 * data.mean()
#
#     elif instrument.lower() == 'flow_meter_abb_trio_wirl_vt4':
#         meas_uncertainty = 0.75e-2 * data.mean()
#
#     elif instrument.lower() == 'flow_meter_abb_processmaster_630':
#         meas_uncertainty = 0.4e-2 * data.mean()
#
#     elif instrument.lower() == 'level_meter_igema_na750':
#         meas_uncertainty = 5
#
#     elif instrument.lower() == 'conductivity_meter_prominent_portamess_911':
#         meas_uncertainty = 0.5e-2 * data.mean()
#
#     elif instrument.lower() == 'power_meter_circutor_cem31':
#         # Class 1 according to IEC 62053-21
#         meas_uncertainty = 0.01 * data.mean()
#
#     elif instrument.lower() == 'unknown':
#         meas_uncertainty = 0
#
#     # elif instrument.lower() == ''
#     else:
#         raise ValueError(f'Unknown instrument {instrument}')
#
#     return meas_uncertainty



def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(((y_true - y_pred) ** 2).mean())

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(100 * np.abs(y_true - y_pred) / y_true)

def calculate_nrmse_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized RMSE by mean of y_true."""
    rmse = calculate_rmse(y_true, y_pred)
    return rmse / np.mean(y_true)

def calculate_nrmse_range(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized RMSE by range of y_true."""
    rmse = calculate_rmse(y_true, y_pred)
    return rmse / (np.max(y_true) - np.min(y_true))

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: list[MetricNames]
) -> dict[str, float]:
    
    available_metrics: dict[str, callable] = {
        "r2": calculate_r2,
        "rmse": calculate_rmse,
        "mae": calculate_mae,
        "mape": calculate_mape,
        "nrmse_mean": calculate_nrmse_mean,
        "nrmse_range": calculate_nrmse_range,
    }

    results: dict[str, float] = {}
    for metric in metrics:
        if metric in available_metrics:
            results[metric] = available_metrics[metric](y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return results
