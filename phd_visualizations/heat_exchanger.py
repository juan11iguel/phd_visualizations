import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Annotated
import pandas as pd
import numpy as np
from iapws import IAPWS97 as w_props
from loguru import logger

dateStrType = Annotated[str, 'Date of the simulation', '2023-10-30 12:00:00']

def steady_state_viz(df: pd.DataFrame, date_idx: dateStrType | list[dateStrType],
                     primary_ids: tuple[str, str] = ('Thx_p_in', 'Thx_p_out'),
                     secondary_ids: tuple[str, str] = ('Thx_s_in', 'Thx_s_out'),
                     include_limits: bool = True) -> go.Figure:
    # date_idx = '2023-10-30 12:00:00'
    date_idx = date_idx if isinstance(date_idx, list) else [date_idx]

    # Define x and y values
    x_values = [0, 0.5, 1]

    # Add traces to a figure
    fig = make_subplots(rows=1, cols=len(date_idx), subplot_titles=[f"{date[11:]}" for date in date_idx],
                        shared_yaxes=True)

    for i, date in enumerate(date_idx):
        Tp_values = [df.loc[date][primary_ids[0]], df.loc[date][primary_ids[1]]]
        Ts_values = [df.loc[date][secondary_ids[1]], df.loc[date][secondary_ids[0]]]

        Tp_values.insert(1, (Tp_values[0]+Tp_values[1])/2)
        Ts_values.insert(1, (Ts_values[0]+Ts_values[1])/2)

        # Create line traces
        fig.add_trace(
            go.Scatter(
                x=x_values, y=Tp_values, mode='lines+markers+text', name='Tp', line=dict(shape='linear', color='red'),
                marker=dict(symbol='triangle-se', size=15), showlegend=True if i==0 else False,
                text=["T<sub>p,in</sub>", None, "T<sub>p,out</sub>"] if i==0 else None,
                textposition="top center", textfont=dict(size=12,)
            ),
            row=1, col=i+1
        )

        fig.add_trace(
            go.Scatter(
                x=x_values, y=Ts_values, mode='lines+markers+text', name='Ts', line=dict(shape='linear', color='blue'),
                marker=dict(symbol='triangle-nw', size=15), showlegend=True if i==0 else False,
                text=["T<sub>s,out</sub>", None, "T<sub>s,in</sub>"] if i == 0 else None,
                textposition="top center", textfont=dict(size=12, )
            ),
            row=1, col=i+1
        )

        if include_limits:
            Tp_in = Tp_values[0]
            Ts_in = Tp_values[-1]
            Tp_out = Tp_values[-1]
            Ts_out = Ts_values[0]

            qp = df.loc[date]['qhx_p']
            qs = df.loc[date]['qhx_s']

            wprops_p = w_props(P=0.16, T=Tp_in + 273.15)
            wprops_s = w_props(P=0.16, T=Ts_in + 273.15)

            cp_p = wprops_p.cp * 1e3  # P=1 bar->0.1 MPa C, cp [KJ/kg·K] -> [J/kg·K]
            cp_s = wprops_s.cp * 1e3  # P=1 bar->0.1 MPa C, cp [KJ/kg·K] -> [J/kg·K]

            Cp = qp / 3600 * wprops_p.rho * cp_p  # P=1 bar->0.1 MPa C, cp [KJ/kg·K] -> [J/kg·K]
            Cs = qs / 3600 * wprops_s.rho * cp_s  # P=1 bar->0.1 MPa C, cp [KJ/kg·K] -> [J/kg·K]
            Cmin = np.min([Cp, Cs])

            Qmax = Cmin * (Tp_in - Ts_in)

            # In the limit case
            Tp_out_min = Tp_in - Qmax / Cp
            Ts_out_max = Ts_in + Qmax / Cs

            # Calculate power
            Phx_p = qp / 3600 * wprops_p.rho * cp_p * (Tp_in - Tp_out)
            Phx_s = qs / 3600 * wprops_s.rho * cp_s * (Ts_out - Ts_in)
            #
            # logger.info(f'Tp_out_min: {Tp_out_min:.2f} C, Ts_out_max: {Ts_out_max:.2f} C')
            # logger.info(f'Measured values, Tp_out: {ds["Thx_p_out"]:.2f} C, Ts_out: {ds["Thx_s_out"]:.2f} C')
            # logger.info(f'Measured power, Phx_p: {Phx_p:.2f} W, Phx_s: {Phx_s:.2f} W')

            logger.info(f"Qmax: {Qmax:.0f} W, Phx_p: {Phx_p*1e-3:.0f} kW, Phx_s: {Phx_s*1e-3:.0f} kW, Cp: {Cp:.0f} W/K, Cs: {Cs:.0f} W/K")

            fig.add_hline(
                y=Tp_out_min, line=dict(color='red', width=1, dash='dash'), row=1, col=i + 1,
                annotation=dict(font_size=14),
                annotation_text=f'T<sub>p,out,min</sub> {Tp_out_min:.0f}ºC', annotation_position='bottom right'
            )

            fig.add_hline(
                y=Ts_out_max, line=dict(color='blue', width=1, dash='dash'), row=1, col=i + 1,
                annotation=dict(font_size=14),
                annotation_text=f'T<sub>s,out,max</sub> {Ts_out_max:.0f}ºC', annotation_position='top right'
            )

    # Define layout
    fig.update_layout(
        xaxis_title="x/L<sub>hx</sub>",
        yaxis_title="Temperature (ºC)",
        plot_bgcolor="rgba(182, 182, 182, 0.1)",
        title=f"<b>Heat Exchanger</b> state at <br>{date_idx[0]}</br>" if len(date_idx) == 1 else f"<b>Heat Exchanger</b> state(s) at {date_idx[0][0:10]}",
        legend=dict(x=1, y=1.2, xanchor='right', bgcolor='rgba(0,0,0,0)'),
        autosize=False,
        width=np.min([500*len(date_idx), 1200]),
        height=500,
        margin=dict(l=50, r=50, b=100, t=100, pad=4)
    )

    return fig