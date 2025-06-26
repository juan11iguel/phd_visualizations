import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
import numpy as np
from phd_visualizations.calculations import calculate_uncertainty, SupportedInstruments
from phd_visualizations.constants import default_fontsize, plt_colors, symbols_open as symbols
from phd_visualizations.utils import hex_to_rgba_str

def regression_plot(
    df_ref: pd.DataFrame,
    df_mod: pd.DataFrame | list[pd.DataFrame],
    var_ids: list[str],
    units: list[str] = None,
    instruments: list[SupportedInstruments] = None,
    alternative_labels: list[str] = None,
    show_error_metrics: bool = True,
    var_labels: list[str] = None,
    **kwargs
) -> go.Figure:
    
    if instruments is None:
        instruments = [None] * len(var_ids)
    if units is None:
        units = [None] * len(var_ids)
    if var_labels is None:
        var_labels = var_ids

    kwargs.setdefault('template', 'plotly_white')
    kwargs.setdefault('showlegend', True)
    kwargs.setdefault('title_text', "Model validation")
    kwargs.setdefault('height', 300 * len(var_ids))
    kwargs.setdefault('font', dict(size=default_fontsize))

    # Ensure df_mod is a list
    if not isinstance(df_mod, list):
        df_mod = [df_mod]

    if show_error_metrics:
        rmse_list = [
            np.sqrt(mean_squared_error(df_ref[var_id].values, df_mod[0][var_id].values))
            for var_id in var_ids
        ]
        
    fig = make_subplots(
        rows=len(var_ids),
        cols=1,
        subplot_titles=[
            f"<b>{var_label}</b><br>RMSE={rmse:.2f} [{unit}]"
            for var_label, rmse, unit in zip(var_labels, rmse_list, units)
        ] if show_error_metrics else [f"<b>{var_label}</b>" for var_label in var_labels],
    )

    for i, var_id in enumerate(var_ids):
        x = df_ref[var_id].values
        
        # Sort by increasing x values and get the sorted indices
        sorted_indices = np.argsort(x)
        x = x[sorted_indices]

        # Calculate uncertainty
        if instruments[i]:
            uncertainty = calculate_uncertainty(x, instruments[i])
        else:
            uncertainty = np.zeros_like(x)

        # Uncertainty bounds
        color = plt_colors[-1]
        color_ = hex_to_rgba_str(color, alpha=0.4)
        upper_bound = go.Scatter(
            x=x,
            y=x + uncertainty,
            mode='lines',
            fill=None,
            line=dict(color=color, width=0.1),
            fillcolor=color_,
            showlegend=False,
            name=f'Uncertainty {var_id}',
        )
        lower_bound = go.Scatter(
            x=x,
            y=x - uncertainty,
            mode='lines',
            fill='tonexty',
            line=dict(color=color, width=0.1),
            fillcolor=color_,
            showlegend=True if i == 0 else False,
            name='Sensor uncertainty',
        )
        
        fig.add_trace(upper_bound, row=i + 1, col=1)
        fig.add_trace(lower_bound, row=i + 1, col=1)

        # Add scatter for each df_mod
        for j, df_mod_j in enumerate(df_mod):
            y = df_mod_j[var_id].values[sorted_indices]
            scatter = go.Scatter(
                x=x,
                y=y,
                mode='markers',
                name=alternative_labels[j] if alternative_labels is not None else 'Model results',
                showlegend=(i == 0),
                marker=dict(
                    color=plt_colors[j],
                    size=8,
                    symbol=symbols[j % len(symbols)],
                ),
                legendgroup=f"model{j}",
            )
            fig.add_trace(scatter, row=i + 1, col=1)

        # Perfect fit line
        regression_line = go.Scatter(
            x=x,
            y=x,
            mode='lines',
            name='Perfect fit',
            showlegend=(i == 0),
            line=dict(color=plt_colors[0], width=2),
        )
        fig.add_trace(regression_line, row=i + 1, col=1)

        fig.update_yaxes(
            title_text=f"Predicted values [{units[i] if units else ''}]",
            row=i + 1,
            col=1,
        )
        fig.update_xaxes(
            title_text=f"Experimental values [{units[i] if units else ''}]",
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ) if len(df_mod) == 1 else dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1,
            font=dict(size=default_fontsize - 4)
        ),
        **kwargs,
    )

    return fig
