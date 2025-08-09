from typing import Optional, Literal
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from loguru import logger
from phd_visualizations.calculations import (calculate_uncertainty, 
                                             SupportedInstruments,
                                             calculate_metrics,
                                             MetricNames,
                                             MetricsDict)
from phd_visualizations.constants import (default_fontsize, 
                                          plt_colors, 
                                          symbols_open as symbols,
                                          color_palette)
from phd_visualizations.utils import hex_to_rgba_str, compute_axis_range
from phd_visualizations.super_makers import add_super_scatter_trace, SuperMarker

def regression_plot(
    df_ref: pd.DataFrame,
    df_mod: pd.DataFrame | list[pd.DataFrame],
    var_ids: list[str],
    units: list[str] = None,
    instruments: list[SupportedInstruments] = None,
    alternative_labels: list[str] = None,
    show_error_metrics: Optional[list[MetricNames]] = None,
    inline_error_metrics_text: bool = False,
    var_labels: list[str] = None,
    legend_pos: Optional[Literal["side", "top", "top_spaced"]] = None,
    super_marker: Optional[SuperMarker] = None,
    title_margin: int = 100,
    vertical_spacing: float = .1,
    **kwargs
) -> go.Figure:
    
    if instruments is None:
        instruments = [None] * len(var_ids)
    if units is None:
        units = [None] * len(var_ids)
    if var_labels is None:
        var_labels = var_ids
        
    # width = kwargs.get("width", 600)
    v_spacing = kwargs.get("v_spacing", 0.08)
    n_rows = len(var_ids)
    kwargs.setdefault('width', 600)
    height = int(n_rows * kwargs["width"] + (n_rows - 1) * (kwargs["width"] * v_spacing) + title_margin)

    kwargs.setdefault('height', height)
    kwargs.setdefault('margin', dict(l=10, r=10, t=title_margin, b=10))
    kwargs.setdefault('template', 'plotly_white')
    kwargs.setdefault('hoverlabel_namelength', -1) # Show full variable names in hover
    kwargs.setdefault('showlegend', True)
    kwargs.setdefault('title_text', "Model validation")
    kwargs.setdefault('font', dict(size=default_fontsize))
    kwargs.setdefault('hovermode', 'x unified')
    kwargs.setdefault('hoverlabel', dict(
        bgcolor="white",  # "rgb(143, 240, 164, 0.8)",
        font_size=13,
        bordercolor="rgba(192, 191, 188,0.8)",
        namelength=40,
    ))
    
    if legend_pos == "top":
        legend_items = dict(
            legend_orientation="h",
            legend_yanchor="bottom",
            legend_y=1.02,
            legend_xanchor="right",
            legend_x=1,
        )
    elif legend_pos == "side":
        legend_items = dict(
            legend_orientation="v",
            legend_yanchor="top",
            legend_y=1,
            legend_xanchor="left",
            legend_x=1,
            legend_font=dict(size=default_fontsize - 4)
        )
    elif legend_pos == "top_spaced":
        legend_items = dict(
            legend_orientation="h",
            legend_yanchor="bottom",
            legend_y=1.02,
            legend_xanchor="center",
            legend_x=0.5,
            legend_font=dict(size=12),
            legend_valign="middle",
        )
    else:
        raise ValueError("legend_pos must be either 'top' or 'side'")
    
    [kwargs.setdefault(k, v) for k, v in legend_items.items()]
    
    # print(kwargs)

    # Ensure df_mod is a list
    if not isinstance(df_mod, list):
        df_mod = [df_mod]

    if show_error_metrics is not None:
        if len(df_mod) > 1:
            logger.warning(
                "Multiple model results provided, but error metrics will be calculated only for the first model."
            )

        metrics_dicts = [
            calculate_metrics(
                y_true=df_ref[var_id].values,
                y_pred=df_mod[0][var_id].values,
                metrics=show_error_metrics
            )
            for var_id in var_ids
        ]

        separator_str = "<br>" if not inline_error_metrics_text else " | "

        subplot_titles = []
        for var_label, metrics_dict, unit in zip(var_labels, metrics_dicts, units):
            metric_str_parts = []
            for metric_key, value in metrics_dict.items():
                label = MetricsDict[metric_key]["label"]
                unit_type = MetricsDict[metric_key]["unit"]

                # Resolve unit: "idem" -> same as `unit`, "idem_squared" -> f"{unit}²"
                if unit_type == "idem":
                    display_unit = unit
                elif unit_type == "idem_squared":
                    display_unit = f"{unit}<sup>2</sup>"
                else:
                    display_unit = unit_type

                metric_str_parts.append(f"{label}={value:.2f} [{display_unit}]")

            metric_str = separator_str.join(metric_str_parts)
            title = f"<b>{var_label}</b>{separator_str}{metric_str}"
            subplot_titles.append(title)
            
    else:
        subplot_titles = [f"<b>{var_label}</b>" for var_label in var_labels]
        
    fig = make_subplots(
        rows=len(var_ids),
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=vertical_spacing,
        # column_widths=[width] * len(var_ids),
        # row_heights=[width] * len(var_ids),
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
            hoverinfo='skip',
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
            name = alternative_labels[j] if alternative_labels is not None else 'Model results'
            
            if super_marker is None:
                scatter = go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    name=name,
                    showlegend=(i == 0),
                    marker=dict(
                        color=plt_colors[j],
                        size=8,
                        symbol=symbols[j % len(symbols)],
                    ),
                    legendgroup=f"model{j}",
                )
                fig.add_trace(scatter, row=i + 1, col=1)
            else:
                fig = add_super_scatter_trace(
                    fig,
                    x=x,
                    y=y,
                    trace_label=name,
                    size_var_vals=df_ref[super_marker.size_var_id].values,
                    size_var_range=super_marker.size_var_range,
                    fill_var_vals=df_ref[super_marker.fill_var_id].values if super_marker.fill_var_id else None,
                    fill_var_range=super_marker.fill_var_range,
                    ring_vars_vals=[df_ref[var_id].values for var_id in super_marker.ring_vars_ids] if super_marker.ring_vars_ids else None,
                    ring_labels=super_marker.ring_labels if super_marker.ring_labels is not None else super_marker.ring_vars_ids,
                    size_label=super_marker.size_label if super_marker.size_label else super_marker.size_var_id,
                    fill_label=super_marker.fill_label if super_marker.fill_label else super_marker.fill_var_id,
                    marker_params=super_marker.params,
                    row=i + 1,
                    col=1,
                    xref=f"{i+1 if i > 0 else ''}",
                    yref=f"{i+1 if i > 0 else ''}",
                    showlegend=True if i== 0 else False,
                    show_markers_index=super_marker.show_markers_index,
                )

        # Perfect fit line
        regression_line = go.Scatter(
            x=x,
            y=x,
            mode='lines',
            name='Perfect fit',
            showlegend=(i == 0),
            line=dict(color=color_palette["dark_gray"], width=2),
        )
        fig.add_trace(regression_line, row=i + 1, col=1)
        
        y_range = compute_axis_range(x) if super_marker else None
        x_range = compute_axis_range(x) if super_marker else None

        fig.update_yaxes(
            title_text=f"Predicted values [{units[i] if units else ''}]",
            row=i + 1,
            col=1,
            range=y_range,
        )
        fig.update_xaxes(
            title_text=f"Experimental values [{units[i] if units else ''}]",
            row=i + 1,
            col=1,
            range=x_range,
        )

    fig.update_layout(
        **kwargs,
    )

    return fig
