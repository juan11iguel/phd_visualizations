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
    reference_error_lines: list[float] = None,
    df_ref_bg: Optional[pd.DataFrame] = None,
    df_mod_bg: Optional[pd.DataFrame] = None,
    bg_label: Optional[str] = None,
    **kwargs
) -> go.Figure:
    
    """Create a regression plot comparing reference and model data.
    Optionally, background data can be added for context.
    
    Parameters
    ----------
    df_ref : pd.DataFrame
        DataFrame containing reference data.
    df_mod : pd.DataFrame or list of pd.DataFrame
        DataFrame(s) containing model data to compare against reference.
    var_ids : list of str
        List of variable IDs (column names) to plot.
    units : list of str, optional
        List of units corresponding to each variable ID. If None, no units are shown.
    instruments : list of SupportedInstruments, optional
        List of instruments corresponding to each variable ID for uncertainty calculation.
        If None, no uncertainty is shown.
    alternative_labels : list of str, optional
        Alternative labels for the variables. If None, var_ids are used.
    show_error_metrics : list of MetricNames, optional
        List of error metrics to calculate and display in subplot titles.
    inline_error_metrics_text : bool, default False
        If True, error metrics are shown inline in subplot titles. Otherwise, they are shown below.
    var_labels : list of str, optional
        List of labels for the variables to be used in subplot titles. If None, var_ids are used.
    legend_pos : str, optional
        Position of the legend. Options are "side", "top", or "top_spaced". If None, no legend is shown.
    super_marker : SuperMarker, optional
        Configuration for super markers to highlight specific data points.
    title_margin : int, default 100
        Margin for the title area in pixels.
    vertical_spacing : float, default 0.1
        Vertical spacing between subplots as a fraction of the plot height.
    reference_error_lines : list of float, optional
        List of relative error values (e.g., [0.05, 0.1]) to plot as reference lines.
    df_ref_bg : pd.DataFrame, optional
        DataFrame containing background reference data for context.
    df_mod_bg : pd.DataFrame, optional
        DataFrame containing background model data for context.
    bg_label : str, optional
        Label for the background data in the legend.
    **kwargs
        Additional keyword arguments passed to plotly.graph_objects.Figure.
    
    Returns
    -------
    go.Figure
        Plotly Figure object containing the regression plots.
    """
    
    assert df_ref_bg is None and df_mod_bg is None or (df_ref_bg is not None and df_mod_bg is not None), \
        "Both df_ref_bg and df_mod_bg must be provided to plot background data."
    
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
            showlegend=True if (i == 0 and instruments != [None]*len(var_ids)) else False,
            name='Sensor uncertainty',
        )
        
        fig.add_trace(upper_bound, row=i + 1, col=1)
        fig.add_trace(lower_bound, row=i + 1, col=1)

        # Reference error lines
        if reference_error_lines is not None:
            for rel_error in reference_error_lines:
                if rel_error <= 0:
                    logger.waring(f"Reference error line {rel_error} is not positive and will be skipped.")
                    continue
                if rel_error > 1.0:
                    logger.waring(f"Reference error line {rel_error} is greater than 1.0 (100%) and will be skipped.")
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=x * (1 + rel_error),
                        mode='lines+text',
                        name=f'+{rel_error*100:.0f}%',
                        showlegend=False, # (i == 0),
                        line=dict(color=color_palette["bg_gray"], width=1, dash='dash'),
                        hoverinfo='skip',
                             # Nothing in for all but last point
                        text=[''] * (len(x) - 1) + [f'{rel_error*100:.0f}%'],
                        textposition='top center',
                        textfont=dict(size=10, color=color_palette["bg_gray"]),
                    ), 
                    row=i + 1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=x * (1 - rel_error),
                        mode='lines',
                        name=f'-{rel_error*100:.0f}%',
                        showlegend=False,
                        line=dict(color=color_palette["bg_gray"], width=1, dash='dash'),
                        hoverinfo='skip',
                    ), 
                    row=i + 1, col=1
                )
                
        # Add background data if provided
        if df_ref_bg is not None:
            x_bg = df_ref_bg[var_id].values[sorted_indices]
            y_bg = df_mod_bg[var_id].values[sorted_indices]
            fig.add_trace(
                go.Scatter(
                    x=x_bg,
                    y=y_bg,
                    mode='markers',
                    name=bg_label if bg_label is not None else 'Background data',
                    showlegend=(i == 0) and (bg_label is not None),
                    marker=dict(
                        color=hex_to_rgba_str(color_palette["bg_gray"], alpha=0.3),
                        size=6,
                        symbol='circle',
                    ),
                    legendgroup="background",
                ), 
                row=i + 1, col=1
            )
                
        # Perfect fit line
        fig.add_trace(
            go.Scatter(
                x=x,
                y=x,
                mode='lines',
                name='Perfect fit',
                showlegend=(i == 0),
                line=dict(color=color_palette["dark_gray"], width=2),
            ), 
            row=i + 1, col=1
        )
        
        
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
