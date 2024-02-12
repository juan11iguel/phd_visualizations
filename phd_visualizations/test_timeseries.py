
# TODO: Plotly resampler, resampling does not work whenever the is more than one plot, works with multiple legends,
#  and multiple yaxes, must have something to do with the way axis are configured
# TODO: Uncertainty bounds not showing after last changes, used to work before (see figure from PID2024 article)
# TODO: Take instrument from plot_config -> variables_config
# TODO: Parameter to configure limits either: 'auto', 'manual' (calculated from min max of traces in axes), [min, max]

import pandas as pd
import numpy as np
import datetime
from typing import Literal
import plotly
import plotly.graph_objects as go
from loguru import logger
from .constants import color_palette, default_fontsize, newshape_style, ArrayLike
from .calculations import calculate_uncertainty
from plotly_resampler import FigureResampler

legends_plotly_ids = {}

def add_trace(fig: go.Figure | FigureResampler, trace_conf: dict, df: pd.DataFrame, yaxes_idx: int, xaxes_idx: int,
              resample: bool, show_arrow: bool = False, trace_color: str = None, uncertainty: ArrayLike = None, row_idx: int = None,
              axis_side:Literal['left', 'right'] = 'left', arrow_xrel_pos:int = None, var_config:dict = None,
              df_comp: pd.DataFrame = None) -> go.Figure | FigureResampler:


    """ Add custom trace to plotly figure, it can include:
        - Uncertainty bounds
        - Axis arrow


    """
    var_config = var_config if var_config is not None else {}

    if trace_color is not None:
        color = color_palette[trace_color] if trace_color in color_palette.keys() else trace_color
    else:
        color = None

    legend_id = trace_conf.get('legend_id', None)

    if legend_id is not None:
        legend = legends_plotly_ids.get(legend_id, None) # Should always be found
        
        if legend is None:
            logger.warning(f'legend_id {legend_id} not found in plot_config->legends, using default legend')
            legend = 'legend'
    else:
        legend = None

    showlegend = trace_conf.get('showlegend', False)
    name = trace_conf.get('name', None)
    if name is None:
        name = var_config.get('label_html', None)
    if name is None:
        name = var_config['var_id']

    if showlegend:
        logger.debug(f'legend_id: {legend_id}, legend: {legend} for trace {name}')

    # Add uncertainty
    if uncertainty is not None:
        color_rgb = color_palette[trace_color + '_rgb'] if trace_color + '_rgb' in color_palette.keys() else trace_color

        plotly_resample_kwargs = {'hf_x': df.index, 'hf_y': df[trace_conf['var_id']] - uncertainty} if resample else {}
        fig.add_trace(
            go.Scattergl(
                x=df.index if not resample else None,
                y=df[trace_conf['var_id']] - uncertainty if not resample else None,
                name=f"{name} uncertainty lower bound",
                fill=None, line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                xaxis=f'x{xaxes_idx}', yaxis=f'y{yaxes_idx}',
            ),
            **plotly_resample_kwargs
        )

        plotly_resample_kwargs = {'hf_x': df.index, 'hf_y': df[trace_conf['var_id']] + uncertainty} if resample else {}
        fig.add_trace(
            go.Scattergl(
                x=df.index if not resample else None,
                y=df[trace_conf['var_id']] + uncertainty if not resample else None,
                name=f"{name} uncertainty",
                fill='tonexty', fillcolor=f'rgba({color_rgb}, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                xaxis=f'x{xaxes_idx}', yaxis=f'y{yaxes_idx}',
            ),
            **plotly_resample_kwargs
        )

    # Add trace
    plotly_resample_kwargs = {'hf_x': df.index, 'hf_y': df[trace_conf['var_id']]} if resample else {}
    plotly_resample_kwargs_comp = {'hf_x': df_comp.index, 'hf_y': df_comp[trace_conf['var_id']]} if (resample and df_comp is not None) else {}

    fig.add_trace(
        go.Scattergl(
            x=df.index if not resample else None,
            y=df[trace_conf['var_id']] if not resample else None,
            name=name,
            mode=trace_conf.get('mode', 'lines'),
            line=dict(
                color=color,
                dash=trace_conf.get('dash', None),
                width=trace_conf.get('width', None)
            ),
            showlegend=showlegend,
            legend=legend,
            xaxis=f'x{xaxes_idx}',
            yaxis=f'y{yaxes_idx}',

            # Add customdata to show comparison values if provided
            customdata=df_comp[trace_conf['var_id']] if df_comp is not None else None,
            hovertemplate = "%{y:.2f}" if df_comp is None else "%{y:.2f} (<span style='color:gray'> %{customdata:.2f} </span>)"
        ),
        **plotly_resample_kwargs
    )

    # If comparison dataframe is given, add comparison trace
    if df_comp is not None:
        fig.add_trace(
            go.Scattergl(
                x=df_comp.index if not resample else None,
                y=df_comp[trace_conf['var_id']] if not resample else None,
                name=f"{name} comparison",
                mode='lines',
                line=dict(
                    color='gray',
                    dash='dash',
                    width=1
                ),
                showlegend=False,
                xaxis=f'x{xaxes_idx}',
                yaxis=f'y{yaxes_idx}',
                # Hide tooltip
                hoverinfo='skip'
            ),
            **plotly_resample_kwargs_comp
        )

    logger.info(
        f'Trace {name} added in yaxis{yaxes_idx} ({axis_side}), row {row_idx + 1}, uncertainty: '
        f'{True if uncertainty is not None else False}, comparison: {True if df_comp is not None else False}'
    )

    # If right axis traces and specified in trace configuration, add arrow to indicate axis
    # If specified in configuration, maybe the right axis should not be a requirement?
    if show_arrow:
        # Get the trace
        trace = fig.data[-1]

        tr_idx = 0 if axis_side == 'left' else -1
        xshift = -1 if axis_side == 'left' else 1
        ax = 25 if axis_side == 'left' else -25

        fig.add_annotation(
            x=trace.x[tr_idx] + xshift * datetime.timedelta(seconds=arrow_xrel_pos),
            y=trace.y[tr_idx] * trace_conf.get('arrow_yrel_pos', 1.05),
            arrowhead=2, arrowsize=1.5, arrowwidth=1,  # Arrow line width
            arrowcolor=trace.line.color,  # Arrow color
            ax=ax, ay=0, xref=f'x{xaxes_idx}', yref=f'y{xaxes_idx}', showarrow=True,
        )

    return fig

def experimental_results_plot(plt_config: dict, df: pd.DataFrame, df_opt: pd.DataFrame | None = None,
                              df_comp: pd.DataFrame | None = None, title_text: str = None,
                              resample: bool = True, vars_config: dict = None) -> go.Figure:

    """ Generate plotly figure with experimental results

    # TODO: If df_comp is given, duplicate each trace from plt_config with a gray color and a dashed line,
    # and tune the tooltip to show the original and the comparison value

    """

    if resample:
        fig = FigureResampler(go.Figure(), )
    else:
        fig = go.Figure()

    row_heights = []
    n_yaxis = []
    subplot_titles = []
    for plot_props in plt_config['plots'].values():
        row_heights.append(plot_props["row_height"])
        # plot_bg_colors.append( plot_props.get("plot_bg_colors", "steelblue") )
        subplot_titles.append(plot_props.get("title", ""))

    # n_yaxis          = [2, 2, 3, 1,   1,  1,  1, 1,   1, 1,   1, 1,   1]
    # plot_bg_colors   = ["steelblue" for _ in range(2)]
    # plot_bg_colors.append("#e5a50a")
    # plot_bg_colors.extend(["#B6B6B6" for _ in range(10)])
    # yaxis_labels     = [("T<sub>amb</sub> (ºC)", "ɸ (%)"), ("T<sub>v</sub> (ºC)", "P<sub>th</sub> (kW<sub>th</sub>)"),
    #                     ("C (u.m.)", "C<sub>e</sub> (kWh)", "C<sub>w</sub> (l/h)"), "cv ()", ""]

    # additional_space = [0, 0, 0, 0,   0,  0,  0, 0,   0, 0,   0, 0,   0] # Number of additional vertical_spacing to leave
    # vertical_spacing = 0.03
    vertical_spacing = plt_config["vertical_spacing"]
    xdomain = plt_config["xdomain"]
    height = plt_config["height"]
    width = plt_config["width"]
    yaxis_right_pos = [.86, .95]
    arrow_xrel_pos = plt_config.get("arrow_xrel_pos", 20)
    default_active_color = {'active': color_palette['plotly_green_rgb'], 'inactive': color_palette['gray_rgb']}

    # Configure plot legends
    plot_ids_set = set(plt_config['plots'].keys())
    legend_ids_set = set(plt_config.get('legends', {}).keys())
    common_ids = plot_ids_set & legend_ids_set
    
    plots_legend_axes = {}

    if not plt_config.get('show_optimization_updates', False):
        logger.info('Optimization updates not shown in plot, show_optimization_updates: false')

    global legend_plotly_ids

    # First global legends
    leg_idx = 1
    for id in legend_ids_set - common_ids:
        legends_plotly_ids[id] = f'legend{leg_idx if leg_idx > 1 else ""}'
        leg_idx += 1
    
    # And then individual plot legends
    for id in common_ids:
        legends_plotly_ids[id] = f'legend{leg_idx if leg_idx > 1 else ""}'
        leg_idx += 1
        
    rows = len(row_heights)

    cum_sum = float(sum(row_heights))
    heights = []
    for idx, h in enumerate(row_heights):
        height_ = (1.0 - vertical_spacing * (rows - 1)) * (h / cum_sum)
        heights.append(round(height_, 3))

    # print(heights)
    # print(sum(heights))

    domains = [];
    y2 = 0 - vertical_spacing
    for row_idx in reversed(range(rows)):
        y1 = round(y2 + vertical_spacing, 3)
        y2 = round(y1 + heights[row_idx], 3)
        domains.append((y1, y2))

    domains[-1] = (domains[-1][0], round(domains[-1][-1]))

    # display(domains)
    # display( [(tup[1] - tup[0]) for tup in domains] )

    # See plotly start_cell parameter
    domains.reverse()

    xaxes_settings = {}
    yaxes_settings = {}
    # yaxes_settings['yaxis'] = {'domain': domains(1)}

    # Prepare vertical highlights
    vert_values = None
    if 'vertical_highlights' in plt_config.keys():
        # If vertical highlights are specified in plot configuration
        vert_conf = plt_config['vertical_highlights']
        if vert_conf is not None:
            highlight_type = vert_conf.get('type', "normal_trace")
            if highlight_type == "normal_trace":
                if 'var_id' in vert_conf.keys():
                    if vert_conf['var_id'] in df.columns:
                        vert_values = df[vert_conf['var_id']]
                    else:
                        raise ValueError(
                            f'var_id `{vert_conf["var_id"]}` not found in dataframe')
                else:
                    raise ValueError('var_id must be specified in plot configuration if vertical_highlights are enabled')

                ### Color
                if 'color' in vert_conf.keys():
                    if vert_conf['color'] in color_palette.keys():
                        vert_color = color_palette[vert_conf['color']]
                    else:
                        logger.warning(f'Color {conf["active_color"]} not found in color_palette, using default color')
                        vert_color = color_palette['plotly_yellow']

                else:
                    vert_color = color_palette['plotly_yellow']

                ### Calculate times when the system changes state
                change_times_vert = vert_values.index[vert_values.diff() != 0]
                change_times_vert = change_times_vert.insert(len(change_times_vert), vert_values.index[-1])

                ### Make change_times index of vert_values
                vert_values = vert_values.reindex(change_times_vert, method='ffill')
            else:
                raise ValueError(f'highlight_type {highlight_type} not supported')


    shapes = []
    idx = 1
    for row_idx, plot_id in zip(range(rows), plt_config['plots'].keys()):
        conf = plt_config['plots'][plot_id]
        # traces = copy.deepcopy(traces_test)

        axes_idx = idx if idx > 1 else ""
        xaxes_settings[f'xaxis{axes_idx}'] = dict(anchor=f'y{axes_idx}', matches='x' if idx > 1 else None,
                                                  showticklabels=True if row_idx == rows - 1 else False,
                                                  tickcolor="rgba(0,0,0,0)" if row_idx != rows - 1 else None)  # title= idx,
        title = conf.get('ylabels_left', [None])[0]  # Only one axis is supported anyway

        if conf.get('tigth_vertical_spacing', None):
            domain = (domains[row_idx + 1][-1] + vertical_spacing / 3,
                      # Fill the space between the current and the next axis with some vertical_spacing
                      domains[row_idx][-1])  # Not changed
        else:
            domain = domains[row_idx]

        # If axis limits specified manually, autorange must be set to False, otherwise it overrides the manual limits
        yaxes_settings[f'yaxis{axes_idx}'] = {"domain": domain, 'anchor': f'x{axes_idx}', 'title': title, 'showgrid': True,
                                              "autorange": False}

        # Plot configuration
        ## Associate legend  to axes
        if plot_id in common_ids:
            plots_legend_axes[plot_id] = {'axes_idx': axes_idx, 'domain': domain}

        ## Add background color
        bg_color = conf.get("bg_color", None)
        bg_color = color_palette[bg_color] if bg_color in color_palette.keys() else bg_color
        shapes.append(
            dict(
                type="rect", xref=f"x{axes_idx} domain", yref=f"y{axes_idx} domain", opacity=0.1, layer="below",
                line_width=0,
                fillcolor=bg_color,
                x0=-0.01, x1=1.01,
                y0=-0.01, y1=1.01,
            ),
        )

        traces_right = conf.get('traces_right', [])
        overlaying_axis = f'y{idx if idx > 1 else ""}'  # Overlaying axis used to configure right axes

        ## Add decision variables updates
        if plt_config.get('show_optimization_updates', False):
            for index, row in df_opt.iterrows():
                shapes.append(
                    dict(
                        type="rect", xref=f"x{axes_idx}", yref=f"y{axes_idx} domain", opacity=0.4, line_width=0,
                        # layer="below",
                        fillcolor="#deddda",
                        x0=index - datetime.timedelta(seconds=row["computation_time"]), x1=index,
                        y0=-0.01, y1=1.01,
                    ),
                )

        ## Add vertical highlights
        if vert_values is not None and conf.get('show_vertical_highlights', True):
            # If vertical highlights are specified in plot configuration, and not explicitly disabled in plot configuration

            ### Add traces for every state change
            for i_act in range(1, len(vert_values)):
                value = vert_values.iloc[i_act - 1]  # The value of the current span is the previous one, until the change
                span = [vert_values.index[i_act - 1], vert_values.index[i_act]]

                if value:
                    logger.debug(f'Adding vertical highlight for {span[0]} - {span[1]}')

                    shapes.append(
                        dict(
                            type="rect", xref=f"x{axes_idx}", yref=f"y{axes_idx} domain", opacity=0.1,
                            layer="below",
                            line_width=0,
                            fillcolor=vert_color,
                            x0=span[0], x1=span[1],
                            y0=-0.01, y1=1.01,
                        ),
                    )

        ## Active state plot
        if conf.get('show_active', False):
            if 'active_var_id' in conf.keys():
                if conf['active_var_id'] in df.columns:
                    active = df[conf['active_var_id']]
                else:
                    raise ValueError(f'active_var_id `{conf["active_var_id"]}` not found in dataframe for plot {plot_id}')
            else:
                raise ValueError('active_var_id must be specified in plot configuration if show_active is True')

            if 'active_color' in conf.keys():
                if conf['active_color'] in color_palette.keys():
                    color = color_palette[conf['active_color']]
                else:
                    logger.warning(f'Color {conf["active_color"]} not found in color_palette, using default color')
                    color = default_active_color['active']

                active_color = {'active': color, 'inactive': default_active_color['inactive']}
            else:
                active_color = default_active_color

            ### Shift axes idx to +100 to avoid overlapping with other axes
            aux_idx = axes_idx + 100

            ### Calculate times when the system changes state
            change_times = active.index[active.diff() != 0]
            change_times = change_times.insert(len(change_times), active.index[-1])

            ### Make change_times index of active
            active = active.reindex(change_times, method='ffill')

            ### Configure axis so that it's plotted between the current axes and the next one
            yaxes_settings[f'yaxis{aux_idx}'] = {'domain': (domains[row_idx + 1][-1], domains[row_idx][0] - vertical_spacing / 1.5),
                                                 'anchor': f'x{aux_idx}', 'showgrid': False, 'showticklabels': False,
                                                 'showline': False, 'zeroline': False, 'showspikes': False,
                                                 'fixedrange': True, 'tickcolor': "rgba(0,0,0,0)"}
            xaxes_settings[f'xaxis{aux_idx}'] = {'anchor': f'y{aux_idx}', 'matches': 'x', 'showticklabels': False,
                                                 'showgrid': False, 'showline': False, 'zeroline': False,
                                                 'showspikes': False, 'tickcolor': "rgba(0,0,0,0)"}

            ### Add traces for every state change
            for i_act in range(1, len(active)):
                value = active.iloc[i_act-1] # The value of the current span is the previous one, until the change
                span = [active.index[i_act - 1], active.index[i_act]]

                height_active = .8 if value else 0.3
                color = active_color['active'] if value else active_color['inactive']

                # print(value, span, color)

                trace_active = \
                    go.Scatter(
                        x=span, y=[height_active, height_active],
                        name=conf['active_var_id'],
                        showlegend=False, fill='tozeroy', mode='none', fillcolor=f'rgba({color}, 0.6)',
                        xaxis=f'x{aux_idx}', yaxis=f'y{aux_idx}'
                    )

                fig.add_trace(trace_active)

        # Add left traces
        min_y = 9999; max_y = -9999
        for trace_conf in conf['traces_left']:

            if trace_conf['var_id'].endswith('*'):
                # Group of variables
                group = trace_conf['var_id'][:-1]
                group_vars = [var for var in df.columns if var.startswith(group)]

                for group_var in group_vars:
                    trace_conf['var_id'] = group_var
                    var_config = vars_config.get(group_var, {'var_id': group_var})

                    # Axis range
                    min_y = np.min([min_y, df[group_var].min()])
                    max_y = np.max([max_y, df[group_var].max()])

                    fig = add_trace(
                        fig=fig, trace_conf=trace_conf, df=df, yaxes_idx=idx, xaxes_idx=axes_idx,
                        resample=resample,
                        axis_side='left',
                        row_idx=row_idx,
                        var_config=var_config,
                        df_comp=df_comp
                    )

            else:
                trace_color = trace_conf.get("color", None)
                uncertainty = calculate_uncertainty(df[trace_conf['var_id']], trace_conf['instrument']) if trace_conf.get(
                    "instrument", None) else None
                show_arrow = len(traces_right) > 0 and trace_conf.get('axis_arrow', False)
                var_config = vars_config.get(trace_conf['var_id'], None) if vars_config is not None else None

                # Axis range
                min_y = np.min([min_y, df[trace_conf['var_id']].min()])
                max_y = np.max([max_y, df[trace_conf['var_id']].max()])
            
            fig = add_trace(fig=fig, trace_conf=trace_conf, df=df, yaxes_idx=idx, xaxes_idx=axes_idx, resample=resample,
                            show_arrow=show_arrow, trace_color=trace_color, uncertainty=uncertainty, axis_side='left',
                            row_idx=row_idx, arrow_xrel_pos=arrow_xrel_pos, var_config=var_config, df_comp=df_comp)

        # Manually set range for left axis, for some reason it does not work correctly automatically
        if conf.get('ylims_left', None) == 'manual':
            padding = (max_y - min_y) * 0.1 # Creo que lo hice mejor en webscada, revisar
            yaxes_settings[f'yaxis{axes_idx}']['range'] = [min_y - padding, max_y + padding]

        # logger.debug(f'Left axis range: {yaxes_settings[f"yaxis{axes_idx}"]["range"]}')

        idx += 1

        # Traces right
        if len(traces_right) > 0:
            if isinstance(traces_right[0], dict):
                axis_right_configs = [traces_right]  # Single right yaxis
            else:
                axis_right_configs = traces_right  # Multiple right yaxis

            for pos_idx, traces_config in enumerate(axis_right_configs):
                titles = conf.get('ylabels_right', [None] * len(traces_config))

                yaxes_settings[f'yaxis{idx}'] = dict(overlaying=overlaying_axis, side='right', showgrid=False,
                                                     anchor='free', position=yaxis_right_pos[pos_idx],
                                                     title=titles[pos_idx])

                for trace_idx, trace_conf in enumerate(traces_config):
                    # Add trace
                    trace_color = trace_conf.get("color", None)
                    # color = color_palette[trace_color] if trace_color in color_palette.keys() else trace_color
                    uncertainty = calculate_uncertainty(df[trace_conf['var_id']],
                                                        trace_conf['instrument']) if trace_conf.get("instrument",
                                                                                                    None) else None
                    show_arrow = trace_conf.get('axis_arrow', False)
                    var_config = vars_config.get(trace_conf['var_id'], None) if vars_config is not None else None

                    logger.debug(f'Adding trace {trace_conf["var_id"]} to right axis {idx}')
                    fig = add_trace(fig=fig, trace_conf=trace_conf, df=df, yaxes_idx=idx, xaxes_idx=axes_idx, resample=resample,
                                    show_arrow=show_arrow, trace_color=trace_color, uncertainty=uncertainty, axis_side='right',
                                    row_idx=row_idx, arrow_xrel_pos=arrow_xrel_pos, var_config=var_config, df_comp=df_comp)

                # Add index for each right axis added
                idx += 1


    # Legends
    legends_layout = {}

    # Global legends
    for id in legend_ids_set - common_ids:
        conf = plt_config['legends'][id]

        color = color_palette[conf['bgcolor']] if conf['bgcolor'] in color_palette.keys() else conf['bgcolor']
        color = f"rgba({color},0.1)"

        legends_layout[ legends_plotly_ids[id] ] = dict(
            xref='paper', yref='paper',
            x=conf["x"], y=conf["y"],

            title=f"<b>{conf['title']}</b>" if 'title' in conf.keys() else None,
            bgcolor=color,
            font=dict(size=default_fontsize)
        )

    # Individual plot legends
    for id in common_ids:
        conf = plt_config['legends'][id]

        color = color_palette[conf['bgcolor']] if conf['bgcolor'] in color_palette.keys() else conf['bgcolor']
        color = f"rgba({color},0.1)"

        legends_layout[ legends_plotly_ids[id] ] = dict(
            # Can't use axis domain because it's not available for legends
            xref='paper', yref='paper',
            xanchor='right', # yanchor='top',
            x=0.9, y=plots_legend_axes[id]['domain'][-1] + vertical_spacing,
            orientation='h',

            title=f"<b>{conf['title']}</b>" if 'title' in conf.keys() else None,
            bgcolor=color,
            font=dict(size=10)
        )

    if title_text is None:
        # Get from plot configuration
        title_text = f"<b>{plt_config.get('title', None)}</b><br>{plt_config.get('subtitle', None)}</br>"

    fig.update_layout(
        title_text=title_text,
        title_x=0.05,  # Title position, 0 is left, 1 is right
        height=height,
        width=width,
        # plot_bgcolor='#ffffff',
        plot_bgcolor='rgba(0,0,0,0)',
        # paper_bgcolor='#ffffff',
        # paper_bgcolor='rgba(0,0,0,0)',
        # title_text="Complex Plotly Figure Layout",
        # margin=dict(l=20, r=200, t=100, b=20, pad=5),
        margin=plt_config.get("margin", dict(l=20, r=200, t=100, b=20, pad=5)),
        **xaxes_settings,
        **yaxes_settings,
        **legends_layout,
        shapes=shapes,
        newshape=newshape_style,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            # font_family="Rockwell"
        ),
    )
    fig.update_xaxes(domain=xdomain)

    # Add subplot titles
    axes_domains = []
    for ydomain in domains:
        axes_domains.append(xdomain)
        axes_domains.append(ydomain)

    # Better to left center them
    plot_title_annotations = plotly._subplots._build_subplot_title_annotations(
        subplot_titles, axes_domains
    )
    fig.layout.annotations = fig.layout.annotations + tuple(plot_title_annotations)

    return fig
