import copy

# TODO: Implement reset_colors_per_plot, so that plotly starts from the same colors for each plot
# TODO: Plotly resampler, resampling does not work whenever the is more than one plot, works with multiple legends,
#  and multiple yaxes, must have something to do with the way axis are configured
# TODO: Uncertainty bounds not showing after last changes, used to work before (see figure from PID2024 article)
# TODO: Take instrument from plot_config -> variables_config

import pandas as pd
import numpy as np
import datetime
from typing import Literal
import re
import plotly
import plotly.graph_objects as go
from loguru import logger
from phd_visualizations.constants import color_palette, default_fontsize, newshape_style, ArrayLike, named_css_colors
from phd_visualizations.calculations import calculate_uncertainty
from phd_visualizations.utils import tuple_to_string, ColorChooser, Operators, hex_to_rgba_str
from plotly.colors import hex_to_rgb, qualitative

logger.disable(__name__)

legends_dict = {"global": {}, "plots": {}}
color_chooser = ColorChooser([
    color_palette['plotly_green'],
    color_palette['plotly_red']
])


def add_trace(
    fig: go.Figure, 
    trace_conf: dict, 
    df: pd.DataFrame, 
    yaxes_idx: int, 
    xaxes_idx: int,
    resample: bool, 
    show_arrow: bool = False, 
    trace_color: str = None, 
    uncertainty: ArrayLike = None,
    row_idx: int = None,
    axis_side: Literal['left', 'right'] = 'left', 
    arrow_xrel_pos: int = None, 
    var_config: dict = None,
    df_comp: list[pd.DataFrame] = None,
    index_adaptation_policy: Literal['adapt_to_ref', 'combine'] = 'adapt_to_ref',
    legend_id: str = None,
    legend_yaxis_indicator: str = None,
    **kwargs
) -> go.Figure:
    """ Add custom trace to plotly figure, it can include:
        - Uncertainty bounds
        - Axis arrow


    """

    var_id = trace_conf["var_id"]
    logger.debug(f'Attempting to add {var_id}')
    var_config = var_config if var_config is not None else {}

    if trace_color is not None:
        color = color_palette[trace_color] if trace_color in color_palette.keys() else trace_color
    else:
        color = None

    showlegend = trace_conf.get('showlegend', False)
    legend = None
    if showlegend:
        if legend_id in trace_conf:
            # Global legend
            legend_id = trace_conf['legend_id']
            assert legend_id in legends_dict["global"], f'legend_id {legend_id} not found in plot_config->legends. If a legen_id is speficied, a global legend with that id should exist in the configuration'
            
            legend = legends_dict["global"][legend_id]["plotly_id"]  # Should always be found

        else:
            # Plot legend
            legend = legends_dict["plots"][legend_id]["plotly_id"]  # Should always be found

    name = trace_conf.get('name', None)
    if name is None:
        name = var_config.get('label_html', None)
    if name is None:
        name = var_config.get('var_id', None)
    if name is None:
        name = trace_conf.get('var_id', None)
    if legend_yaxis_indicator is not None:
        name = f"{name} {legend_yaxis_indicator}"

    # if name is None:
    #     raise KeyError(f'No name for variable {} could be found in any of the available options')

    if showlegend:
        logger.debug(f'legend_id: {legend_id}, legend: {legend} for trace {name}')

    # Conditional plot
    active_signal = None
    if 'conditional' in trace_conf:
        c = trace_conf['conditional']
        try:
            active_signal = df.apply(lambda row: Operators[c['operator']](row[c['var_id']], c['threshold_value']),
                                     axis="columns")

            df.loc[~active_signal, var_id] = np.nan

            logger.debug(
                f'Conditional trace {var_id}: Not showing {np.sum(~active_signal)} out of {len(df)} points since condition {c["var_id"]} {c["operator"]} {c["threshold_value"]} was not met')
        except KeyError:
            raise KeyError(
                f"Conditional plot set up for {var_id}, but one of the required arguments is missing: 'operator', 'threshold_value', 'var_id'")

    # Add uncertainty
    if uncertainty is not None:
        plotly_resample_kwargs = {'hf_x': df.index, 'hf_y': df[var_id] - uncertainty} if resample else {}
        fig.add_trace(
            go.Scattergl(
                x=df.index if not resample else None,
                y=df[var_id] - uncertainty if not resample else None,
                name=f"{name} uncertainty lower bound (invisible, required by plotly)",
                mode='lines',
                fill=None, line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                xaxis=f'x{xaxes_idx}', yaxis=f'y{yaxes_idx}',
            ),
            **plotly_resample_kwargs
        )

        plotly_resample_kwargs = {'hf_x': df.index, 'hf_y': df[var_id] + uncertainty} if resample else {}
        fig.add_trace(
            go.Scattergl(
                x=df.index if not resample else None,
                y=df[trace_conf['var_id']] + uncertainty if not resample else None,
                name=f"{name} uncertainty",
                mode='lines',
                fill='tonexty', fillcolor=hex_to_rgba_str(color, alpha=0.3),
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                xaxis=f'x{xaxes_idx}', yaxis=f'y{yaxes_idx}',
            ),
            **plotly_resample_kwargs
        )

    # Add comparison trace(s)
    customdata = None
    if df_comp is not None:
        N_comp = len(df_comp)
        # default = np.ones(df[trace_conf['var_id']].shape)*np.nan
        # The comparison data provided is a list of dataframes, which might or might not include data for the specified variable
        customdata = []
        for i, df_comp_ in enumerate(df_comp):
            data_comp = None
            if var_id in df_comp_.columns:

                # Propagate the conditionals to the comparison data
                if active_signal is not None:
                    # Re-evaluate for the comp trace
                    try:
                        active_signal = df_comp_.apply(
                            lambda row: Operators[c['operator']](row[c['var_id']], c['threshold_value']),
                            axis="columns")
                        df_comp_.loc[~active_signal, var_id] = np.nan
                    except KeyError:
                        logger.warning(
                            f"Conditional plot set up for {var_id}, but one of the required signals is missing from the comparison dataframe(s): {df_comp_['var_id']}"
                        )

                data_comp = df_comp_[var_id]

                # If index of df_comp[i] is not the same as df, fill with NaNs
                if not data_comp.index.equals(df.index):

                    if index_adaptation_policy == 'adapt_to_ref':
                        data_comp = data_comp.reindex(df.index, fill_value=np.nan)
                        logger.warning(
                            f'Index of comparison dataframe {i} is not the same as the main dataframe, adapting comparison data to main dataframe index')

                    elif index_adaptation_policy == 'combine':
                        combined_index = df.index.union(data_comp.index)
                        df = df.reindex(combined_index, fill_value=np.nan)
                        data_comp = data_comp.reindex(combined_index, fill_value=np.nan)

            customdata.append(data_comp)
        # customdata = [df_comp_.get(trace_conf['var_id'], None) for df_comp_ in df_comp]

    hovertemplate = "%{y:.2f}"
    if customdata is not None and not all(element is None for element in customdata):
        customdata = np.stack(customdata, axis=-1)

        # hovertemplate = "%{y:.2f} (<span style='color:gray'> %{customdata:.2f} </span>)"
        hovertemplate = "%{y:.2f} "
        for i in range(customdata.shape[1]):
            comp = customdata[:, i]
            if comp is not None:

                plotly_resample_kwargs_comp = {'hf_x': df.index, 'hf_y': comp} if resample else {}

                hovertemplate += f"(<span style='color:{named_css_colors[i]}'> %{{customdata[{i}]:.2f}} </span>) "

                if (i > 0 and color is None) or N_comp > 1:
                    color_comp = named_css_colors[i]
                else:
                    color_comp = color
                    
                stackgroup_comp = kwargs.get('stackgroup', None)
                if stackgroup_comp is not None:
                    stackgroup_comp = f"{stackgroup_comp}_comp_{i}"

                # if trace_conf['var_id'] in df_comp[i].columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index if not resample else None,
                        y=comp if not resample else None,
                        name=f"{name} comparison {i}",
                        mode='lines',
                        line=dict(
                            color=color_comp,
                            dash='dot',
                            width=2
                        ),
                        showlegend=False,
                        xaxis=f'x{xaxes_idx}',
                        yaxis=f'y{yaxes_idx}',
                        # Hide tooltip
                        hoverinfo='skip',
                        # **trace_conf.get("kwargs", {}),
                        stackgroup=stackgroup_comp,
                        fillcolor="rgba(0,0,0,0)",
                    ),
                    **plotly_resample_kwargs_comp
                )
            else:
                logger.debug(
                    f'Cant add comparison trace for {trace_conf["var_id"]} since it does not exist in df_comp[{i}]')        

    # Configure fill_between
    if trace_conf.get('fill_between', False):
        fill_between = True
        compared_var_id = trace_conf['fill_between']
        try:
            comp_name = f"{var_id}_gt_{compared_var_id}"
            df[comp_name] = df.apply(
                # Should be > but for some reason it works with <, probably because of the way the traces are added
                lambda row: row[var_id] if row[var_id] < row[compared_var_id] else row[compared_var_id], axis="columns"
            )
        except KeyError:
            raise KeyError(
                f'Attempted to add a trace with a fill between variable {compared_var_id}, but it could not be found in the dataframe')
    else:
        fill_between = False

    # Add trace
    plotly_resample_kwargs = {'hf_x': df.index, 'hf_y': df[trace_conf['var_id']]} if resample else {}

    # Set trace color with opacity if specified
    if color is not None and 'opacity' in trace_conf:
        if not color.startswith("rgba"):
            color = hex_to_rgba_str(color, alpha=trace_conf['opacity'])

    width = trace_conf.get('width', 2)
    if df_comp is not None:
        width = width * 1.25

    fig.add_trace(
        go.Scatter(
            x=df.index if not resample else None,
            y=df[trace_conf['var_id']] if not resample else None,
            name=name,
            mode=trace_conf.get('mode', 'lines'),
            line=dict(
                color=color,
                dash=trace_conf.get('dash', None),
                width=width
            ),
            fill=trace_conf.get('fill', None) if not fill_between else None,  # fillcolor=f'rgba({color_rgb}, 0.1)',
            fillpattern=dict(shape=trace_conf.get('fill_pattern', None)),
            stackgroup=kwargs.get('stackgroup', None),
            showlegend=showlegend,
            legend=legend,
            xaxis=f'x{xaxes_idx}',
            yaxis=f'y{yaxes_idx}',
            **trace_conf.get("kwargs", {}),

            # Add customdata to show comparison values if provided
            customdata=customdata,
            hovertemplate=hovertemplate
        ),
        **plotly_resample_kwargs
    )

    # Add additional trace for fill_between
    if fill_between:
        if color is not None:
            # color_rgb should've been generated together with color if a color was specified in the trace config
            color_fill = hex_to_rgba_str(color, alpha=0.3)
        else:
            logger.warning(
                f'No color specified for fill_between trace {var_id}, choosing randomly between predefined options: {color_chooser.color_options}')
            color_fill = hex_to_rgba_str(color_chooser.choose(), alpha=0.3)

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[comp_name],
                mode='lines',
                name=comp_name,
                line=dict(color=color, width=0),
                connectgaps=False,
                fill='tonexty',
                fillcolor=color_fill,
                showlegend=False,
                legend=legend,
                xaxis=f'x{xaxes_idx}',
                yaxis=f'y{yaxes_idx}',
                hoverinfo='skip'
            )
        )

        # https://stackoverflow.com/a/75094341/13853313
        # We need to use an invisible trace so we can reset "next y" for the negative area indicator
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[var_id],
                line_color="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=f'x{xaxes_idx}',
                yaxis=f'y{yaxes_idx}',
                hoverinfo='skip'
            )
        )

    # If comparison dataframe is given, add comparison trace
    # if customdata is not None:
    #     for i, comp in enumerate(customdata):
    #         if comp is not None:
    #     else:
    #         logger.debug(f'Cant add comparison trace for {trace_conf["var_id"]} since it does not exist in df_comp')

    logger.info(
        f'Trace {name} added in yaxis{yaxes_idx} ({axis_side}), row {row_idx + 1}, uncertainty: '
        f'{True if uncertainty is not None else False}, comparison: {True if df_comp is not None else False}'
    )

    # If right axis traces and specified in trace configuration, add arrow to indicate axis
    # If specified in configuration, maybe the right axis should not be a requirement?
    if show_arrow:
        # Get the trace
        trace = fig.data[-1]
        ax = 25 if axis_side == 'left' else -25

        fig.add_annotation(
            x=0.1 if axis_side == 'left' else 0.9,
            # x=trace.x[tr_idx] + xshift * datetime.timedelta(seconds=arrow_xrel_pos),
            y=trace_conf.get('arrow_yrel_pos', 0.1),  # y=trace.y[tr_idx] * trace_conf.get('arrow_yrel_pos', 1.05),
            arrowhead=2, arrowsize=1.5, arrowwidth=1,  # Arrow line width
            arrowcolor=trace.line.color,  # Arrow color
            ax=ax, ay=0, xref=f'x{xaxes_idx} domain', yref=f'y{xaxes_idx} domain', showarrow=True,
        )

    return fig


def experimental_results_plot(
    plt_config: dict, 
    df: pd.DataFrame, 
    df_opt: pd.DataFrame | None = None, # TODO: Should be removed
    df_comp: pd.DataFrame | list[pd.DataFrame] = None, 
    title_text: str = None,
    resample: bool = True, 
    vars_config: dict = None,
    index_adaptation_policy: Literal['adapt_to_ref', 'combine'] = 'adapt_to_ref',
    reset_colors_per_plot: bool = False,
    legend_yaxis_indicator_symbols: tuple[str, str] = ("❮", "❯"),
) -> go.Figure:
    
    """ Generate plotly figure with experimental results
    """

    global legends_dict
    
    if reset_colors_per_plot:
        raise NotImplementedError('reset_colors_per_plot not implemented yet')

    if resample:
        from plotly_resampler import FigureWidgetResampler

    if resample:
        fig = FigureWidgetResampler(go.Figure(), )
    else:
        fig = go.Figure()

    if df_comp is not None and not isinstance(df_comp, list):
        df_comp = [df_comp]

    # Create a copy of the input dataframe to avoid modifying the original
    df = df.copy()

    # n_yaxis          = [2, 2, 3, 1,   1,  1,  1, 1,   1, 1,   1, 1,   1]
    # plot_bg_colors   = ["steelblue" for _ in range(2)]
    # plot_bg_colors.append("#e5a50a")
    # plot_bg_colors.extend(["#B6B6B6" for _ in range(10)])
    # yaxis_labels     = [("T<sub>amb</sub> (ºC)", "ɸ (%)"), ("T<sub>v</sub> (ºC)", "P<sub>th</sub> (kW<sub>th</sub>)"),
    #                     ("C (u.m.)", "C<sub>e</sub> (kWh)", "C<sub>w</sub> (l/h)"), "cv ()", ""]

    # additional_space = [0, 0, 0, 0,   0,  0,  0, 0,   0, 0,   0, 0,   0] # Number of additional vertical_spacing to leave
    # vertical_spacing = 0.03
    vertical_spacing = plt_config.get("vertical_spacing", 0.03)
    # reduced_vs = vertical_spacing / 3
    xdomain = plt_config.get("xdomain", [0, 0.85])
    height = plt_config["height"]
    width = plt_config["width"]
    yaxis_right_pos = [.86, .95]
    arrow_xrel_pos = plt_config.get("arrow_xrel_pos", 20)
    default_active_color = {'active': color_palette['plotly_green'], 'inactive': color_palette['gray']}
    # tigth_vertical_spacing = [plot_props.get('tigth_vertical_spacing', False) for plot_props in plt_config['plots'].values()]

    n_plots = len(plt_config['plots'])

    # Every time the vertical spacing is reduced, that reduction is plot area that is asigned equally for all the plots
    # gained_height_reduced_vs = np.sum(tigth_vertical_spacing) * (vertical_spacing - reduced_vs) / n_plots

    row_heights = []
    n_yaxis = []
    subplot_titles = []
    for plot_props in plt_config['plots'].values():
        # plot_bg_colors.append( plot_props.get("plot_bg_colors", "steelblue") )
        subplot_titles.append(plot_props.get("title", ""))
        row_heights.append(plot_props.get("row_height", 1))  # + gained_height_reduced_vs)

    
    # TODO: Remove optimization updates
    if not plt_config.get('show_optimization_updates', False):
        logger.info('Optimization updates not shown in plot, show_optimization_updates: false')
        
    # Configure plot ydomain
    rows = len(row_heights)

    total_row_heights = float(sum(row_heights))
    heights = []
    for idx, h in enumerate(row_heights):
        # vs = reduced_vs if tigth_vertical_spacing[idx] else vertical_spacing
        vs = vertical_spacing  # Left unfinished, revisit in another moment
        height_ = (1.0 - vs * (rows - 1)) * (h / total_row_heights)
        heights.append(round(height_, 3))

    # print(heights)
    # print(sum(heights))

    domains = []
    y2 = 0 - vertical_spacing
    for row_idx in reversed(range(rows)):
        # vs = reduced_vs if tigth_vertical_spacing[row_idx] else vertical_spacing # Left unfinished, revisit in another moment
        vs = vertical_spacing

        y1 = round(y2 + vs, 3)
        y2 = round(y1 + heights[row_idx], 3)
        domains.append((y1, y2))

    domains[-1] = (domains[-1][0], round(domains[-1][-1]))

    # display(domains)
    # display( [(tup[1] - tup[0]) for tup in domains] )

    # See plotly start_cell parameter
    domains.reverse()
    
    # Configure legends
    plot_ids: list[str] = plt_config['plots'].keys()
    plot_ids_set = set(plot_ids)
    legend_ids_global_set = set(plt_config.get('legends', {}).keys())
    common_ids = plot_ids_set & legend_ids_global_set
    legend_ids_plots: list[str] = [plt_id for plt_id, conf in plt_config['plots'].items() if conf.get("showlegend", False)]
    if len(common_ids) > 0:
        raise ValueError(f'Global legend ids need to be different from plot ids: {common_ids}')

    # Setup plotly legend ids
    # First global legends
    leg_idx = 1
    for id in legend_ids_global_set:
        logger.debug(f'Configuring global legend {id}')
        legends_dict["global"][id] = {
            "plotly_id": f'legend{leg_idx if leg_idx > 1 else ""}',
            # Plotly limitation to use paper:
            # https://plotly.com/python/legend/
            "xref": 'paper', "yref": 'paper',
            "bgcolor": plt_config["legends"].get("bgcolor_rgba", "rgba(0,0,0,0)"),
            "orientation": plt_config["legends"].get("orientation", "v"),
            "title": plt_config["legends"].get("title", None),
            "font": plt_config["legends"].get("font", {"size": default_fontsize}),
            "x": plt_config["legends"].get("x", 0.5),
            "y": plt_config["legends"].get("y", 0.0),
        }
        leg_idx += 1
    # And then individual plot legends
    # req_field_ids = ["legend_position", ]
    row_idx = -1
    for plot_id in plot_ids:
        row_idx += 1
        
        if plot_id not in legend_ids_plots:
            continue
        
        logger.debug(f'Configuring plot legend for {plot_id}')
        # Validate
        # for req_field in req_field_ids:
        #     assert req_field in plt_config['plots'][id].keys(), f'{req_field} not found in plot configuration for plot {id}'
        legend_pos = plt_config['plots'][plot_id].get("legend_position", "side")
        legend_xmargin = plt_config['plots'][plot_id].get("legend_xmargin", 0.05)
        legend_delta_y = plt_config['plots'][plot_id].get("legend_delta_y", plt_config.get("legend_delta_y", 0.0))
        if legend_pos == "side":
            y = domains[row_idx][1] + legend_delta_y
        else:
            y = domains[row_idx][1] + .6* vertical_spacing + legend_delta_y
            
        legends_dict["plots"][plot_id] = {
            "plotly_id": f'legend{leg_idx if leg_idx > 1 else ""}',
            # Plotly limitation to use paper:
            # https://plotly.com/python/legend/
            "xref": 'paper', "yref": 'paper',
            "bgcolor": "rgba(0,0,0,0)",
            # "orientation": "h",
            "orientation": "v" if legend_pos == "side" else "h",
            "yanchor": "top",
            "font": dict(
                size=plt_config['plots'][plot_id].get("legend_fontsize", plt_config.get("legend_fontsize", 11))
            ),
            "xanchor": "left" if legend_pos == "side" else "right", # "yanchor": "top",
            "x": xdomain[1] + legend_xmargin if legend_pos == "side" else xdomain[1],
            "y": y
        }
        leg_idx += 1
        
        # print(f"{plot_id}: {row_idx=}, {domains[row_idx]=}, {legends_dict['plots'][plot_id]['y']=}")

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
            if highlight_type != "normal_trace":
                raise ValueError(f'highlight_type {highlight_type} not supported')
            if 'var_id' not in vert_conf.keys():
                raise ValueError('var_id must be specified in plot configuration if vertical_highlights are enabled')
            if vert_conf['var_id'] not in df.columns:
                raise ValueError(f'var_id `{vert_conf["var_id"]}` not found in dataframe')
            vert_values = df[vert_conf['var_id']]

            ### Color
            if 'color' not in vert_conf.keys():
                vert_color = color_palette['plotly_yellow']
            else:
                if vert_conf['color'] in color_palette.keys():
                    vert_color = color_palette[vert_conf['color']]
                else:
                    logger.warning(f'Color {vert_conf["active_color"]} not found in color_palette, using default color')
                    vert_color = color_palette['plotly_yellow']

            ### Calculate times when the system changes state
            change_times_vert = vert_values.index[vert_values.diff() != 0]
            change_times_vert = change_times_vert.insert(len(change_times_vert), vert_values.index[-1])

            ### Make change_times index of vert_values
            vert_values = vert_values.reindex(change_times_vert, method='ffill')

    shapes = []
    idx = 1
    for row_idx, plot_id in zip(range(rows), plt_config['plots'].keys()):
        conf = plt_config['plots'][plot_id]
        # traces = copy.deepcopy(traces_test)

        axes_idx = idx if idx > 1 else ""
        xaxes_settings[f'xaxis{axes_idx}'] = dict(
            anchor=f'y{axes_idx}', 
            matches='x' if idx > 1 and not conf.get("independent_xaxis", False) else None,
            showticklabels=True if row_idx == rows - 1 or conf.get("independent_xaxis", False) else False,
            tickcolor="rgba(0,0,0,0)" if row_idx != rows - 1 else None,
            title_text=conf.get("xaxis_title_text", None),
            title_standoff=conf.get("xaxis_title_standoff", None),
            minor={"showgrid": plt_config.get("xminor", False)},
        )  # title= idx,
        # print(f"{xaxes_settings=}")
        
        title = conf.get('ylabels_left', [None])[0]  # Only one axis is supported anyway

        if conf.get('tigth_vertical_spacing', None):
            domain = (domains[row_idx + 1][-1] + vertical_spacing / 3,
                      # Fill the space between the current and the next axis with some vertical_spacing
                      domains[row_idx][-1])  # Not changed
        else:
            domain = domains[row_idx]

        # If axis limits specified manually, autorange must be set to False, otherwise it overrides the manual limits
        yaxes_settings[f'yaxis{axes_idx}'] = {
            "domain": domain, 
            'anchor': f'x{axes_idx}', 
            'title': title,
            'showgrid': True,
            "autorange": False
        }

        # Plot configuration

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
                value = vert_values.iloc[
                    i_act - 1]  # The value of the current span is the previous one, until the change
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

        ## Add horizontal area
        if 'horizontal_area' in conf.keys():
            area_conf = conf['horizontal_area']
            area_values = area_conf['values']

            ### Get color
            default_color = "cool_green"
            if 'color' not in area_conf.keys():
                shape_color = color_palette[default_color]
                logger.info(
                    f'Color not specified for horizontal area, using default color {default_color}')

            else:
                if area_conf['color'] in color_palette.keys():
                    shape_color = color_palette[area_conf['color']]
                else:
                    logger.warning(
                        f'Color {area_conf["active_color"]} not found in color_palette, using default color {default_color}')
                    shape_color = color_palette[default_color]

            ### Add the shape
            # TODO: For some reason, adding this shape makes kaleido not able
            # to export the figure to an image:
            # ValueError: Transform failed with error code 525: Cannot read property 'append' of undefined

            shapes.append(
                dict(
                    type="rect", xref=f"x{axes_idx} domain", yref=f"y{axes_idx}", opacity=0.1,
                    layer="between",
                    line_width=2,
                    fillcolor=shape_color,
                    x0=-0.01, x1=1.01,
                    y0=area_values[1], y1=area_values[0],
                ),
            )

        ## Active state plot
        if conf.get('show_active', False):
            if 'active_var_id' in conf.keys():
                if conf['active_var_id'] in df.columns:
                    active = df[conf['active_var_id']]
                else:
                    raise ValueError(
                        f'active_var_id `{conf["active_var_id"]}` not found in dataframe for plot {plot_id}')
            else:
                raise ValueError('active_var_id must be specified in plot configuration if show_active is True')

            if 'active_color' in conf.keys():
                color = color_palette[conf['active_color']] if conf['active_color'] in color_palette.keys() else conf[
                    'active_color']

                active_color = {'active': color, 'inactive': default_active_color['inactive']}
            else:
                active_color = default_active_color

            ### Shift axes idx to +100 to avoid overlapping with other axes
            aux_idx = (axes_idx if isinstance(axes_idx, int) else 0) + 100

            ### Calculate times when the system changes state
            change_times = active.index[active.diff() != 0]
            change_times = change_times.insert(len(change_times), active.index[-1])

            ### Make change_times index of active
            active = active.reindex(change_times, method='ffill')

            ### Configure axis so that it's plotted between the current axes and the next one
            yaxes_settings[f'yaxis{aux_idx}'] = {
                'domain': (domains[row_idx + 1][-1], domains[row_idx][0] - vertical_spacing / 1.5),
                'anchor': f'x{aux_idx}', 'showgrid': False, 'showticklabels': False,
                'showline': False, 'zeroline': False, 'showspikes': False,
                'fixedrange': True, 'tickcolor': "rgba(0,0,0,0)"
            }
            xaxes_settings[f'xaxis{aux_idx}'] = {
                'anchor': f'y{aux_idx}', 'matches': 'x', 'showticklabels': False,
                'showgrid': False, 'showline': False, 'zeroline': False,
                'showspikes': False, 'tickcolor': "rgba(0,0,0,0)"
            }

            ### Add traces for every state change
            for i_act in range(1, len(active)):
                value = active.iloc[i_act - 1]  # The value of the current span is the previous one, until the change
                span = [active.index[i_act - 1], active.index[i_act]]

                height_active = .8 if value else 0.3
                color = active_color['active'] if value else active_color['inactive']

                # print(value, span, color)

                trace_active = \
                    go.Scatter(
                        x=span, y=[height_active, height_active],
                        name=conf['active_var_id'],
                        showlegend=False, fill='tozeroy', mode='none', fillcolor=hex_to_rgba_str(color, 0.6),
                        xaxis=f'x{aux_idx}', yaxis=f'y{aux_idx}'
                    )

                fig.add_trace(trace_active)

        # Add left traces
        min_y = 9999
        max_y_ = 0
        max_y = 0
        
        # If no traces are specified, add a placeholder trace to avoid empty plots
        if 'traces_left' not in conf.keys() or len(conf.get('traces_left', [])) == 0:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=np.full((df.index.shape), np.nan),
                    name=plot_id,
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                    legend=legends_dict["plots"].get("plotly_id", {"plotly_id": None})["plotly_id"],
                    xaxis=f'x{axes_idx}', yaxis=f'y{axes_idx}',
                )
            )
        
        for trace_conf in conf['traces_left']:
            
            ylims_left = conf.get('ylims_left', None)

            if '*' in trace_conf['var_id']:
                # Validation
                assert vars_config is not None, f'vars_config must be provided if a group of variables is specified (wildcard selector): {trace_conf["var_id"]}'

                # Group of variables
                pattern = re.escape(trace_conf["var_id"]).replace(r'\*', '.*')
                group_vars = [var for var in df.columns if re.match(pattern, var)]
                group = trace_conf["var_id"][:-1].replace('*', '')
                # group = trace_conf["var_id"][:-1]
                # group_vars = [var for var in df.columns if var.startswith(group)]

                logger.debug(f'Found a group of variables to be plot {trace_conf["var_id"]}: {group_vars}')

                trace_conf_copy = copy.deepcopy(trace_conf)
                vars_config_copy = copy.deepcopy(vars_config)
                for color_idx, group_var in enumerate(group_vars):
                    trace_conf_copy['var_id'] = group_var

                    # Ensure color_idx cycles through the length of qualitative.Plotly
                    color_idx = color_idx % len(qualitative.Plotly)

                    if 'color' in trace_conf_copy:
                        logger.warning(f'Color for group of variables is not supported, using default color palette')

                    trace_color = hex_to_rgba_str(qualitative.Plotly[color_idx],
                                                  alpha=trace_conf_copy.get('opacity', 1))

                    var_config = vars_config_copy.get(group_var, {'var_id': group_var})

                    # Axis range
                    min_y = np.nanmin([min_y, df[trace_conf['var_id']].min()])
                    max_y_ = np.nanmax([max_y_, df[trace_conf['var_id']].max()])
                    # If variables are stacked, the range is calculated based on the sum of the variables
                    max_y = max_y_ if stackgroup is None else max_y + df[trace_conf['var_id']].max()

                    fig = add_trace(
                        fig=fig, trace_conf=trace_conf_copy, df=df, yaxes_idx=idx, xaxes_idx=axes_idx,
                        resample=resample,
                        axis_side='left',
                        row_idx=row_idx,
                        var_config=var_config,
                        trace_color=trace_color,  # To assign a different color to each trace
                        df_comp=df_comp,
                        stackgroup=group if 'fill' in trace_conf else None,
                        index_adaptation_policy=index_adaptation_policy,
                        legend_id=plot_id,
                    )

            else:
                trace_color = trace_conf.get("color", None)
                uncertainty = calculate_uncertainty(df[trace_conf['var_id']],
                                                    trace_conf['instrument']) if trace_conf.get("instrument", None) else None
                show_arrow = len(traces_right) > 0 and trace_conf.get('axis_arrow', False)
                var_config = vars_config.get(trace_conf['var_id'], None) if vars_config is not None else None
                stackgroup = trace_conf.get("stackgroup", None)
                # Show yaxis indicator in legend if more than one left-yaxis is used
                legend_yaxis_indicator = legend_yaxis_indicator_symbols[0] if len(traces_right) > 0 else None
                
                # Axis range
                min_y = np.nanmin([min_y, df[trace_conf['var_id']].min()])
                max_y_ = np.nanmax([max_y_, df[trace_conf['var_id']].max()])
                # If variables are stacked, the range is calculated based on the sum of the variables
                max_y = max_y_ if stackgroup is None else max_y + df[trace_conf['var_id']].max()

                fig = add_trace(
                    fig=fig, 
                    trace_conf=trace_conf, 
                    df=df, 
                    yaxes_idx=idx, 
                    xaxes_idx=axes_idx,
                    resample=resample,
                    show_arrow=show_arrow, 
                    trace_color=trace_color, 
                    uncertainty=uncertainty,
                    axis_side='left',
                    stackgroup=stackgroup,
                    row_idx=row_idx, 
                    arrow_xrel_pos=arrow_xrel_pos, 
                    var_config=var_config, 
                    df_comp=df_comp,
                    index_adaptation_policy=index_adaptation_policy,
                    legend_id=plot_id,
                    legend_yaxis_indicator=legend_yaxis_indicator
                )

        # Manually set range for left axis, for some reason it does not work correctly automatically
        padding = (max_y - min_y) * 0.1  # Creo que lo hice mejor en webscada, revisar
        if ylims_left in ['manual', "manual_from_zero"]:
            padding_min = padding
            padding_max = padding
            if ylims_left == "manual_from_zero":
                min_y = 0
                padding_min = 0
            yaxes_settings[f'yaxis{axes_idx}']['range'] = [min_y - padding_min, max_y + padding_max]

        elif isinstance(ylims_left, list):
            yaxes_settings[f'yaxis{axes_idx}']['range'] = [conf['ylims_left'][0], conf['ylims_left'][1]]

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

                    if trace_conf['var_id'].endswith('*'):
                        raise NotImplementedError(
                            f'Currently, groups of variables are not supported for the right yaxis')

                    # Add trace
                    trace_color = trace_conf.get("color", None)
                    # color = color_palette[trace_color] if trace_color in color_palette.keys() else trace_color
                    uncertainty = calculate_uncertainty(df[trace_conf['var_id']],
                                                        trace_conf['instrument']) if trace_conf.get("instrument",
                                                                                                    None) else None
                    show_arrow = trace_conf.get('axis_arrow', False)
                    var_config = vars_config.get(trace_conf['var_id'], None) if vars_config is not None else None
                    # Show yaxis indicator in legend if more than one left-yaxis is used
                    legend_yaxis_indicator = "".join([legend_yaxis_indicator_symbols[1]] * (pos_idx+1))


                    logger.debug(f'Adding trace {trace_conf["var_id"]} to right yaxis {idx}')
                    fig = add_trace(
                        fig=fig, 
                        trace_conf=trace_conf, 
                        df=df, 
                        yaxes_idx=idx, 
                        xaxes_idx=axes_idx,
                        resample=resample,
                        show_arrow=show_arrow, 
                        trace_color=trace_color, 
                        uncertainty=uncertainty,
                        axis_side='right',
                        row_idx=row_idx, 
                        arrow_xrel_pos=arrow_xrel_pos, 
                        var_config=var_config,
                        df_comp=df_comp,
                        index_adaptation_policy=index_adaptation_policy,
                        legend_id=plot_id,
                        legend_yaxis_indicator=legend_yaxis_indicator
                    )

                # if isinstance(conf.get('ylims_right', None), list):
                #     yaxes_settings[f'yaxis{idx}']['range'] = [conf['ylims_right'][0], conf['ylims_right'][1]]

                # Add index for each right axis added
                idx += 1


    # Build legends objects
    legends_layout = {}
    keys_to_skip = ["plotly_id", ]
    for lg_values in [*legends_dict["global"].values(), *legends_dict["plots"].values()]:
        legends_layout[lg_values["plotly_id"]] = {
            name: value for name, value in lg_values.items() if name not in keys_to_skip
        }

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
            bgcolor="rgba(0,0,0,0)",
            font_size=12,
            bordercolor="rgba(0,0,0,0)"
            # font_family="Rockwell"
        ),
    )
    fig.update_xaxes(domain=xdomain)

    # Add subplot titles
    axes_domains = []
    for ydomain in domains:
        axes_domains.append((xdomain[0], xdomain[0]+0.15)) # To left align the titles
        # axes_domains.append(xdomain) # To center align the titles
        axes_domains.append(ydomain)
        
    # print(f"{domains=}")

    plot_title_annotations = plotly._subplots._build_subplot_title_annotations(
        subplot_titles, axes_domains,#title_edge="right"
    )
    # Left-align titles and check for delta_y
    [plt_anot.update({"xanchor": "left", 
                      "x": plt_config.get("subplot_titles_x", 0.0),
                      "y": plt_anot["y"]+plt_conf.get("title_delta_y", 0.0)}) for plt_anot, plt_conf in zip(plot_title_annotations, plt_config["plots"].values())]
    fig.layout.annotations = fig.layout.annotations + tuple(plot_title_annotations)

    return fig
