import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import Output, VBox
import numpy as np
from pprint import pprint
import plotly.express as px
from plotly_resampler import FigureWidgetResampler
import pandas as pd

plotly_colors = px.colors.qualitative.Plotly


def interactive_selection_plot(df: pd.DataFrame, plot_config: dict, pt_sizes: tuple[int, int] = (10, 20)) -> tuple[
    VBox, dict[str, dict[str, np.ndarray]]]:
    # fig = timeseries_simple(df, plot_config=plot_config)
    assert sum(subplot_config.get('selectable', False) for subplot_config in
               plot_config["plots"].values()) == 1, "Exactly one `selectable` subplot is allowed"

    fig1 = make_subplots(
        rows=len(plot_config["plots"]) - 1,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[plot["title"] for plot in plot_config["plots"].values() if not plot.get('selectable', False)],
        # vertical_spacing=plot_config["vertical_spacing"]
    )
    fig1.layout.hovermode = 'x unified'
    fig1 = FigureWidgetResampler(fig1)
    fig1.update_layout(
        title=plot_config["title"],
        height=plot_config.get("height", None),
        width=plot_config.get("width", None),
        margin=plot_config.get("margin", None),
        showlegend=False,
    )
    # Add traces to each subplot
    row = 1
    for plot in plot_config["plots"].values():
        if plot.get('selectable', False):
            continue
        for trace in plot["traces_left"]:
            fig1.add_trace(
                go.Scattergl(
                    mode=trace.get("mode", "lines"),
                    name=trace["var_id"],
                    line=dict(color=trace.get("color", None), width=trace.get("width", 2)),
                    fill=trace.get("fill", None)
                ),
                hf_x=df.index,
                hf_y=df[trace["var_id"]],
                row=row, col=1
            )
        row += 1

    fig = FigureWidgetResampler(resampled_trace_prefix_suffix=('', ''))  # go.FigureWidget()
    fig.layout.hovermode = 'closest'
    fig.update_layout(
        title='<b>Selectable traces for calibration</b><br>Click on the traces to select points and click again on an already selected point to remove it<br>',
        height=600,
        width=fig1.layout.width,
        xaxis=dict(
            range=[df.index[0], df.index[-1]],
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1min", step="minute", stepmode="backward"),
                    dict(count=10, label="10min", step="minute", stepmode="backward"),
                ]),
            ),
            # rangeslider=dict(visible=True),
        ),
        title_y=0.95
    )
    # Add traces to each subplot
    plot_name = [plot for plot in plot_config["plots"].keys() if plot_config["plots"][plot].get('selectable', False)][0]
    selected_pts: dict[str, dict[str, np.ndarray]] = {}
    for idx, trace in enumerate(plot_config["plots"][plot_name]["traces_left"]):
        fig.add_trace(
            go.Scattergl(
                mode='markers',
                name=trace["var_id"],
                marker=dict(line=dict(width=0), opacity=0.7),
                # marker=dict(color=trace.get("color", plotly_colors[idx]), size=pt_sizes[0]),
            ),
            hf_x=df.index,
            hf_y=df[trace["var_id"]],
        )
        trace_data = fig.data[-1]
        trace_data.marker.color = [plotly_colors[idx]] * len(df)
        trace_data.marker.size = [pt_sizes[0]] * len(df)

        selected_pts.update({
            trace["var_id"]: {
                'default_color': trace_data.marker.color[0],
                'idxs_plot': np.array([], dtype=int),
                'idxs_df': np.array([], dtype=int)
            }
        })

    clickable_traces: list[go.Scatter] = [trace for trace in fig.data]
    # for clickable_trace in clickable_traces:
    #     clickable_trace.marker.color = [clickable_trace.line.color] * len(df)
    #     clickable_trace.marker.size = [pt_sizes[0]] * len(df)
    # pprint(selected_pts)

    # Synx xdomains
    # fig1.layout.xaxis.domain = (0.065, 0.8)
    domain = (0.06, 0.82)
    [fig1.update_layout({f"{xaxis_key}.domain": domain, f"{xaxis_key}.fixedrange": True})
     for xaxis_key in fig1.layout if xaxis_key.startswith('xaxis')]

    # Output
    out = Output()

    # Callback with output decorator
    @out.capture(clear_output=True)
    def update_point(trace, points, selector):
        # print(f"{trace.marker.color=}, \n {selector=}, \n {points=}")
        trace_id = [trace_id for trace_id in selected_pts if trace_id in trace.name][0]
        # pprint(f"Updated selected points: {trace_id}: {selected_pts[trace_id]['idxs']}")

        c = list(trace.marker.color)
        s = list(trace.marker.size)
        shapes = list(fig.layout.shapes)
        shapes1 = list(fig1.layout.shapes)

        for i in points.point_inds:
            # Decide whether to add it, or to remove it if it's already there

            if i in selected_pts[trace_id]['idxs_plot']:
                # Remove point
                # print(c[i], selected_pts[trace_id]['default_color'])
                c[i] = selected_pts[trace_id]['default_color']
                s[i] = pt_sizes[0]

                selector_ = np.where(selected_pts[trace_id]['idxs_plot'] == i)
                selected_pts[trace_id].update({
                    'idxs_plot': np.delete(selected_pts[trace_id]['idxs_plot'], selector_),
                    'idxs_df': np.delete(selected_pts[trace_id]['idxs_df'], selector_)
                })

                # Remove vertical line
                # with fig1.batch_update():
                # for row in range(1, len(plot_config["plots"])):
                #     fig1.layout.shapes = (shape for shape in fig1.layout.shapes if shape.x0 != trace.x[i])
                # Remove vertical line in the main plot
                for shape in shapes:
                    if shape.x0 == trace.x[i]:
                        shapes.remove(shape)
                        break
                # Remove vertical line in the secondary plot
                # Iterate over a copy of shapes1 to avoid modifying the list while iterating
                for shape in shapes1[:]:
                    if shape.x0 == trace.x[i]:
                        shapes1.remove(shape)


            else:
                # Add it
                c[i] = '#bae2be'
                s[i] = pt_sizes[1]

                selected_pts[trace_id].update({
                    'idxs_plot': np.append(selected_pts[trace_id]['idxs_plot'], i),
                    'idxs_df': np.append(selected_pts[trace_id]['idxs_df'], df.index.get_loc(trace.x[i]))
                })

                fig1.add_vline(x=trace.x[i], line_dash="dash", line_color="RoyalBlue", row="all")
                fig.add_vline(x=trace.x[i], line_dash="dash", line_color="RoyalBlue")

                shapes = list(fig.layout.shapes)
                shapes1 = list(fig1.layout.shapes)

        # pprint(f"Updated selected points: {trace.name}: {selected_pts[trace.name]['idxs']}")
        # Plot update
        with fig.batch_update():
            #     print(f"Selected point: {i}")
            trace.marker.color = c
            trace.marker.size = s
            fig.layout.shapes = shapes

        with fig1.batch_update():
            fig1.layout.shapes = shapes1

        pprint([f"{trace_id}: {selected_pts[trace_id]['idxs_df']}" for trace_id in selected_pts])

    def sync_axes(layout, xrange):
        # Sync axes with fig1
        # fig1.layout.xaxis.range = xrange
        [fig1.update_layout({f"{xaxis_key}.range": xrange}) for xaxis_key in fig1.layout if
         xaxis_key.startswith('xaxis')]

    # def sync_hover(trace, points, selector):
    #     print(f"{trace=}, \n {points=}, \n {selector=}")

    #     point_index = points.point_inds[0]

    #     # Trigger hover event on the second plot
    #     plotly.Fx.hover('plot1', {'xval': x[point_index], 'yval': y2[point_index]})

    # print('What')
    # pprint(selected_pts)

    # Attach callback to traces, for some reason is only attaching to the last one
    [trace.on_click(update_point) for trace in clickable_traces]
    [trace.on_selection(update_point) for trace in clickable_traces]
    # fig.data[-1].on_hover(sync_hover)
    fig.layout.on_change(sync_axes, 'xaxis.range')

    return VBox([fig1, fig, out]), selected_pts