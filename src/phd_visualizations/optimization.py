from collections.abc import Iterable
import numpy as np
import plotly.graph_objects as go
import plotly.colors

from phd_visualizations.utils import find_n_best_values_in_list

# Constants
plt_colors = plotly.colors.qualitative.Plotly * 5
gray_colors = plotly.colors.sequential.Greys[2:][::-1]
green_colors = plotly.colors.sequential.Greens[2:][::-1]

        
def plot_obj_scape_comp_1d(fitness_history_list: list[np.ndarray[float]], algo_ids: list[str], highlight_best: int = 0, **kwargs) -> go.Figure:
    
    assert len(fitness_history_list) == len(algo_ids), "fitness_history_list and algo_ids should have the same length"
    
    best_fit_idxs, _ = find_n_best_values_in_list(fitness_history_list, n=highlight_best, objective="minimize")

    # First create the base plot calling plot_obj_space_1d_no_animation
    fig = plot_obj_space_1d_no_animation(fitness_history_list[0], algo_id=algo_ids[0], 
                                         showlegend=True,
                                         line_color=plt_colors[0] if highlight_best == 0 else gray_colors[-1],)
        
    for idx, (algo_id, fitness_history) in enumerate( zip(algo_ids[1:], fitness_history_list[1:]) ):
        avg_fitness = [np.mean(x) for x in fitness_history]
        generation = np.arange(len(fitness_history))
        
        if highlight_best > 0:
            showlegend = False
            line_color = gray_colors[-1]
        else:
            showlegend = True
            line_color = plt_colors[idx+1]
        
        fig.add_trace(go.Scatter(x=generation, y=avg_fitness, mode="lines", name=algo_id.replace("_", " "), showlegend=showlegend, line=dict(color=line_color)))
        
    # Add best traces at the end
    if highlight_best > 0:
        for idx, best_fit_idx in enumerate(best_fit_idxs):
            avg_fitness = [np.mean(x) for x in fitness_history_list[best_fit_idx]]
            generation = np.arange(len(fitness_history_list[best_fit_idx]))
            
            fig.add_trace(go.Scatter(x=generation, y=avg_fitness, mode="lines", name=f"{algo_ids[best_fit_idx].replace('_', ' ')}", line=dict(color=plt_colors[idx], width=3)))
        
    fig.update_layout(**kwargs)
    
    return fig
        
"""
From here is basically copied from EvoX: https://github.com/EMI-Group/evox/blob/main/src/evox/vis_tools/plot.py#L4
Have to find a better way to import this without having to install the whole package nor copying code
"""
        
def plot_obj_space_1d(fitness_history: list[np.ndarray[float]], animation: bool = True, **kwargs) -> go.Figure:
    if animation:
        return plot_obj_space_1d_animation(fitness_history, **kwargs)
    else:
        return plot_obj_space_1d_no_animation(fitness_history, **kwargs)


def plot_obj_space_1d_no_animation(fitness_history: list[np.ndarray[float]], algo_id: str = None, line_color=plt_colors[0],**kwargs) -> go.Figure:

    avg_fitness = [np.mean(x) for x in fitness_history]
    generation = np.arange(len(fitness_history))

    additional_scatters = []
    if isinstance(fitness_history[0], Iterable):
        min_fitness = [np.min(x) for x in fitness_history]
        max_fitness = [np.max(x) for x in fitness_history]
        median_fitness = [np.median(x) for x in fitness_history]
        
        additional_scatters = [
            go.Scatter(x=generation, y=min_fitness, mode="lines", name="Min"),
            go.Scatter(x=generation, y=max_fitness, mode="lines", name="Max"),
            go.Scatter(x=generation, y=median_fitness, mode="lines", name="Median"),
        ]
        
    # Layout defaults
    kwargs.setdefault("yaxis_title", "Fitness")
    kwargs.setdefault("xaxis_title", "Number of objective function evaluations")
    kwargs.setdefault("title_text", "<b>Fitness evolution</b><br>comparison between different algorithms")
    kwargs.setdefault("showlegend", True)
    
        
    fig = go.Figure(
        [
            *additional_scatters,
            go.Scatter(x=generation, y=avg_fitness, mode="lines", name="Average" if algo_id is None else algo_id.replace("_", " "),
                       line=dict(color=line_color)),
        ],
        layout=go.Layout(
            # legend={
            #     "x": 1,
            #     "y": 1,
            #     "xanchor": "auto",
            # },
            # margin={"l": 0, "r": 0, "t": 0, "b": 0},
            **kwargs
        ),
    )

    return fig

def plot_obj_space_1d_animation(fitness_history: list[np.ndarray[float]], **kwargs) -> go.Figure:
    """

    Args:
        fitness_history (list[np.ndarray[float]]): List of fitness values for each individual per generation

    Returns:
        go.Figure: Figure object
        
    Example:
    # This is the last population, after evolution
    # pop = isl.get_population()
    # Properties
    # - best_idx
    # - worst_idx
    # - champion_f
    # - champion_x
    log = isl.get_algorithm().extract(type(algorithm)).get_log()

    # We only have information from the best individual per generation
    fitness_history = [l[2] for l in log]
    
    fig = plot_obj_space_1d_animation(fitness_history=fitness_history, title="Fitness evolution")
    fig

    """

    min_fitness = [np.min(x) for x in fitness_history]
    max_fitness = [np.max(x) for x in fitness_history]
    median_fitness = [np.median(x) for x in fitness_history]
    avg_fitness = [np.mean(x) for x in fitness_history]
    generation = np.arange(len(fitness_history))

    frames = []
    steps = []
    for i in range(len(fitness_history)):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=generation[: i + 1],
                        y=min_fitness[: i + 1],
                        mode="lines",
                        name="Min",
                        showlegend=True,
                    ),
                    go.Scatter(
                        x=generation[: i + 1],
                        y=max_fitness[: i + 1],
                        mode="lines",
                        name="Max",
                    ),
                    go.Scatter(
                        x=generation[: i + 1],
                        y=median_fitness[: i + 1],
                        mode="lines",
                        name="Median",
                    ),
                    go.Scatter(
                        x=generation[: i + 1],
                        y=avg_fitness[: i + 1],
                        mode="lines",
                        name="Average",
                    ),
                ],
                name=str(i),
            )
        )

        step = {
            "label": i,
            "method": "animate",
            "args": [
                [str(i)],
                {
                    "frame": {"duration": 200, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 200},
                },
            ],
        }
        steps.append(step)

    sliders = [
        {
            "currentvalue": {"prefix": "Generation: "},
            "pad": {"b": 1, "t": 10},
            "len": 0.8,
            "x": 0.2,
            "y": 0,
            "yanchor": "top",
            "xanchor": "left",
            "steps": steps,
        }
    ]
    lb = min(min_fitness)
    ub = max(max_fitness)
    fit_range = ub - lb
    lb = lb - 0.05 * fit_range
    ub = ub + 0.05 * fit_range
    fig = go.Figure(
        data=frames[-1].data,
        layout=go.Layout(
            legend={
                "x": 1,
                "y": 1,
                "xanchor": "auto",
            },
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            sliders=sliders,
            xaxis={"range": [0, len(fitness_history)], "autorange": False},
            yaxis={"range": [lb, ub], "autorange": False},
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 200, "redraw": False},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "x": 0.2,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                    "direction": "left",
                    "pad": {"r": 10, "t": 30},
                },
            ],
            **kwargs,
        ),
        frames=frames,
    )

    return fig

def plot_dec_space(population_history, **kwargs,) -> go.Figure:
    """A Built-in plot function for visualizing the population of single-objective algorithm.
    Use plotly internally, so you need to install plotly to use this function.

    If the problem is provided, we will plot the fitness landscape of the problem.
    """

    all_pop = np.concatenate(population_history, axis=0)
    x_lb = np.min(all_pop[:, 0])
    x_ub = np.max(all_pop[:, 0])
    x_range = x_ub - x_lb
    x_lb = x_lb - 0.1 * x_range
    x_ub = x_ub + 0.1 * x_range
    y_lb = np.min(all_pop[:, 1])
    y_ub = np.max(all_pop[:, 1])
    y_range = y_ub - y_lb
    y_lb = y_lb - 0.1 * y_range
    y_ub = y_ub + 0.1 * y_range

    frames = []
    steps = []
    for i, pop in enumerate(population_history):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=pop[:, 0],
                        y=pop[:, 1],
                        mode="markers",
                        marker={"color": "#636EFA"},
                    ),
                ],
                name=str(i),
            )
        )
        step = {
            "label": i,
            "method": "animate",
            "args": [
                [str(i)],
                {
                    "frame": {"duration": 200, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 200},
                },
            ],
        }
        steps.append(step)

    sliders = [
        {
            "currentvalue": {"prefix": "Generation: "},
            "pad": {"b": 1, "t": 10},
            "len": 0.8,
            "x": 0.2,
            "y": 0,
            "yanchor": "top",
            "xanchor": "left",
            "steps": steps,
        }
    ]

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            legend={
                "x": 1,
                "y": 1,
                "xanchor": "auto",
            },
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            sliders=sliders,
            xaxis={"range": [x_lb, x_ub]},
            yaxis={"range": [y_lb, y_ub]},
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 200, "redraw": False},
                                    "fromcurrent": True,
                                    "transition": {
                                        "duration": 200,
                                        "easing": "linear",
                                    },
                                    "mode": "immediate",
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "x": 0.2,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                    "direction": "left",
                    "pad": {"r": 10, "t": 30},
                },
            ],
            **kwargs,
        ),
        frames=frames,
    )

    return fig