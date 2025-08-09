import plotly.graph_objects as go
import numpy as np
from typing import Iterable, Optional
from phd_visualizations.constants import color_palette
from dataclasses import dataclass, field

@dataclass
class MarkerParams:
    ring_colors: tuple[str, str] = (color_palette["dc_green"], color_palette["wct_purple"])
    fill_var_colors: tuple[str, str] = ("#ffffff", color_palette["plotly_yellow"])
    marker_size_range: tuple[float, float] = (0.01, 0.05)  # Normalized minimum and maximum size of the donut hole (relative to y-axis range)
    ring_width: float = 1.2  # Width of the donut ring

@dataclass
class SuperMarker:
    size_var_id: str
    size_var_range: tuple[float, float]
    ring_vars_ids: Optional[tuple[str, str]] = None
    fill_var_id: Optional[str] = None
    fill_var_range: Optional[tuple[float, float]] = None 
    ring_labels: Optional[tuple[str, str]] = None
    fill_label: Optional[str] = None
    size_label: Optional[str] = None
    show_markers_index: bool = False
    params: MarkerParams = field(default_factory=MarkerParams)

def lerp_color(c1, c2, t):
    # Linear interpolation between the two colors
    c1 = np.array([int(c1[i:i+2], 16) for i in (1, 3, 5)])
    c2 = np.array([int(c2[i:i+2], 16) for i in (1, 3, 5)])
    c = (1 - t) * c1 + t * c2
    return f"#{int(c[0]):02X}{int(c[1]):02X}{int(c[2]):02X}"

def add_super_scatter_trace(
    fig: go.Figure, 
    x: Iterable,
    y: Iterable,
    trace_label: str,
    size_var_vals: Iterable,
    size_var_range: tuple[float, float],
    ring_vars_vals: Optional[tuple[Iterable, Iterable]] = None,
    fill_var_vals: Optional[Iterable] = None,
    fill_var_range: Optional[tuple[float, float]] = None,
    fill_label: Optional[str] = None,
    size_label: Optional[str] = None,
    ring_labels: Optional[tuple[str, str]] = None,
    row: int = 1,
    col: int = 1,
    xref: str = "",
    yref: str = "",
    showlegend: bool = True,
    show_markers_index: bool = False,
    
    # Parameters
    marker_params: MarkerParams = MarkerParams(),
):
    """    
    Add a "scatter" trace (actually will be made up with shapes) that has 
    three different ways of showing information:
    - (required) The size of the marker is determined by the size_var_vals
    - (optional) The color of the marker is determined by the fill_var_vals
    - (optional) The marker is split into two segments, each representing a different ring_var_vals
    """
    mp = marker_params
    
    num_points = 100 # Number of points to generate for the arc path
    
    # Calculate y-axis range to normalize marker sizes
    y_range = np.nanmax(y) - np.nanmin(y)
    
    # Convert normalized marker_size_range to actual radius values based on y-axis range
    actual_marker_size_range = (mp.marker_size_range[0] * y_range, mp.marker_size_range[1] * y_range)
    
    max_radius = actual_marker_size_range[1] + mp.ring_width * actual_marker_size_range[0]
    
    # print(f"Adding super scatter trace with {len(x)} points, size_var_range: {size_var_range}, fill_var_range: {fill_var_range}. Size label: {size_label}, Fill label: {fill_label}, Ring labels: {ring_labels}. {max_radius=}, {mp.ring_width=}, {mp.marker_size_range=}, y_range: {y_range}, actual_marker_size_range: {actual_marker_size_range}")

    # In the caller function, if not explicitly set, just take the column name in the dataframe

    legend_added_flag = False
    added_idxs = []
    for idx in range(len(x)):
        
        if np.isnan(x[idx]) or np.isnan(y[idx]):
            added_idxs.append(None)
            continue
        
        added_idxs.append(idx)
        
        cx, cy = x[idx], y[idx]  # Center

        r_inner = actual_marker_size_range[0] + (size_var_vals[idx] - size_var_range[0]) / (size_var_range[1] - size_var_range[0]) * (actual_marker_size_range[1] - actual_marker_size_range[0])

        # If no ring_vars_vals are provided, skip the donut creation
        if ring_vars_vals is not None:
            # Map Qc[0] from [Qc_min, Qc_max] to [min_size, max_size] for r_inner
            # r_outer is 10% of marker_size_range[0] larger than r_inner
            r_outer = r_inner + mp.ring_width * actual_marker_size_range[0]

            # Calculate the arc span for the first ring varibale
            theta_start = 0    # Starting angle of partial arc (in degrees)
            theta_end = ring_vars_vals[0][idx] / np.sum([ring_var_val[idx] for ring_var_val in ring_vars_vals]) * 360    # Ending angle of partial arc

            # Generate arc path (SVG path syntax)
            theta = np.linspace(theta_start, theta_end, num_points)
            x_arc = cx + r_outer * np.cos(np.radians(theta))
            y_arc = cy + r_outer * np.sin(np.radians(theta))
            path = f"M{x_arc[0]},{y_arc[0]}" + ''.join([f"L{x},{y}" for x, y in zip(x_arc[1:], y_arc[1:])])
            path += f"L{cx},{cy}Z"  # Close the path back to the center

            # Add full outer circle i.e. second variable! (base)
            fig.add_shape(
                layer="between",
                type="circle",
                x0=cx - r_outer, y0=cy - r_outer,
                x1=cx + r_outer, y1=cy + r_outer,
                fillcolor=mp.ring_colors[1],
                # layer="between",
                line=dict(width=0),  # Disable the line
                showlegend=True if (not legend_added_flag and showlegend) else False,  # Show legend only for the first donut
                name=ring_labels[1] if not legend_added_flag else "",
                xref=f"x{xref}",
                yref=f"y{yref}",
            )

            # Add the calculated partial arc segment corresponding to the first ring variable  (on top of base)
            fig.add_shape(
                layer="between",
                type="path",
                path=path,
                fillcolor=mp.ring_colors[0],
                line=dict(width=0),  # Disable the line
                showlegend=True if (not legend_added_flag and showlegend) else False,  # Show legend only for the first donut
                name=ring_labels[0] if not legend_added_flag else "",
                xref=f"x{xref}",
                yref=f"y{yref}",
            )

        if fill_var_vals is not None:
            # Add inner filled circle (donut hole)
            # Interpolate color from cool_green to cool_red based on Tamb
            # Normalize Tamb to [0, 1]
            t_norm = (fill_var_vals[idx] - fill_var_range[0]) / (fill_var_range[1] - fill_var_range[0])
            fillcolor = lerp_color(mp.fill_var_colors[0], mp.fill_var_colors[1], t_norm)
        else:
            fillcolor = mp.fill_var_colors[-1]

        fig.add_shape(
            layer="between",
            type="circle",
            x0=cx - r_inner, y0=cy - r_inner,
            x1=cx + r_inner, y1=cy + r_inner,
            fillcolor=fillcolor,
            opacity=1,
            line_color="rgba(0, 0, 0, 0)",  # Transparent
            xref=f"x{xref}",
            yref=f"y{yref}",
        )

        # Add inner filled circle (donut hole)
        fig.add_shape(
            layer="below",
            type="circle",
            x0=cx - max_radius, y0=cy - max_radius,
            x1=cx + max_radius, y1=cy + max_radius,
            fillcolor="rgba(0, 0, 0, 0)",  # Transparent
            line_color=color_palette["bg_gray"],
            line=dict(dash='dot'),
            showlegend=True if (not legend_added_flag and showlegend) else False,  # Show legend only for the first donut
            name=size_label if not legend_added_flag else "",
            xref=f"x{xref}",
            yref=f"y{yref}",
        )
        
        legend_added_flag = True  # Only add legend for the first donut
                
    # Add an additional scatter trace to retain the hover info and precise value-position
    
    marker_text_data = {}
    if show_markers_index:
        marker_text_data = dict(
            text=added_idxs,
            textposition="bottom right",
            textfont=dict(size=9,),
        ) 

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name=trace_label,
            mode="text+markers" if show_markers_index else "markers",
            marker=dict(
                size=5,  # Small size to avoid clutter
                color="#333333",
                # symbol="circle",
                # opacity=1,
            ),
            showlegend=False,
            **marker_text_data
            
        ),
        row=row, col=col
    ) 

    # Add an additional (out of the range) weather circle for the legend
    fig.add_shape(
        type="circle",
        x0=-9999,
        x1=-9999,
        fillcolor=mp.fill_var_colors[1],
        opacity=1,
        line_color="rgba(0, 0, 0, 0)",  # Transparent
        showlegend=showlegend,
        name=fill_label,
        xref=f"x{xref}",
        yref=f"y{yref}",
    )
    
    return fig