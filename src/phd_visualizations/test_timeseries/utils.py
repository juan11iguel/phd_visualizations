from typing import Optional
from pathlib import Path
import pandas as pd
import copy

def update_plot_config(
    plot_config: dict, 
    show_titles: bool, 
    show_left_axis_titles: bool,
    show_right_axis_titles: bool, 
    showlegends: bool,
    show_left_axis: bool = True,
    show_right_axis: bool = True,
    show_main_title: bool = False,
    width: Optional[int] = None,
) -> dict:
    pc = copy.deepcopy(plot_config)
    
    if not show_main_title:
        pc["title"] = ""
        pc["subtitle"] = ""
        pc["margin"]["t"] = 30  # reduce top margin if no title
        
    if width is not None:
        pc["width"] = width
    for plot in pc["plots"].values():
        if not show_titles:
            plot["title"] = ""
        if not show_left_axis_titles:
            plot["ylabels_left"] = ["" for _ in plot["ylabels_left"]]
        if "ylabels_right" in plot and not show_right_axis_titles:
                plot["ylabels_right"] = ["" for _ in plot["ylabels_right"]]
        
        plot["show_left_axis"] = show_left_axis
        plot["show_right_axis"] = show_right_axis
        
        if not showlegends:
            plot["showlegend"] = False
            for trace in plot.get("traces_left", []):
                trace["showlegend"] = False
            for trace in plot.get("traces_right", []):
                if isinstance(trace, list):
                    for t in trace:
                        t["showlegend"] = False
                else:
                    trace["showlegend"] = False
                    
        if not show_right_axis_titles:
            if not show_right_axis:
                pc["xdomain"] = [0, 1]
            else:
                pc["xdomain"] = [0, 0.95]
    return pc


def set_common_ylims(plot_config: dict, dfs_: list[pd.DataFrame]) -> dict:

    plot_config = copy.deepcopy(plot_config)

    for plt in plot_config["plots"].values():
        
        # Set common ylims across all datasets
        for side in ["left", "right"]:
            first_element = plt.get(f"traces_{side}", [None])
            if first_element is None or not first_element:
                continue
            first_element = first_element[0]
            
            if isinstance(first_element, dict):
                # Just one axis with multiple traces
                var_ids = [trace["var_id"] for trace in plt[f"traces_{side}"]]
                max_y = max([df_[var_ids].max().max() for df_ in dfs_ if not df_[var_ids].empty])
                min_y = min([df_[var_ids].min().min() for df_ in dfs_ if not df_[var_ids].empty])
                plt[f"ylims_{side}"] = [0.9 * min_y, max_y * 1.1]
            elif isinstance(first_element, list):
                # Multiple axes with multiple traces
                max_y_list = []
                min_y_list = []
                for axis in plt[f"traces_{side}"]:
                    var_ids = [trace["var_id"] for trace in axis]
                    max_y_list.append( max([df_[var_ids].max().max() for df_ in dfs_ if not df_[var_ids].empty]) )
                    min_y_list.append( min([df_[var_ids].min().min() for df_ in dfs_ if not df_[var_ids].empty]) )
                plt[f"ylims_{side}"] = [[0.9 * min_y, max_y * 1.1] for min_y, max_y in zip(min_y_list, max_y_list)]
                
    return plot_config