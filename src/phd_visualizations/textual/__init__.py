from typing import Any, Optional

def generate_latex_table(
    regular_col_ids: list[str], 
    regular_col_labels: list[str], 
    metric_info: dict[str, str], 
    data: list[dict[str, Any]], 
    submetric_ids: list[str] = [], 
    group_row_ids: Optional[list[str]] = None, 
    submetric_labels: Optional[list[str]] = None,
    group_submetric_ids: Optional[list[str]] = None,
    numbers_precision: int = 2,
) -> str:
    """
    Generate a LaTeX table string with multirow/multicolumn headers for metrics and submetrics.

    Args:
        regular_col_ids: List of column IDs for regular (non-metric) columns
        regular_col_labels: List of LaTeX-formatted labels for regular columns
        metric_info: Dictionary mapping metric IDs to their LaTeX-formatted labels
        data: List of dictionaries containing the table data
        submetric_ids: List of submetric IDs (optional)
        group_row_ids: List of column IDs to group with multirow when consecutive rows have same values (optional)
        submetric_labels: List of LaTeX-formatted labels for submetrics (optional)
        group_submetric_ids: List of submetric IDs to group with multirow when consecutive rows have same values (optional)

    Returns:
        LaTeX table string

    Example usage:

    regular_col_ids = [
        "variable",
        "alternative",
        "test_id",
        "time",
    ]

    regular_col_labels = [
        r"Predicted\\ variable",
        r"Modelling\\ alternative",
        r"Test\\ date",
        r"Evaluation\\ time (s)",
    ]

    group_row_ids = [
        "variable",
        "time",
        "alternative"
    ]

    metric_info = {
        "r2": r"R$^2$\\ (-)",
        "rmse": r"RMSE\\ (s.u.)",
        "mae": r"MAE\\ (s.u.)",
    }

    submetric_ids = ["T", "V"]
    submetric_labels = ["T$_{dc,out}$ ($^\\circ$C)", "V$_{dc,out}$ (V)"]

    data = [
        {
            "variable": "T$_{dc,out}$ ($^\\circ$C)",
            "alternative": "Physical model",
            "time": "0.035",
            "test_id": "YYYYMMDD",
            "metrics": {
                "r2": {"T": "0.98", "V": "0.97"},
                "rmse": {"T": "0.50", "V": "0.52"},
                "mae": {"T": "0.45", "V": "0.48"},
            }
        },
        {
            "variable": "T$_{dc,out}$ ($^\\circ$C)",
            "alternative": "Physical model",
            "time": "0.035",
            "test_id": "YYYYMMDD",
            "metrics": {
                "r2": {"T": "0.98", "V": "0.97"},
                "rmse": {"T": "0.50", "V": "0.52"},
                "mae": {"T": "0.45", "V": "0.48"},
            }
        },
        {
            "variable": "T$_{dc,out}$ ($^\\circ$C)",
            "alternative": "Physical model",
            "time": "0.035",
            "test_id": "YYYYMMDD",
            "metrics": {
                "r2": {"T": "0.98", "V": "0.97"},
                "rmse": {"T": "0.50", "V": "0.52"},
                "mae": {"T": "0.45", "V": "0.48"},
            }
        },
    ]
    
    generate_latex_table(regular_col_ids, metric_info, submetric_ids, data)

    """

    def get_content_col_positions():
        # Helper function to calculate actual column positions with spacing
        positions = []
        col_num = 1
        
        # Regular columns
        for i in range(n_regular):
            positions.append(col_num)
            col_num += 1
            if i < n_regular - 1:  # Skip spacer column
                col_num += 1
        
        # Skip spacer between regular and metrics
        if n_regular > 0 and n_metrics > 0:
            col_num += 1
        
        # Metric columns
        metric_positions = []
        if n_metrics > 0:
            if n_submetrics > 0:
                for m_idx in range(n_metrics):
                    for s_idx in range(n_submetrics):
                        metric_positions.append(col_num)
                        col_num += 1
                        # Skip spacer between submetrics and between metrics
                        if s_idx < n_submetrics - 1 or m_idx < n_metrics - 1:
                            col_num += 1
            else:
                for m_idx in range(n_metrics):
                    metric_positions.append(col_num)
                    col_num += 1
                    if m_idx < n_metrics - 1:  # Skip spacer between metrics
                        col_num += 1
        
        return positions, metric_positions

    def find_row_groups():
        """
        Find groups of consecutive rows that have the same values for each column in group_row_ids independently.
        Returns a dictionary mapping column_id to list of (start_idx, group_size) tuples for each group.
        """
        if not group_row_ids:
            return {}
        
        column_groups = {}
        
        for col_id in group_row_ids:
            groups = []
            i = 0
            while i < len(data):
                # Start a new group for this column
                group_start = i
                group_value = data[i].get(col_id, "")
                
                # Find consecutive rows with same value for this column
                j = i + 1
                while j < len(data):
                    current_value = data[j].get(col_id, "")
                    if current_value != group_value:
                        break
                    j += 1
                
                group_size = j - i
                groups.append((group_start, group_size))
                i = j
            
            column_groups[col_id] = groups
        
        return column_groups

    def find_submetric_groups():
        """
        Find groups of consecutive rows that have the same values for group_submetric_ids submetrics.
        Returns a dictionary mapping (metric_id, submetric_id) to list of (start_idx, group_size) tuples.
        """
        if not group_submetric_ids:
            return {}
        
        submetric_groups = {}
        
        for metric_id in metric_keys:
            for submetric_id in group_submetric_ids:
                if submetric_id in submetric_ids:
                    groups = []
                    i = 0
                    while i < len(data):
                        # Start a new group
                        group_start = i
                        group_value = data[i].get("metrics", {}).get(metric_id, {}).get(submetric_id, "")
                        
                        # Find consecutive rows with same values
                        j = i + 1
                        while j < len(data):
                            current_value = data[j].get("metrics", {}).get(metric_id, {}).get(submetric_id, "")
                            if current_value != group_value:
                                break
                            j += 1
                        
                        group_size = j - i
                        groups.append((group_start, group_size))
                        i = j
                    
                    submetric_groups[(metric_id, submetric_id)] = groups
        
        return submetric_groups

    n_regular = len(regular_col_ids)
    n_metrics = len(metric_info)
    n_submetrics = len(submetric_ids)
    metric_keys = list(metric_info.keys())

    # Build column format string
    col_format = ""
    # Add regular columns with spacing
    for i in range(n_regular):
        col_format += "c"
        if i < n_regular - 1:  # Add spacer column between regular columns
            col_format += "c"

    # Add spacer between regular and metric sections
    if n_regular > 0 and n_metrics > 0:
        col_format += "c"

    # Add metric columns with spacing
    if n_metrics > 0:
        if n_submetrics > 0:
            for m_idx in range(n_metrics):
                for s_idx in range(n_submetrics):
                    col_format += "c"
                    # Add spacer between submetrics within same metric, and between metrics
                    if s_idx < n_submetrics - 1 or m_idx < n_metrics - 1:
                        col_format += "c"
        else:
            for m_idx in range(n_metrics):
                col_format += "c"
                if m_idx < n_metrics - 1:  # Add spacer between metrics
                    col_format += "c"

    # Header rows
    header = []
    # First header row
    row1 = []
    for i, col in enumerate(regular_col_labels):
        row1.append(f"\\multirow{{{2 if n_submetrics==0 else 3}}}{{*}}{{\\textbf{{\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{col}\\end{{tabular}}}}}}")
        if i < n_regular - 1:  # Add empty spacer between regular columns
            row1.append("")

    # Add spacer between regular and metric sections
    if n_regular > 0 and n_metrics > 0:
        row1.append("")

    if n_metrics > 0:
        if n_submetrics > 0:
            # Calculate span: all metric columns + spacing between them
            total_metric_content_cols = n_metrics * n_submetrics
            total_spacing_cols = total_metric_content_cols - 1  # spacing between each metric column
            span = total_metric_content_cols + total_spacing_cols
            row1.append(f"\\multicolumn{{{span}}}{{c}}{{\\textbf{{Performance metric}}}}")
        else:
            # Calculate span: all metrics + spacing between them
            span = n_metrics + (n_metrics - 1)
            row1.append(f"\\multicolumn{{{span}}}{{c}}{{\\textbf{{Performance metric}}}}")

        header.append(" & ".join(row1))
            
        # --- Clines ---
        regular_positions, metric_positions = get_content_col_positions()

        # 1. Cline under "Performance metric" (spans all metric columns)
        clines = []
        if n_metrics > 0:
            perf_start = metric_positions[0]
            perf_end = metric_positions[-1]
            clines.append(f"\\cline{{{perf_start}-{perf_end}}}")
        header.append("\\\\" + "".join(clines))

    # Second header row (metrics)
    if n_metrics > 0:
        row2 = []
        # Add empty cells for regular columns and their spacing
        for i in range(n_regular):
            row2.append("")
            if i < n_regular - 1:  # Add empty spacer between regular columns
                row2.append("")
        
        # Add spacer between regular and metric sections
        if n_regular > 0:
            row2.append("")
        
        # Add metric headers with spacing
        for m_idx, m in enumerate(metric_keys):
            if n_submetrics > 0:
                # Calculate span: submetrics for this metric + spacing between them
                span = n_submetrics + (n_submetrics - 1)
                row2.append(f"\\multicolumn{{{span}}}{{c}}{{\\textbf{{\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{metric_info[m]}\\end{{tabular}}}}}}")
                # Add spacer between metrics (but not after last metric)
                if m_idx < n_metrics - 1:
                    row2.append("")
            else:
                row2.append(f"\\textbf{{\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{metric_info[m]}\\end{{tabular}}}}")
                # Add spacer between metrics (but not after last metric)
                if m_idx < n_metrics - 1:
                    row2.append("")
        
        header.append(" & ".join(row2))

        # 2. Clines under each metric group (if submetrics)
        clines = []
        if n_metrics > 0 and n_submetrics > 0:
            for m_idx in range(n_metrics):
                start_idx = m_idx * n_submetrics
                end_idx = start_idx + n_submetrics - 1
                start = metric_positions[start_idx]
                end = metric_positions[end_idx]
                clines.append(f"\\cline{{{start}-{end}}}")
        header.append(" \\\\" + "".join(clines))

    # Third header row (submetrics)
    if n_submetrics > 0:
        row3 = []
        # Add empty cells for regular columns and their spacing
        for i in range(n_regular):
            row3.append("")
            if i < n_regular - 1:  # Add empty spacer between regular columns
                row3.append("")
        
        # Add spacer between regular and metric sections
        if n_regular > 0:
            row3.append("")
        
        # Add submetric labels with spacing
        for m_idx in range(n_metrics):
            for s_idx, s in enumerate(submetric_labels):
                row3.append(f"{s}")
                # Add spacer between submetrics and between metrics
                if s_idx < n_submetrics - 1 or m_idx < n_metrics - 1:
                    row3.append("")
        
        header.append(" & ".join(row3))

    # 3. Final clines under each actual content column
    clines = []
    # Regular columns
    for pos in regular_positions:
        clines.append(f"\\cline{{{pos}-{pos}}}")
    # Metric columns
    for pos in metric_positions:
        clines.append(f"\\cline{{{pos}-{pos}}}")
    header.append(" \\\\" + "".join(clines))

    # Data rows
    row_groups = find_row_groups()
    submetric_groups = find_submetric_groups()
    
    # Create mappings of row index to group info for quick lookup per column
    column_row_to_group = {}
    for col_id, groups in row_groups.items():
        column_row_to_group[col_id] = {}
        for group_start, group_size in groups:
            for i in range(group_start, group_start + group_size):
                column_row_to_group[col_id][i] = (group_start, group_size)
    
    # Create mappings for submetric groups
    submetric_row_to_group = {}
    for (metric_id, submetric_id), groups in submetric_groups.items():
        submetric_row_to_group[(metric_id, submetric_id)] = {}
        for group_start, group_size in groups:
            for i in range(group_start, group_start + group_size):
                submetric_row_to_group[(metric_id, submetric_id)][i] = (group_start, group_size)
    
    data_rows = []
    for row_idx, row in enumerate(data):
        row_cells = []
        
        # Add regular columns with spacing
        for i, col in enumerate(regular_col_ids):
            
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.{numbers_precision}f}"
            value = str(value)
            
            # Check if this column should be grouped and if this is the first row of a group
            if (group_row_ids and col in group_row_ids and 
                col in column_row_to_group and 
                row_idx in column_row_to_group[col]):
                
                group_start, group_size = column_row_to_group[col][row_idx]
                
                if row_idx == group_start and group_size > 1:
                    # First row of a group - use multirow
                    row_cells.append(f"\\multirow{{{group_size}}}{{*}}{{{value}}}")
                elif row_idx > group_start:
                    # Subsequent rows of a group - empty cell
                    row_cells.append("")
                else:
                    # Single row group - normal cell
                    row_cells.append(value)
            else:
                # Normal cell (not grouped)
                row_cells.append(value)
                
            if i < n_regular - 1:  # Add empty spacer between regular columns
                row_cells.append("")
        
        # Add spacer between regular and metric sections
        if n_regular > 0 and n_metrics > 0:
            row_cells.append("")
        
        # Add metric columns with spacing
        metrics = row.get("metrics", {})
        if n_metrics > 0:
            if n_submetrics > 0:
                for m_idx, m in enumerate(metric_keys):
                    for s_idx, s in enumerate(submetric_ids):
                        
                        value = metrics.get(m, {}).get(s, "")
                        if isinstance(value, float):
                            value = f"{value:.{numbers_precision}f}"
                        value = str(value)
                        
                        # Check if this submetric should be grouped
                        if (group_submetric_ids and s in group_submetric_ids and 
                            (m, s) in submetric_row_to_group and 
                            row_idx in submetric_row_to_group[(m, s)]):
                            
                            group_start, group_size = submetric_row_to_group[(m, s)][row_idx]
                            
                            if row_idx == group_start and group_size > 1:
                                # First row of a group - use multirow
                                row_cells.append(f"\\multirow{{{group_size}}}{{*}}{{{value}}}")
                            elif row_idx > group_start:
                                # Subsequent rows of a group - empty cell
                                row_cells.append("")
                            else:
                                # Single row group - normal cell
                                row_cells.append(value)
                        else:
                            # Normal cell (not grouped)
                            row_cells.append(value)
                            
                        # Add empty spacer between submetrics and between metrics
                        if s_idx < n_submetrics - 1 or m_idx < n_metrics - 1:
                            row_cells.append("")
            else:
                for m_idx, m in enumerate(metric_keys):
                    
                    value = metrics.get(m, "")
                    if isinstance(value, float):
                        value = f"{value:.{numbers_precision}f}"
                    value = str(value)
                    
                    row_cells.append(value)
                    if m_idx < n_metrics - 1:  # Add empty spacer between metrics
                        row_cells.append("")
        
        data_rows.append(" & ".join(row_cells) + " \\\\")

    # Combine all
    latex = []
    latex.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex.append("\\hline")
    latex.extend(header)
    latex.extend(data_rows)
    latex.append("\\hline")
    latex.append("\\end{tabular}")

    return "\n".join(latex)