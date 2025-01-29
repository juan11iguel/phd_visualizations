import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from phd_visualizations.calculations import calculate_uncertainty, SupportedInstruments
from phd_visualizations.constants import default_fontsize, plt_colors
from phd_visualizations.utils import hex_to_rgba_str

def regression_plot(df_ref: pd.DataFrame, df_mod: pd.DataFrame, var_ids: list[str], units: list[str] = None, instruments: list[SupportedInstruments] = None, **kwargs) -> go.Figure:
    if instruments is None:
        instruments = [None] * len(var_ids)
    
    if units is None:
        units = [None] * len(var_ids)
            
    rmse_list = [np.sqrt(mean_squared_error(df_ref[var_id].values, df_mod[var_id].values)) for var_id in var_ids]
    fig = make_subplots(rows=len(var_ids), cols=1, 
                        subplot_titles=[f"<b>{var_id}</b><br>RMSE={rmse:.2f} [{unit}]" for var_id, rmse, unit in zip(var_ids, rmse_list, units)])
    
    for i, var_id in enumerate(var_ids):
        x = df_ref[var_id].values
        
        # Perform linear regression
        # model = LinearRegression()
        # model.fit(x, y)
        # y_pred = model.predict(x)
        
        # Calculate uncertainty
        if instruments[i]:
            uncertainty = calculate_uncertainty(x, instruments[i])
        else:
            uncertainty = np.zeros_like(x)
        
        # Create scatter plot
        scatter = go.Scatter(x=x, y=df_mod[var_id], mode='markers', name='Model results', showlegend=True if i==0 else False, 
                             marker=dict(color=plt_colors[1], size=5))
        
        # Create regression line plot
        regression_line = go.Scatter(x=x, y=x, mode='lines', name='Perfect fit', showlegend=True if i==0 else False,
                                     line=dict(color=plt_colors[0], width=2))
        # Create uncertainty plot
        color = plt_colors[-1]
        color_ = hex_to_rgba_str(color, alpha=0.3)
        upper_bound = go.Scatter(x=x, y=x + uncertainty, mode='lines', fill=None, line=dict(color=color, width=0.1), fillcolor=color_, showlegend=False, name=f'Uncertainty {var_id}')
        lower_bound = go.Scatter(x=x, y=x - uncertainty, mode='lines', fill='tonexty', line=dict(color=color, width=0.1), fillcolor=color_, showlegend=False)
        
        # Add plots to subplot
        fig.add_trace(upper_bound, row=i+1, col=1)
        fig.add_trace(lower_bound, row=i+1, col=1)
        fig.add_trace(scatter, row=i+1, col=1)
        fig.add_trace(regression_line, row=i+1, col=1)
        
        # Update subplot title with RMSE
        fig.update_yaxes(title_text=f"Predicted values [{units[i] if units else ''}]", row=i+1, col=1)
        fig.update_xaxes(title_text=f"Experimental values [{units[i] if units else ''}]", row=i+1, col=1)
    
    fig.update_layout(
        height=300*len(var_ids), 
        title_text="Model validation",
        font_size=kwargs.get('font_size', default_fontsize),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    # Allign subplot titles to the left
    # [annotation.update(x=0.025) for annotation in fig.layout.annotations]
    
    return fig