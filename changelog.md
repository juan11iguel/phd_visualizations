# Changelog

All notable changes to this project will be documented in this file.

## Version 202403

Many changes to the `test_timeseries.experimental_plot` function, including:

- Ability to compare one or more dataframes using the keyword argument `df_comp`.
- Each dataframe will have its own color, and the tooltip will show the compared values for each available trace.
- A filled area between traces can be displayed with the `fill_between` argument in the input plot dictionary.
- Show group of variables. Multiple traces can be displayed when they have similar names finishing the `var_id` arugment with an asterisk, e.g. `'var_id':'T*'`.
- The `fill` argument can be used with this kind of variables to have a cumulative filled area plot.
- Better plot arrows positioning by having changed the reference system to the plot coordinates.
- Ability to show an "active plot", which is a plot on top of another one showing a boolean variable.
- Y axis limits were buggy by default so now they can either be manually specified with `ylims: [value, value]` or they can be handled by the function using `ylims: 'manual'`