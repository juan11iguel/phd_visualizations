import pandas as pd
from loguru import logger


def rename_signal_ids_to_var_ids(df: pd.DataFrame, vars_config: dict) -> pd.DataFrame:
    """
    Rename signal ids to var ids in a dataframe.
    :param df: Dataframe to rename signal ids to var ids in.
    :param vars_config: Dictionary with variables configuration, should contain var_id
    and signal_id for each variable.
    :return: None
    """

    # Simple solution when there are no duplicates
    # var_ids, signal_ids = zip(*[(var_info['var_id'], var_info['signal_id']) for var_info in vars_config.values()])
    # return df.rename(columns=dict(zip(signal_ids, var_ids)))

    # Create a dictionary to keep track of the signal_id duplicates
    duplicates = {}

    for var_info in vars_config.values():
        if 'signal_id' not in var_info:
            logger.warning(f"Signal id not found in variable {var_info['var_id']}, skipping")
            continue
        signal_id = var_info['signal_id']
        var_id = var_info['var_id']

        if signal_id in df.columns:
            if signal_id in duplicates:
                # If the signal_id is already in the duplicates dictionary, create a new column
                idx = len(duplicates[signal_id])
                df[f"{signal_id}_{idx}"] = df[signal_id]
                duplicates[signal_id].append(var_id)
            else:
                # If the signal_id is not in the duplicates dictionary, add it
                duplicates[signal_id] = [var_id]
        else:
            logger.warning(f"Signal id {signal_id} not found in dataframe columns.")

    # Assign each var_id to one of the copied signal_id{some_idx} columns
    for signal_id, var_ids in duplicates.items():
        for idx, var_id in enumerate(var_ids):
            signal_name = f"{signal_id}_{idx}" if idx > 0 else signal_id

            df.rename(columns={signal_name: var_id}, inplace=True)

    return df
