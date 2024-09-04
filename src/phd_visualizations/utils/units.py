import numpy as np
import pandas as pd
from loguru import logger
from iapws import IAPWS97 as w_props
from typing import Literal

ArrayLike = np.ndarray | pd.Series | pd.DataFrame | float | int

# logger = logging.getLogger(__name__)

# unit_mapping = {
#     'ºC': 'C',
#     'm³/h': 'm3h',
#     'm3/h': 'm3h',
#     'mbar': 'mbar',
#     'ms/cm': 'mScm',
#     'mS/cm': 'mScm',
#     'us/cm': 'uScm',
#     'uS/cm': 'uScm',
#     'kW': 'kW',
#     'W': 'W',
#     'MPa': 'MPa',
#     'kg/kg': 'kgkg',
#     'L/s': 'Ls',
# }

"""
Data structure:

unit_category: {
    unit_id: [unit_aliases]
}

New categories and units within them can be added as needed, a good practice would 
be to support all possible unit conversion in both directions in `unit_conversion` function.
"""

unit_mapping = {

    "temperatures":{
        "C": ["ºC", "C", "c"],
        "K": ["K", "k"],
    },
    "pressures":{
        "MPa": ["MPa", "mpa"],
        "bar": ["bar", "BAR"],
        "Pa":  ["Pa", "pa"],
        "mbar": ["mbar", "MBAR", "mBar"],
    },
    "mass_flows":{
        "kgs": ["kg/s", "kgs"],
        "kgh": ["kg/h", "kgh"],
    },
    "volumetric_flows":{
        "Ls": ["L/s", "Ls", "ls"],
        "Lmin": ["L/min", "Lmin", "lmin", "LMIN"],
        "m3h": ["m³/h", "m3h", "m3/h"],
    },
    "concentrations":{
        "kgkg": ["kg/kg", "kgkg", "kgkg"],
        "gL": ["g/L", "gl", "gl"],
        "gkg": ["g/kg", "gkg"],
    },
    "powers":{
        "kW": ["kW", "kw", "KW"],
        "W": ["W", "w"],
    },
    "conductivities":{
        "mScm": ["mS/cm", "ms/cm", "mscm", "mscm"],
        "uScm": ["uS/cm", "us/cm", "uscm", "uscm"],
        "s/m": ["S/m", "s/m", "sm", "sm"],
    },
    "irradiances":{
        "Wm2": ["W/m2", "wm2"],
    },
}

supported_temperature_units = [key for key in unit_mapping["temperatures"].keys()]
supported_pressure_units = [key for key in unit_mapping["pressures"].keys()]
supported_concentration_units = [key for key in unit_mapping["concentrations"].keys()]
supported_power_units = [key for key in unit_mapping["powers"].keys()]
supported_conductivity_units = [key for key in unit_mapping["conductivities"].keys()]
supported_mass_flow_units = [key for key in unit_mapping["mass_flows"].keys()]
supported_volumetric_flow_units = [key for key in unit_mapping["volumetric_flows"].keys()]
supported_flow_units = supported_mass_flow_units + supported_volumetric_flow_units


def unsupported_unit_conversion_msg(input_unit: str, output_unit: str):
    logger.warning(f'Asked for unsupported unit conversion: {input_unit} -> {output_unit}. Skipping unit conversion.')


def find_symbol(symbol: str) -> str:

    # Maybe not the most efficient way to do this
    for unit_type, unit_dict in unit_mapping.items():
        for unit_id, unit_aliases in unit_dict.items():
            if symbol in unit_aliases:
                return unit_id

    else:
         raise ValueError(f"Unknown unit symbol {symbol}, must be one of {unit_mapping}")


def to_K(T: ArrayLike, unit:Literal[supported_temperature_units]) -> ArrayLike:

    if unit == "C":
        return T + 273.15
    elif unit == "K":
        return T
    else:
        raise ValueError(f"Unknown temperature unit {unit}, must be one of {supported_temperature_units}")


def K_to(T: ArrayLike, unit:Literal[supported_temperature_units]) -> ArrayLike:

    if unit == "C":
        return T - 273.15
    elif unit == "K":
        return T
    else:
        raise ValueError(f"Unknown temperature unit {unit}, must be one of {supported_temperature_units}")


def to_MPa(P: ArrayLike, unit:Literal[supported_pressure_units]) -> ArrayLike:

    if unit == "bar":
        return P * 0.1
    elif unit == "MPa":
        return P
    elif unit == "Pa":
        return P * 1e-6
    elif unit == "mbar":
        return P * 1e-3
    else:
        raise ValueError(f"Unknown pressure unit {unit}, must be one of {supported_pressure_units}")


def MPa_to(P: ArrayLike, unit:Literal[supported_pressure_units]) -> ArrayLike:

    if unit == "bar":
        return P * 10
    elif unit == "MPa":
        return P
    elif unit == "Pa":
        return P * 1e6
    elif unit == "mbar":
        return P * 1e3
    else:
        raise ValueError(f"Unknown pressure unit {unit}, must be one of {supported_pressure_units}")


def to_kW(P: ArrayLike, unit:Literal[supported_power_units]) -> ArrayLike:

    if unit == "W":
        return P * 1e-3
    elif unit == "kW":
        return P
    else:
        raise ValueError(f"Unknown power unit {unit}, must be one of {supported_power_units}")


def kW_to(P: ArrayLike, unit:Literal[supported_power_units]) -> ArrayLike:

        if unit == "W":
            return P * 1e3
        elif unit == "kW":
            return P
        else:
            raise ValueError(f"Unknown power unit {unit}, must be one of {supported_power_units}")


def to_kgs(q: ArrayLike, unit:Literal[supported_mass_flow_units+supported_volumetric_flow_units],
           P_MPa: ArrayLike = None, T_K: ArrayLike = None) -> ArrayLike:

    def calculate_m(row):
        return row['q'] * w_props(T=row['T_K'], P=row['P_MPa']).rho / 3600 # m³/h * rho [kg/m³] / 3600 [s/h] = kg/s

    if unit == "kgh":
        return q / 3600
    elif unit == "kgs":
        return q
    elif unit in supported_volumetric_flow_units:
        q = to_m3h(q, unit)
        if P_MPa is None or T_K is None:
            raise ValueError("Pressure (MPa) and temperature (K) must be provided to convert volumetric flow to mass flow")
        else:
            # Bazofia del año?
            temp_df = pd.DataFrame({'q': q, 'P_MPa': P_MPa, 'T_K': T_K})
            m = temp_df.apply(calculate_m, axis=1)

            return m
    else:
        raise ValueError(f"Unknown mass flow unit {unit}, must be one of {supported_mass_flow_units}")


def kgs_to(m: ArrayLike, unit:Literal[supported_mass_flow_units+supported_volumetric_flow_units],
           P_MPa: ArrayLike = None, T_K: ArrayLike = None) -> ArrayLike:

    if unit == "kgh":
        return m * 3600
    elif unit == "kgs":
        return m
    elif unit in supported_volumetric_flow_units:
        if P_MPa is None or T_K is None:
            raise ValueError("Pressure (MPa) and temperature (K) must be provided to convert mass flow to volumetric flow")
        else:
            q = to_m3h(m, unit='kgs', P_MPa=P_MPa, T_K=T_K)
            return  m3h_to(q, unit)
    else:
        raise ValueError(f"Unknown mass flow unit {unit}, must be one of {supported_mass_flow_units}")


def to_m3h(q: ArrayLike, unit:Literal[supported_volumetric_flow_units + supported_mass_flow_units],
           P_MPa: ArrayLike = None, T_K: ArrayLike = None) -> ArrayLike:

    def calculate_q(row):
        return row['m'] / w_props(T=row['T_K'], P=row['P_MPa']).rho * 3600 # kg/s * 1/rho [kg/m³] * 3600 [s/h] = m³/h

    if unit == "m3h":
        return q
    elif unit == "Ls":
        return q * 3.6
    elif unit == "Lmin":
        return q * 60/1000
    elif unit in supported_mass_flow_units:
        if P_MPa is None or T_K is None:
            raise ValueError("Pressure (MPa) and temperature (K) must be provided to convert mass flow to volumetric flow")
        else:
            m = to_kgs(q, unit, P_MPa=P_MPa, T_K=T_K) # kg/s
            # Bazofia del año?
            temp_df = pd.DataFrame({'m': m, 'P_MPa': P_MPa, 'T_K': T_K})
            q = temp_df.apply(calculate_q, axis=1)

            return q
    else:
        raise ValueError(f"Unknown volumetric flow unit {unit}, must be one of {supported_volumetric_flow_units}")


def m3h_to(q: ArrayLike, unit:Literal[supported_volumetric_flow_units + supported_mass_flow_units],
           P: ArrayLike = None, T: ArrayLike = None) -> ArrayLike:

    if unit == "m3h":
        return q
    elif unit == "Ls":
        return q / 3.6
    elif unit == "Lmin":
        return q * 1000/60
    elif unit in supported_mass_flow_units:
        if P is None or T is None:
            raise ValueError("Pressure (MPa) and temperature (K) must be provided to convert mass flow to volumetric flow")
        else:
            m = to_kgs(q, unit='m3h', P_MPa=P, T_K=T) # kg/s
            return kgs_to(m, unit)
    else:
        raise ValueError(f"Unknown volumetric flow unit {unit}, must be one of {supported_volumetric_flow_units}")

# TODO: Redo as with flows, when converting from concentration to conductivity, first convert to the common cateogry
#  unit and then do the conversion to the desired unit, and viceversa.

# TODO: For conductivity, the common unit should be one supported by conductivity_to_mass_fraction

def to_kgkg(w: ArrayLike, unit:Literal[supported_concentration_units + supported_conductivity_units]) -> ArrayLike:

    if unit == "kgkg":
        return w
    elif unit == "gkg":
        return w * 1e-3
    elif unit == "gL":
        return w * 1e-3
    elif unit == "mScm":
        return conductivity_to_mass_fraction(w, input_unit='mS/cm', output_unit='kg/kg')
    elif unit == "uScm":
        return conductivity_to_mass_fraction(w, input_unit='uS/cm', output_unit='kg/kg')
    else:
        raise ValueError(f"Unknown concentration unit {unit}, must be one of {supported_concentration_units}")


def kgkg_to(w: ArrayLike, unit:Literal[supported_concentration_units + supported_conductivity_units]) -> ArrayLike:

    if unit == "kgkg":
        return w
    elif unit == "gkg":
        return w * 1e3
    elif unit == "gL":
        return w * 1e3
    elif unit == "mScm":
        raise NotImplementedError("Conversion from kg/kg to mS/cm not implemented")
    elif unit == "uScm":
        raise NotImplementedError("Conversion from kg/kg to uS/cm not implemented")
    else:
        raise ValueError(f"Unknown concentration unit {unit}, must be one of {supported_concentration_units}")


def to_Sm(w: ArrayLike, unit:Literal[supported_conductivity_units]) -> ArrayLike:

    if unit == "mScm":
        return w * 1e-2
    elif unit == "uScm":
        return w * 1e-6
    elif unit == "Sm":
        return w
    else:
        raise ValueError(f"Unknown conductivity unit {unit}, must be one of {supported_conductivity_units}")


def Sm_to(w: ArrayLike, unit:Literal[supported_conductivity_units]) -> ArrayLike:

    if unit == "mScm":
        return w * 1e2
    elif unit == "uScm":
        return w * 1e6
    elif unit == "Sm":
        return w
    else:
        raise ValueError(f"Unknown conductivity unit {unit}, must be one of {supported_conductivity_units}")


def unit_conversion(
        df: pd.DataFrame, signals_config: dict,
        input_unit_key: str = "unit", output_unit_key: str = "unit_model",
        update_mode: Literal["replace", "rename_input_unit", "rename_output_unit", "rename_both"] = "replace",
        skip_unknown_symbols:bool = False
) -> pd.DataFrame:

    """
    Converts variables units from a dataframe given a dictionary that
    contains `input_unit_key` and `output_unit_key` keys, from `input_unit_key` to `output_unit_key`.

    Supported conversions:
        - Temperature: ºC -> K
        - Pressure: mbar, bar -> MPa
        - Flow: L/s, m³/h -> kg/s
        - Salinity: mS/cm -> kg/kg

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe including all the required signals values.
    signals_config : dict
        A dictionary including all the required signals configuration:
    input_unit_key: keyword given in configuration to the source/input unit attribute.
    output_unit_key: keyword given in configuration to the destination/output unit attribute.
    update_mode : str, optional, decides how to update the dataframe:
        - "replace": replace the original signal with the converted one.
        - "rename_input_unit": rename the original signal to include the input unit.
        - "rename_output_unit": rename the original signal to include the output unit.
        - "rename_both": rename the original signal to include both the input and output units.

    Returns
    -------
    df : pandas.DataFrame
        An updated pandas dataframe where the original variables are renamed to include
        their units and new ones are added with the model units and a suffix indicating the unit.

    """

    # # Convert flow rates to kg/s
    # Ms_kgs = Ms_Ls * rho_s/1000 # L/s -> kg/s
    # Mprod_kgs = Mprod_m3h * rho_d/3600 # m³/h -> kg/s

    # # Convert temperatures to Kelvin
    # Tsin_K = Tsin_C + 273.15
    # Tsout_K = Tsout_C + 273.15

    # Reorder var_ids to make sure to start by temperatures that are needed for other conversions

    input_units = []
    output_units = []
    var_ids = []
    temp_df = pd.DataFrame()

    for var_id in signals_config:

        if var_id not in df.columns:
            continue  # Skip conversion if signal not in dataframe but in configuration

        if input_unit_key not in signals_config[var_id] or output_unit_key not in signals_config[var_id]:
            logger.warning(
                f"Signal {var_id} does not have {input_unit_key} or {output_unit_key} specified in signals_config. Skipping unit conversion."
            )
        else:
            try:
                # Convert signals_config[var_id][input_unit_id] based on unit_mapping
                input_units.append( find_symbol(signals_config[var_id][input_unit_key]) )
                output_units.append( find_symbol(signals_config[var_id][output_unit_key]) )
                var_ids.append(var_id) # At the end, so if an exception is raised before, the list is not updated
            except ValueError as e:
                if skip_unknown_symbols:
                    logger.warning(f"Skipping unknown symbols in signals_config, {var_id}: {signals_config[var_id].get(input_unit_key, None)} -> {signals_config[var_id].get(output_unit_key, None)}")

                    # Make sure that input_unit is not added if output_unit is not recognised
                    if len(input_units) > len(output_units):
                        input_units.pop(-1)
                else:
                    raise e

            # units.append(signals_config[var_id][input_unit_id])

    # First temperatures
    idxs = [index for index, item in enumerate(input_units) if item in supported_temperature_units]
    # Second pressures
    idxs += [index for index, item in enumerate(input_units) if
             (item in supported_pressure_units) and (index not in idxs)]
    # Everything else
    idxs += [index for index, item in enumerate(input_units) if index not in idxs]

    # Reorder the associated list based on the idxs
    var_ids = [var_ids[index] for index in idxs]
    input_units = [input_units[index] for index in idxs]
    output_units = [output_units[index] for index in idxs]

    for var_id, input_unit, output_unit in zip(var_ids, input_units, output_units):

        converted_value = None

        # Temperatures
        if input_unit in supported_temperature_units:
            if output_unit in supported_temperature_units:
                # Always convert to K and then to the desired unit
                T_K = to_K(df[var_id], input_unit)
                converted_value = K_to(T_K, output_unit)

                # Store the K value
                temp_df[f'{var_id}_K'] = T_K
            else:
                unsupported_unit_conversion_msg(input_unit, output_unit)
                continue

        # Pressures
        elif input_unit in supported_pressure_units:
            if output_unit in supported_pressure_units:
                # Always convert to MPa and then to the desired unit
                P_MPa = to_MPa(df[var_id], input_unit)
                converted_value = MPa_to(P_MPa, output_unit )

                # Store the MPa value
                temp_df[f'{var_id}_MPa'] = P_MPa
            else:
                unsupported_unit_conversion_msg(input_unit, output_unit)
                continue

        # Power
        elif input_unit in supported_power_units:
            if output_unit in supported_power_units:
                # Always convert to kW and then to the desired unit
                converted_value = kW_to( to_kW(df[var_id], input_unit), output_unit )
            else:
                unsupported_unit_conversion_msg(input_unit, output_unit)
                continue

        # Salinity
        elif input_unit in supported_concentration_units:
            if (output_unit in supported_concentration_units or
                    output_unit in supported_conductivity_units):
                # Always convert to kg/kg and then to the desired unit
                converted_value = kgkg_to( to_kgkg(df[var_id], input_unit), output_unit )
            else:
                unsupported_unit_conversion_msg(input_unit, output_unit)
                continue

        # Conductivity
        elif input_unit in supported_conductivity_units:
            if output_unit in supported_conductivity_units:
                # Always convert to S/cm and then to the desired unit
                converted_value = Sm_to( to_Sm(df[var_id], input_unit), output_unit )
            else:
                unsupported_unit_conversion_msg(input_unit, output_unit)
                continue

        # Flows
        elif input_unit in supported_flow_units:
            if output_unit in supported_flow_units:

                # Gather auxiliary variables
                p_aux_id = f'P{var_id[1:]}_MPa'
                if p_aux_id in temp_df:
                    p_aux = temp_df[p_aux_id]  # MPa
                else:
                    p_aux = np.ones(len(df)) * 0.12  # MPa (1.2 bar by default)
                    logger.warning(
                        f'{p_aux_id} not found in available variables. Using default value of {p_aux[0]} MPa')

                # Gather all temperatures that share the same prefix as the flow variable
                # For example: qmed_s -> Tmed_s_in_K, Tmed_s_out_K: Tmed_s*_K
                # And use the average of those temperatures
                T_aux_ids = [col for col in temp_df.columns if col.startswith(f'T{var_id[1:]}') and col.endswith('_K')]
                if len(T_aux_ids) > 0:
                    T_aux = temp_df[T_aux_ids].mean(axis=1)
                else:
                    T_aux = np.ones(len(df)) * 25 + 273.15
                    logger.warning(
                        f'Could not find any variable matching T{var_id[1:]}*_K in available variables. Using default value of {T_aux[0]} K')

                pd.isnull(T_aux).sum()

                if input_unit in supported_mass_flow_units:
                    if output_unit in supported_mass_flow_units:
                        # Always convert to kg/s and then to the desired unit
                        converted_value = kgs_to(to_kgs(df[var_id], input_unit), output_unit)
                    else:
                        # First convert to kg/s, then to m³/h and finally to the desired unit
                        converted_value = m3h_to(
                            to_m3h(
                                to_kgs(df[var_id], input_unit),
                                unit='kgs', P_MPa=p_aux, T_K=T_aux
                            ),
                            output_unit
                        )
                elif input_unit in supported_volumetric_flow_units:
                    if output_unit in supported_volumetric_flow_units:
                        # Always convert to m³/h and then to the desired unit
                        converted_value = m3h_to(to_m3h(df[var_id], input_unit), output_unit)
                    else:
                        # First convert to m³/h, then to kg/s and finally to the desired unit
                        converted_value = kgs_to(
                            to_kgs(
                                to_m3h(df[var_id], input_unit),
                                unit='m3h', P_MPa=p_aux, T_K=T_aux
                            ),
                            output_unit
                        )
                else:
                    unsupported_unit_conversion_msg(input_unit, output_unit)
                    continue
            else:
                unsupported_unit_conversion_msg(input_unit, output_unit)
                continue

        # No conversion needed, just rename
        # Important to check this after all the other conversions,
        # since in some of them, even if the input and output units are the same,
        # their values are stored for other unit conversions (e.g. T_K)
        elif input_unit == output_unit:
            converted_value = df[var_id]

            logger.info(
                f'No unit conversion needed for {var_id}, just performing renaming if set in `update_mode`: {var_id}_{input_unit}'
            )

        # Unknown unit conversion
        else:
            unsupported_unit_conversion_msg(input_unit, output_unit)
            continue

        # After conversions, update the dataframe
        if converted_value is None: # If no conversion was performed
            converted_value = df[var_id]

        # Update dataframe based on `update_mode`
        if update_mode == 'replace':
            df[var_id] = converted_value
            logger.debug(f'Updated {var_id} to {output_unit} from {input_unit}')

        elif update_mode == 'rename_input_unit':
            df[f"{var_id}_{input_unit}"] = df[var_id] # Copy the original value in the renamed column
            df[var_id] = converted_value # Set the converted value in the original signal column
            logger.debug(f'Updated {var_id} to {output_unit} from {input_unit}, original value stored in {var_id}_{input_unit}')

        elif update_mode == 'rename_output_unit':
            df[f"{var_id}_{output_unit}"] = converted_value
            logger.debug(f'Added new variable with {output_unit} from {input_unit} and stored it `{var_id}_{output_unit}`')

        elif update_mode == 'rename_both':
            df.rename(columns={var_id: f'{var_id}_{input_unit}'}, inplace=True) # Rename original signal column
            df[f"{var_id}_{output_unit}"] = converted_value # Create a new column with the converted value

            logger.debug(f'Updated {var_id} with {output_unit} from {input_unit}, '
                         f'original value stored in {var_id}_{input_unit} and new value '
                         f'in {var_id}_{output_unit}')
        else:
            raise ValueError(f'Unknown update_mode {update_mode}, must be `replace`, `rename_input_unit`, '
                             f'`rename_output_unit` or `rename_both`')

    return df


def conductivity_to_mass_fraction(C, input_unit='mS/cm', output_unit='kg/kg'):
    """
    Converts salinity measured with conductivity in mS/cm to concentration
    in kg/kg based on fitted curve

    Parameters
    ----------
    C : float
        Measured conductivity in mS/cm.

    Returns
    -------
    w : float
        Salinity in kg/kg.
    uncertainty : float
        Uncertainty associated with conversion.

    """

    if input_unit == 'uS/cm':
        C = C * 1e-3
    elif input_unit == 'S/m':
        C = C * 10
    elif input_unit != 'mS/cm':
        raise ValueError(f'Unknown input unit {input_unit}, must be mS/cm or uS/cm or S/m')

    if C < 50:
        # Molar conductivity of ions in the NaCl solution (mS/cm)
        kappa = 126 * 1e-3  # Approximate value at 25°C

        conc = C / kappa

    else:
        a0 = 62.469
        a1 = -1.933
        a2 = 0.0362
        a3 = -0.0002
        a4 = 4.00E-07

        conc = C ** 4 * a4 + C ** 3 * a3 + C ** 2 * a2 + C * a1 + a0  # g/L

    if output_unit == 'kg/kg':
        w = 1 / 1023 * conc  # rho = 1023 g/L at 25ºC

    elif output_unit == 'g/L':
        w = conc
    else:
        raise ValueError(f'Unknown output unit {output_unit}, must be kg/kg or g/L')

    return w