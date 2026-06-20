import numpy as np
import pandas as pd

def load_data(base_path=".", use_aged=False):
    exp_file = f"{base_path}/Dataset/Experimental_data_aged_cell.csv" if use_aged else f"{base_path}/Dataset/Experimental_data_fresh_cell.csv"
    ocv_file = f"{base_path}/Dataset/OCV_vs_SOC_curve.csv"
    df = pd.read_csv(exp_file)
    df_ocv = pd.read_csv(ocv_file)
    # Standardize column names if needed
    df.columns = [c.strip().title() for c in df.columns]
    return df, df_ocv

def coulomb_counter_soc(df, capacity_Ah=19.96):
    # Integrate current over time with calibration at voltage limits (proxy from paper’s supplementary)
    df = df.copy()
    dt = df['Time'].diff().fillna(0.0)
    # Coulomb counting (discharge current reduces SOC). Current assumed + discharge; adjust sign if needed.
    soc = np.zeros(len(df))
    soc[0] = 0.5
    cap_As = capacity_Ah * 3600.0
    for i in range(1, len(df)):
        soc[i] = soc[i-1] - (dt.iloc[i] / cap_As) * df['Current'].iloc[i]
        # Simple calibration near end-of-charge and end-of-discharge using voltage thresholds
        if (df['Voltage'].iloc[i] > 4.19 and abs(df['Current'].iloc[i]) < 4) or soc[i] > 1:
            soc[i] = 1.0
        if (df['Voltage'].iloc[i] < 3.01 and abs(df['Current'].iloc[i]) < 4) or soc[i] < 0:
            soc[i] = 0.0
    df['SOC_CC'] = soc
    return df

def compute_reference_soh(df, capacity_Ah_nom=19.96, window_As_mult=2.0):
    # Accumulate charge throughput until threshold, then compare experimental vs model-like throughput to estimate SOH~ ratio
    # Here we approximate: SOH = delivered capacity / nominal capacity in that segment.
    df = df.copy()
    capacity_Ah_nom = 19.96
    cap_As_nom = capacity_Ah_nom * 3600.0

    # crude cycle detection: whenever voltage falls below ~3.1V we assume a discharge cycle
    df['Cycle'] = (df['Voltage'] < 3.1).cumsum()

    soh_values = {}
    for cycle_id, group in df.groupby('Cycle'):
        if len(group) > 10:  # skip tiny segments
            delivered_As = np.trapz(group['Current'], group['Time'])  # integrate current over time
            soh = abs(delivered_As) / cap_As_nom
            soh_values[cycle_id] = soh

    df['SOH_ref'] = df['Cycle'].map(soh_values)
    return df

def ocv_lookup(df_ocv, soc_array):
    # df_ocv must have columns: SOC in [0,1], V0
    soc_vals = df_ocv['SOC'].values
    v0_vals = df_ocv['V0'].values
    return np.interp(soc_array, soc_vals, v0_vals)

def engineer_features(df, df_ocv, capacity_Ah=19.96):
    df = df.copy()
    df['dVdt'] = df['Voltage'].diff().fillna(0.0) / df['Time'].diff().replace(0, np.nan).fillna(1.0)
    df['dIdt'] = df['Current'].diff().fillna(0.0) / df['Time'].diff().replace(0, np.nan).fillna(1.0)
    df['absI'] = df['Current'].abs()
    df['I2'] = df['Current']**2
    df['TempC'] = df['Temperature']
    # OCV estimate at SOC_CC (proxy)
    df['V0_hat'] = ocv_lookup(df_ocv, np.clip(df['SOC_CC'].values, 0, 1))
    # Internal resistance proxy
    df['R_hat'] = (df['V0_hat'] - df['Voltage']).replace([np.inf, -np.inf], np.nan)
    # Rolling stats
    for col in ['Current','Voltage','Temperature','dVdt','dIdt','R_hat']:
        df[f'{col}_mean_60s'] = df[col].rolling(60, min_periods=1).mean()
        df[f'{col}_std_60s'] = df[col].rolling(60, min_periods=1).std().fillna(0.0)
    # Targets
    # SOC target from Coulomb counter (proxy)
    df['SOC_target'] = df['SOC_CC']
    # SOH target will be filled by compute_reference_soh() before calling engineer_features, or you can set later.
    return df

def windowize(df, feature_cols, target_col, window=60):
    X, y = [], []
    vals = df[feature_cols].values
    tgt = df[target_col].values
    for i in range(window, len(df)):
        X.append(vals[i-window:i])
        y.append(tgt[i])
    return np.array(X), np.array(y)
