def make_decision(**context):
    import pickle
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from datetime import datetime, timedelta

    print("[INFO] Starting decision-making process...")

    exec_date = context.get('logical_date', context.get('execution_date'))
    if exec_date is None:
        raise ValueError("No execution date found in context")

    exec_date_naive = exec_date.replace(tzinfo=None) if exec_date.tzinfo else exec_date
    date_str = exec_date_naive.strftime('%Y-%m-%d')
    today_naive = datetime.utcnow().replace(tzinfo=None)

    

    base_dir = Path(__file__).parent.parent
    q_table_path = base_dir / "models" / f"q_table_global.pkl"
    forecast_temp_path = base_dir / "data" / "predictions" / f"prophet_forecast_temperature_{date_str}.csv"
    forecast_flow_path = base_dir / "data" / "predictions" / f"prophet_forecast_flow_{date_str}.csv"
    irradiance_path = base_dir / "data" / "processed" / f"preprocessed_{date_str}.csv"

    if not q_table_path.exists():
        raise FileNotFoundError(f"Q-table missing: {q_table_path}")
    if not forecast_temp_path.exists() or not forecast_flow_path.exists():
        raise FileNotFoundError(f"Forecast files missing for {date_str}")
    if not irradiance_path.exists():
        raise FileNotFoundError(f"Processed irradiance file missing for {date_str}")

    print("📥 [INFO] Loading data...")
    with open(q_table_path, "rb") as f:
        q_table = pickle.load(f)

    df_temp = pd.read_csv(forecast_temp_path)
    df_flow = pd.read_csv(forecast_flow_path)
    df_irr = pd.read_csv(irradiance_path)[["datetime", "solar_irradiance"]]

    def normalize_datetime(df, col="datetime"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")  # parse to UTC
        df[col] = df[col].dt.tz_localize(None)  # make it naive
        return df

    # Apply normalization
    df_temp = normalize_datetime(df_temp)
    df_flow = normalize_datetime(df_flow)
    df_irr = normalize_datetime(df_irr)

    df = pd.merge(df_temp, df_flow, on="datetime", suffixes=("_temp", "_flow"))
    df = pd.merge(df, df_irr, on="datetime")

    if df.empty:
        print(f"[WARN] No merged data available for {date_str}.")
        return


    # Bins normaux (valeurs entre 0 et 1)
    temp_bins = np.linspace(0, 1, 6)       # → 7 classes
    flow_bins = np.linspace(0, 1, 6)       # → 5 classes
    solar_bins = np.linspace(0, 1, 6)     # → 9 classes

    nb_temp_bins = len(temp_bins) - 1
    nb_flow_bins = len(flow_bins) - 1
    nb_solar_bins = len(solar_bins) - 1



    def discretize(value, bins):
        idx = np.digitize(value, bins) - 1
        return max(0, min(idx, len(bins) - 2))

    def get_state(row):
        t = discretize(row["yhat_temp"], temp_bins)
        f = discretize(row["yhat_flow"], flow_bins)
        s = discretize(row["solar_irradiance"], solar_bins)
        return t * nb_flow_bins * nb_solar_bins + f * nb_solar_bins + s


    decisions = []

    for _, row in df.iterrows():
        state = get_state(row)
        if state >= len(q_table):
            print(f"[WARN] Skipping invalid state {state} (Q-table size: {len(q_table)})")
            continue

        action = int(np.argmax(q_table[state]))  # 0 = off, 1 = on
        decision_text = "Chauffer (résistance activée)" if action == 1 else "Ne pas chauffer (solaire ou inutilisé)"

        timestamp = row["datetime"]
        print(f"[INFO] {timestamp} → {decision_text} | Temp: {row['yhat_temp']:.2f}°C, Débit: {row['yhat_flow']:.4f} m³/s, Irrad: {row['solar_irradiance']:.2f} W/m²")

        decisions.append({
            "datetime": str(timestamp),
            "temp": round(row["yhat_temp"], 2),
            "flow": round(row["yhat_flow"], 4),
            "irradiance": round(row["solar_irradiance"], 2),
            "action": action,
            "text": decision_text
        })

    logs_dir = base_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"decision_{date_str}.csv"
    pd.DataFrame(decisions).to_csv(log_path, index=False)
    print(f"[INFO] All hourly decisions saved to {log_path}")

    context['ti'].xcom_push(key="decisions", value=decisions)
    return decisions