def run_q_learning(**context):
    import numpy as np
    import os
    import pickle
    import pandas as pd
    from pathlib import Path
    from datetime import datetime, timedelta
    import matplotlib.pyplot as plt

    print("[INFO] Starting Q-learning process...")

    exec_date = context.get('logical_date', context.get('execution_date'))
    if exec_date is None:
        raise ValueError("No execution date found in context")

    date_str = exec_date.strftime('%Y-%m-%d')
    exec_date_naive = exec_date.replace(tzinfo=None) if exec_date.tzinfo else exec_date
    today_naive = datetime.utcnow().replace(tzinfo=None)


    base_dir = Path(__file__).parent.parent
    pred_dir = base_dir / "data" / "predictions"
    processed_dir = base_dir / "data" / "processed"
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)

    temp_path = pred_dir / f"prophet_forecast_temperature_{date_str}.csv"
    flow_path = pred_dir / f"prophet_forecast_flow_{date_str}.csv"
    irradiance_path = processed_dir / f"preprocessed_{date_str}.csv"
    pred_start = exec_date_naive - pd.Timedelta(days=180)
    pred_end = exec_date_naive - pd.Timedelta(days=1)

    irradiance_files = []
    for f in sorted(processed_dir.glob("preprocessed_*.csv")):
        file_date_str = f.stem.split("_")[-1]
        file_date = pd.to_datetime(file_date_str).tz_localize(None)
        if pred_start <= file_date <= pred_end:
            irradiance_files.append(f)

    if not temp_path.exists() or not flow_path.exists():
        raise FileNotFoundError(f"Missing prediction files: {temp_path} or {flow_path}")
    if not irradiance_path.exists():
        raise FileNotFoundError(f"Missing processed data file: {irradiance_path}")
    if not irradiance_files:
        raise FileNotFoundError(f"No irradiance data between {pred_start} and {pred_end}")

    def normalize_datetime(df, col="datetime"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")  # force parsing with UTC
        df[col] = df[col].dt.tz_localize(None)  # make datetime naive
        return df

    # Normalize all datetime columns before merging
    
    print("[INFO] Loading data files...")
    temp_df = pd.read_csv(temp_path)
    flow_df = pd.read_csv(flow_path)
    irr_df_list = [pd.read_csv(f)[["datetime", "solar_irradiance"]] for f in irradiance_files]
    irr_df = pd.concat(irr_df_list)


    temp_df = normalize_datetime(temp_df)
    flow_df = normalize_datetime(flow_df)
    irr_df = normalize_datetime(irr_df)

    data = pd.merge(temp_df, flow_df, on="datetime", suffixes=("_temp", "_flow"))
    data = pd.merge(data, irr_df, on="datetime")

    data["temp_norm"] = data["yhat_temp"] 
    data["solar_norm"] = data["solar_irradiance"] 
    data["flow_norm"] = data["yhat_flow"] 
    
    print(f"[INFO] Merged dataset contains {len(data)} rows")
    if len(data) < 2:
        raise ValueError(f"Not enough data to train Q-learning for {date_str}")

    np.random.seed(42)
    n_actions = 2

    def discretize(value, bins):
        return min(np.digitize(value, bins) - 1, len(bins) - 2)

    temp_bins = np.linspace(data["yhat_temp"].min(), data["yhat_temp"].max(), 6)
    flow_bins = np.linspace(data["yhat_flow"].min(), data["yhat_flow"].max(), 6)
    solar_bins = np.linspace(data["solar_irradiance"].min(), data["solar_irradiance"].max(), 6)


    n_states = (len(temp_bins) - 1) * (len(flow_bins) - 1) * (len(solar_bins) - 1)


    model_path = model_dir / f"q_table_global.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            q_table = pickle.load(f)
        print(f"[INFO] Loaded existing Q-table")
    else:
        q_table = np.random.uniform(low=0.0, high=0.01, size=(n_states, n_actions))
        print(f"[INFO] Initialized new Q-table")
    alpha = 0.1
    gamma = 0.9
    epsilon = 1.0
    min_epsilon = 0.1
    epsilon_decay = 0.995


    reward_counters = {
        "low_temp_low_solar": 0,
        "high_temp_high_solar": 0,
        "high_flow_low_temp": 0,
        "other": 0
    }

    def get_state(row):
        t = discretize(row["yhat_temp"], temp_bins)
        f = discretize(row["yhat_flow"], flow_bins)
        s = discretize(row["solar_irradiance"], solar_bins)

        return t * (len(flow_bins) - 1) * (len(solar_bins) - 1) + f * (len(solar_bins) - 1) + s

    
    raw_path = processed_dir / f"simulated_solar_heating_{date_str}.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw data file needed for normalization: {raw_path}")

    raw_df = pd.read_csv(raw_path, parse_dates=["datetime"])

    flow_min = raw_df["flow_rate"].min()
    flow_max = raw_df["flow_rate"].max()

    # Définir seuil de flow (exemple 0.006 m3/s)
    seuil_flow = 0.005
    seuil_flow_norm = (seuil_flow - flow_min) / (flow_max - flow_min)
    def reward(row, action):
        temp = row["temp_norm"]
        solar = row["solar_norm"]
        flow = row["flow_norm"]

        if flow < seuil_flow_norm:
            if action == 1:
                return -2.0  
            else:
                return 1.0   

        if temp < 0.4 and solar < 0.3:
            reward_counters["low_temp_low_solar"] += 1
            return 0.5 if action == 1 else 0
        elif temp > 0.7 and solar > 0.6:
            reward_counters["high_temp_high_solar"] += 1
            return -1 if action == 1 else 1
        elif flow > 0.7 and temp < 0.6:
            reward_counters["high_flow_low_temp"] += 1
            return 1 if action == 1 else -1
        else:
            reward_counters["other"] += 1
            return 0.1 if action == 0 else -0.1


    print("[INFO] Starting Q-learning training...")

    for epoch in range(1000):
        total_reward = 0
        for i in range(len(data) - 1):
            row = data.iloc[i]
            next_row = data.iloc[i + 1]

            state = get_state(row)
            next_state = get_state(next_row)

            if np.random.rand() < epsilon:
                action = np.random.choice(n_actions)
            else:
                action = np.argmax(q_table[state])

            r = reward(row, action)
            total_reward += r

            q_table[state, action] += alpha * (
                r + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )

        epsilon = max(min_epsilon, epsilon * epsilon_decay)  # decay epsilon
        if epoch % 200 == 0:
            print("[DEBUG] Reward counters snapshot:", reward_counters)

        if epoch % 50 == 0:
            print(f"[DEBUG] Epoch {epoch}: Total reward = {total_reward:.2f}, Epsilon = {epsilon:.4f}")

    print("\n[DEBUG] Reward rule trigger counts during training:")
    for key, val in reward_counters.items():
        print(f" - {key}: {val} times")

    with open(model_path, "wb") as f:
        pickle.dump(q_table, f)

    print("[INFO] Global Q-table saved (continuous learning)")
    print("[INFO] Starting policy evaluation on last available day...")

    last_day = data.iloc[-24:].copy()
    last_day["action"] = None
    last_day["reward"] = None

    for i in range(len(last_day)):
        row = last_day.iloc[i]
        state = get_state(row)
        action = np.argmax(q_table[state])
        r = reward(row, action)

        last_day.loc[last_day.index[i], "action"] = int(action)
        last_day.loc[last_day.index[i], "reward"] = float(r)

    last_day["action"] = last_day["action"].astype(int)
    last_day["reward"] = last_day["reward"].astype(float)

    print("\n[RESULT] Policy Evaluation on Last Day")
    print(last_day[["datetime", "yhat_temp", "solar_irradiance", "yhat_flow", "action", "reward"]].to_string(index=False))

    q_df = pd.DataFrame(q_table, columns=["Q_action_0", "Q_action_1"])
    q_df["state"] = q_df.index
    q_df = q_df[["state", "Q_action_0", "Q_action_1"]]
    q_df["best_action"] = q_df[["Q_action_0", "Q_action_1"]].idxmax(axis=1)

    print("\n[RESULT] Q-learning Q-table (Top non-zero rows):")
    print(q_df[q_df[["Q_action_0", "Q_action_1"]].max(axis=1) > 0].head(20).to_string(index=False))

    nonzero_states = np.sum(q_table != 0, axis=1) > 0
    explored_states = np.where(nonzero_states)[0]
    print(f"\n[DEBUG] Total states explored: {len(explored_states)} / {q_table.shape[0]}")
    print("[DEBUG] Best action counts:")
    print(q_df["best_action"].value_counts())

    readable_path = model_dir / f"q_table_readable_{date_str}.csv"
    q_df.to_csv(readable_path, index=False)
    print(f"[INFO] Q-table exported to {readable_path}")

    print("[DEBUG] Q-table stats:")
    print(f"Mean Q-value for action 0: {np.mean(q_table[:, 0]):.4f}")
    print(f"Mean Q-value for action 1: {np.mean(q_table[:, 1]):.4f}")
    all_states = [get_state(data.iloc[i]) for i in range(len(data))]
    print(f"[DEBUG] Unique states in data: {len(set(all_states))} / {n_states}")
    state_counts = {}
    for i in range(len(data)):
        state = get_state(data.iloc[i])
        state_counts[state] = state_counts.get(state, 0) + 1
    print(f"[DEBUG] Top 5 most frequent states: {sorted(state_counts.items(), key=lambda x: -x[1])[:5]}")

    print(data[["yhat_temp", "solar_irradiance", "yhat_flow"]].describe())


    pd.plotting.scatter_matrix(data[["temp_norm", "solar_norm", "flow_norm"]], figsize=(10, 8))
    plt.suptitle("Normalized Input Features")
    plt.show()