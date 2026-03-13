import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt

def test_make_decision(start_date_str, nb_days):
    base_dir = Path(__file__).parent.parent
    results = []

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

    temp_bins = np.linspace(0, 80, 8)
    flow_bins = np.linspace(0, 0.015, 6)
    solar_bins = np.linspace(0, 1000, 10)

    def discretize(value, bins):
        idx = np.digitize(value, bins) - 1
        return max(0, min(idx, len(bins) - 2))

    def get_state(row):
        t = discretize(row["yhat_temp"], temp_bins)
        f = discretize(row["yhat_flow"], flow_bins)
        s = discretize(row["solar_irradiance"], solar_bins)
        return t * len(flow_bins) * len(solar_bins) + f * len(solar_bins) + s

    for i in range(nb_days):
        date = start_date + timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")

        q_table_path = base_dir / "models" / f"q_table_{date_str}.pkl"
        forecast_temp_path = base_dir / "data" / "predictions" / f"prophet_forecast_temperature_{date_str}.csv"
        forecast_flow_path = base_dir / "data" / "predictions" / f"prophet_forecast_flow_{date_str}.csv"
        irradiance_path = base_dir / "data" / "processed" / f"preprocessed_{date_str}.csv"

        if not (q_table_path.exists() and forecast_temp_path.exists() and forecast_flow_path.exists() and irradiance_path.exists()):
            print(f"⚠️ Données manquantes pour {date_str}, saut.")
            continue

        with open(q_table_path, "rb") as f:
            q_table = pickle.load(f)

        df_temp = pd.read_csv(forecast_temp_path)
        df_flow = pd.read_csv(forecast_flow_path)
        df_irr = pd.read_csv(irradiance_path)[["datetime", "solar_irradiance"]]

        df = pd.merge(df_temp, df_flow, on="datetime", suffixes=("_temp", "_flow"))
        df = pd.merge(df, df_irr, on="datetime")

        last_row = df.iloc[-1]

        state = get_state(last_row)
        action = int(np.argmax(q_table[state]))

        results.append({
            "date": date_str,
            "temp": last_row["yhat_temp"],
            "flow": last_row["yhat_flow"],
            "irradiance": last_row["solar_irradiance"],
            "action": action
        })

    return pd.DataFrame(results)

# Exemple d’utilisation:
# df_results = test_make_decision("2025-07-01", 15)
# print(df_results)

def plot_decisions(df_results):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(df_results['date'], df_results['temp'], 'r-', label='Température (°C)')
    ax1.set_ylabel('Température (°C)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    plt.xticks(rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(df_results['date'], df_results['irradiance'], 'b--', label='Irradiance (W/m²)')
    ax2.set_ylabel('Irradiance (W/m²)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # Affichage des actions sur un troisième axe
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.15))
    ax3.plot(df_results['date'], df_results['action'], 'go-', label='Action (0=off,1=on)')
    ax3.set_ylabel('Action')
    ax3.set_yticks([0,1])
    ax3.tick_params(axis='y', labelcolor='g')

    plt.title("Décisions Q-learning selon Température et Irradiance")
    fig.tight_layout()
    plt.show()

# Exemple d’utilisation:
# plot_decisions(df_results)
