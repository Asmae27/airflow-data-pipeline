def preprocess_data(**context):
    import pandas as pd
    import os
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

    exec_date = context.get('logical_date', context.get('execution_date'))
    if exec_date is None:
        raise ValueError("No execution date found in context")
    
    date_str = exec_date.strftime('%Y-%m-%d')
    raw_path = f"dags/data/processed/simulated_solar_heating_{date_str}.csv"
    output_path = f"dags/data/processed/preprocessed_{date_str}.csv"

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Fichier manquant : {raw_path}. Veuillez d'abord exécuter fetch_data pour cette date.")

    df = pd.read_csv(raw_path, parse_dates=["datetime"])
    df = df.dropna()
    
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    def get_saison(month):
        if month in [12, 1, 2]:
            return "hiver"
        elif month in [3, 4, 5]:
            return "printemps"
        elif month in [6, 7, 8]:
            return "été"
        else:
            return "automne"
    
    df["saison"] = df["month"].apply(get_saison)

    scaler = MinMaxScaler()
    num_features = [
        "solar_irradiance", "ambient_temperature", "water_input_temperature",
        "flow_rate", "tank_temperature", "water_output_temperature",
        "hour", "day_of_week", "month"
    ]
    df[num_features] = scaler.fit_transform(df[num_features])

    encoder = OneHotEncoder(sparse_output=False)
    saison_encoded = encoder.fit_transform(df[["saison"]])
    saison_df = pd.DataFrame(saison_encoded, columns=encoder.get_feature_names_out(["saison"]))
    df = pd.concat([df.drop(columns=["saison"]), saison_df], axis=1)

    os.makedirs("dags/data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)

    return output_path