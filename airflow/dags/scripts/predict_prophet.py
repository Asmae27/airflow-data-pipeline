def predict_prophet(**context):
    import pandas as pd
    import pickle
    from pathlib import Path
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import os

    exec_date = context.get('logical_date', context.get('execution_date'))
    if exec_date is None:
        raise ValueError("No execution date found in context")

    exec_date_naive = pd.to_datetime(exec_date).tz_localize(None)
    date_str = exec_date_naive.strftime('%Y-%m-%d')

    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / "data" / "processed"
    model_dir = base_dir / "models"
    predictions_dir = base_dir / "data" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "temperature": "water_output_temperature",
        "flow": "flow_rate"
    }

    for name, target_col in targets.items():
        model_path = model_dir / f"prophet_model_{name}_{date_str}.pkl"
        if not model_path.exists():
            print(f"Modèle Prophet {name} manquant : {model_path}")
            continue

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        pred_start = exec_date_naive - pd.Timedelta(days=29)
        pred_end = exec_date_naive 

        all_files = sorted(processed_dir.glob("preprocessed_*.csv"))
        pred_files = []
        for f in all_files:
            file_date_str = f.stem.split("_")[-1]
            file_date = pd.to_datetime(file_date_str).tz_localize(None)
            if pred_start <= file_date <= pred_end:
                pred_files.append(f)

        if not pred_files:
            print(f"Aucune donnée trouvée pour la prédiction {name} du {pred_start} au {pred_end}")
            continue

        pred_df_list = []
        for f in pred_files:
            df = pd.read_csv(f)

            # Check datetime validity
            df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
            if df["datetime"].isna().any():
                print(f"[CORRUPTED FILE] {f}")
                print(df[df["datetime"].isna()][["datetime"]].head(10))
                raise ValueError(f"Invalid datetime values found in file: {f}")

            pred_df_list.append(df)


        pred_df = pd.concat(pred_df_list)
        pred_df["datetime"] = pd.to_datetime(pred_df["datetime"], errors='coerce')

        if pred_df["datetime"].isna().any():
            bad_rows = pred_df[pred_df["datetime"].isna()]
            print("[ERROR] Invalid datetime values found in prediction data:")
            print(bad_rows)
            raise ValueError("Found invalid datetime entries. Fix the input data before continuing.")

        
        future_df = pred_df.rename(columns={"datetime": "ds"})
        future_df['ds'] = pd.to_datetime(future_df['ds'], utc=True)
        future_df['ds'] = future_df['ds'].dt.tz_localize(None) 

        forecast = model.predict(future_df[["ds"]])
        forecast_result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        forecast_result = forecast_result.rename(columns={"ds": "datetime"})
        forecast_result = forecast_result.reset_index(drop=True)
        pred_df = pred_df.reset_index(drop=True)

        forecast_result["actual"] = pred_df[target_col]

        output_path = predictions_dir / f"prophet_forecast_{name}_{date_str}.csv"
        forecast_result.to_csv(output_path, index=False)

    
        mae = mean_absolute_error(forecast_result["actual"], forecast_result["yhat"])
        rmse = mean_squared_error(forecast_result["actual"], forecast_result["yhat"]) ** 0.5

        log_path = predictions_dir / f"prophet_scores_{name}.csv"
        log_exists = log_path.exists()
        df_list = []
        for f in pred_files:
            temp_df = pd.read_csv(f)
            temp_df["datetime"] = pd.to_datetime(temp_df["datetime"], errors="coerce")
            invalid_rows = temp_df[temp_df["datetime"].isna()]
            if not invalid_rows.empty:
                print(f"File {f.name} contains {len(invalid_rows)} invalid datetime rows")
                print(invalid_rows.head())  # optionally print a few rows for inspection
            df_list.append(temp_df)
        df = pd.concat(df_list)

        with open(log_path, "a") as log_file:
            if not log_exists:
                log_file.write("exec_date,prediction_start,prediction_end,mae,rmse\n")
            
            pred_start_str = pred_start.strftime('%Y-%m-%d')
            pred_end_str = pred_end.strftime('%Y-%m-%d')
            
            log_file.write(f"{date_str},{pred_start_str},{pred_end_str},{mae:.4f},{rmse:.4f}\n")


        print(f"Prédictions {name} enregistrées ({output_path.name}) — MAE: {mae:.4f} | RMSE: {rmse:.4f}")

