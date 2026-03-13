def train_prophet(**context):
    import pandas as pd
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    from pathlib import Path
    import pickle
    import os
    from datetime import timedelta
    from itertools import product

    def optimize_prophet_model(df, param_grid=None, horizon='7 days', initial='21 days', period='7 days', verbose=True):
        if param_grid is None:
            param_grid = {
                'changepoint_prior_scale': [0.01, 0.05, 0.1],
                'seasonality_prior_scale': [1.0, 5.0, 10.0],
                'seasonality_mode': ['additive', 'multiplicative']
            }

        all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
        results = []

        for i, params in enumerate(all_params):
            if verbose:
                print(f"Test {i+1}/{len(all_params)}: {params}")

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                **params
            )
            model.add_seasonality(name='hourly', period=24, fourier_order=8)

            try:
                model.fit(df)
                df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon, parallel="processes")
                df_p = performance_metrics(df_cv, rolling_window=1)
                rmse = df_p['rmse'].values[0]
                results.append((params, rmse, model))
            except Exception as e:
                print(f"Erreur pour {params}: {e}")
                continue

        best = min(results, key=lambda x: x[1])
        best_params, best_rmse, best_model = best

        results_df = pd.DataFrame([{
            **params, 'rmse': rmse
        } for params, rmse, _ in results]).sort_values(by='rmse')

        if verbose:
            print("\nMeilleurs hyperparamètres :")
            print(best_params)
            print(f"RMSE = {best_rmse:.4f}")

        return best_model, results_df

    exec_date = context.get('logical_date', context.get('execution_date'))
    if exec_date is None:
        raise ValueError("No execution date found in context")
    exec_date = exec_date.date()
    date_str = exec_date.strftime("%Y-%m-%d")

    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / "data" / "processed"
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)

    train_start = pd.to_datetime("2023-01-01").date()
    train_end = (exec_date - timedelta(days=30))

    if train_end <= train_start:
        raise ValueError(f"Not enough data to train. train_start={train_start}, train_end={train_end}")

    print(f"Training from {train_start} to {train_end}")

    all_files = sorted(processed_dir.glob("preprocessed_*.csv"))
    training_files = [f for f in all_files if train_start <= pd.to_datetime(f.stem.split("_")[-1]).date() <= train_end]
    if not training_files:
        raise ValueError(f"Aucune donnée d'entraînement entre {train_start} et {train_end}.")

    df_list = [pd.read_csv(f) for f in training_files]
    df = pd.concat(df_list)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]) 
    
    # entraînement du modèle température 
    temp_df = df.rename(columns={"datetime": "ds", "water_output_temperature": "y"})
    temp_df['ds'] = pd.to_datetime(temp_df['ds'], errors='coerce')  # convert to datetime, NaT if invalid
    temp_df = temp_df.dropna(subset=['ds'])                        # drop rows with invalid datetime
    temp_df['ds'] = temp_df['ds'].dt.tz_localize(None)             # remove timezone info


    best_model_temp, scores_temp = optimize_prophet_model(temp_df)
    with open(model_dir / f"prophet_model_temperature_{date_str}.pkl", "wb") as f:
        pickle.dump(best_model_temp, f)
    scores_temp.to_csv(model_dir / f"scores_temperature_{date_str}.csv", index=False)

    # entraînement du modèle débit
    flow_df = df.rename(columns={"datetime": "ds", "flow_rate": "y"})
    flow_df['ds'] = pd.to_datetime(flow_df['ds'], errors='coerce')
    flow_df = flow_df.dropna(subset=['ds'])
    flow_df['ds'] = flow_df['ds'].dt.tz_localize(None)


    best_model_flow, scores_flow = optimize_prophet_model(flow_df)
    with open(model_dir / f"prophet_model_flow_{date_str}.pkl", "wb") as f:
        pickle.dump(best_model_flow, f)
    scores_flow.to_csv(model_dir / f"scores_flow_{date_str}.csv", index=False)

    print(f"Modèles Prophet optimisés et sauvegardés pour la date {date_str}")
    print(f"[DEBUG] Training data date range: {temp_df['ds'].min()} → {temp_df['ds'].max()} (n={len(temp_df)})")

