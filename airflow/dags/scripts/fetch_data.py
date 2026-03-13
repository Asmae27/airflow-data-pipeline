def simulate_solar_data(exec_date):
    import pandas as pd
    import numpy as np
    import os
    import requests
    from datetime import datetime
    from astral.sun import sun
    from astral import LocationInfo
    import pytz
    date_only = exec_date.date()

    # 🌍 Position : Casablanca
    city = LocationInfo("Casablanca", "Morocco", "Africa/Casablanca", 33.57, -7.59)
    tz = pytz.timezone(city.timezone)
    s = sun(city.observer, date=date_only, tzinfo=tz)
    fajr_hour = s["dawn"].hour + s["dawn"].minute / 60

    # 🕒 Heures locales de la journée
    hours = pd.date_range(start=f"{date_only} 00:00", end=f"{date_only} 23:00", freq='H', tz=tz)
    hour_vals = hours.hour.to_numpy() #pour que hour_vals soit mutable

    start_date = date_only.strftime("%Y%m%d")

    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point"
        f"?parameters=ALLSKY_SFC_SW_DWN,ALLSKY_KT,T2M"
        f"&community=SB&longitude=-7.59&latitude=33.57"
        f"&start={start_date}&end={start_date}&format=JSON&time-standard=lst"
    )

    response = requests.get(url)
    data = response.json()
    ghi = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
    kt = data["properties"]["parameter"]["ALLSKY_KT"]
    t2m = data["properties"]["parameter"]["T2M"]

    ghi_vals = np.array([ghi.get(h.strftime("%Y%m%d%H"), np.nan) for h in hours])
    kt_vals = np.array([kt.get(h.strftime("%Y%m%d%H"), np.nan) for h in hours])
    t2m_vals = np.array([t2m.get(h.strftime("%Y%m%d%H"), np.nan) for h in hours])

    # Nettoyage
    ghi_vals = np.nan_to_num(ghi_vals, nan=0.0)
    kt_vals = np.clip(np.nan_to_num(kt_vals, nan=0.6), 0.2, 1.0)

    # ☀️ Irradiance corrigée par Kt
    irradiance = ghi_vals * kt_vals

    # 🌡️ Température ambiante basée sur T2M
    ambient_temp = t2m_vals + np.random.normal(0, 0.3, len(t2m_vals))  # bruit léger
    fallback_temp = irradiance * 0.01 + 10
    ambient_temp = np.where(np.isnan(ambient_temp), fallback_temp, ambient_temp)
    ambient_temp = np.clip(ambient_temp, -5, 45)

    # 💧 Débit d'eau (flow_rate) ajusté autour du Fajr
    flow_rate = 0.005 + 0.002 * np.sin((hour_vals - 12) * np.pi / 12)
    flow_rate += np.random.normal(0, 0.0005, len(hour_vals))
    fajr_mask = (hour_vals >= fajr_hour - 0.5) & (hour_vals <= fajr_hour + 0.5)
    flow_rate[fajr_mask] += 0.004
    flow_rate = np.clip(flow_rate, 0.002, 0.015)

    # 💧 Température d'entrée de l'eau
    water_input_temp = ambient_temp.min() + np.random.normal(0, 1.5, len(ambient_temp))
    water_input_temp = np.clip(water_input_temp, 4, 22)

    # 🔥 Température dans le réservoir
    tank_temp = (
        25 + 0.035 * irradiance
        - 0.4 * flow_rate * 1000
        + np.random.normal(0, 0.8, len(flow_rate))
    ).clip(20, 80)

    # 🚿 Température de sortie
    water_output_temp = tank_temp - np.random.uniform(1, 2.5, len(tank_temp))
    water_output_temp = np.maximum(water_output_temp, water_input_temp)
    water_output_temp = np.clip(water_output_temp, 20, 80)

    # 📊 DataFrame final
    df = pd.DataFrame({
        "datetime": hours,
        "solar_irradiance": irradiance,
        "kt": kt_vals,
        "ambient_temperature": ambient_temp,
        "flow_rate": flow_rate,
        "water_input_temperature": water_input_temp,
        "tank_temperature": tank_temp,
        "water_output_temperature": water_output_temp
    })

    # 💾 Sauvegarde CSV
    os.makedirs("dags/data/processed", exist_ok=True)
    output_path = f"dags/data/processed/simulated_solar_heating_{date_only}.csv"
    df.to_csv(output_path, index=False)

    print(f"Données simulées pour {date_only} sauvegardées dans {output_path}")
    return output_path

