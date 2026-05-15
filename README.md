# 🔥 FWI-India: Wildfire Prediction for the Indian Subcontinent

A machine learning pipeline that proves the **Canadian Fire Weather Index (FWI)** can be used to predict wildfires in India.

---

## What it does

1. **Downloads** ERA5-Land weather data from Copernicus CDS (temp, humidity, wind, rainfall)
2. **Fetches** historical fire locations from NASA FIRMS (MODIS + VIIRS satellites)
3. **Maps** each fire point to the nearest ERA5 grid cell
4. **Calculates** all 6 FWI components (FFMC, DMC, DC, ISI, BUI, FWI)
5. **Engineers** 19 interaction features tuned for India's climate
6. **Trains & compares** 6 models — Random Forest, XGBoost, ANN, ST-GNN ,vanilla LSTM and a novel FWI_gated_LSTM
