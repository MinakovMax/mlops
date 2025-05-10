from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
import numpy as np
from collections import Counter
import requests
from datetime import datetime, timedelta
import pandas_ta as ta
from dotenv import load_dotenv
import os
from fastapi.responses import JSONResponse

# Загрузка переменных окружения
load_dotenv(dotenv_path="../infra/.env", override=True)

# Безопасная установка в окружение
for var in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "MLFLOW_S3_ENDPOINT_URL"]:
    value = os.getenv(var)
    if value:
        os.environ[var] = value

if os.getenv("AWS_ACCESS_KEY_ID"):
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
if os.getenv("AWS_SECRET_ACCESS_KEY"):
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
if os.getenv("MLFLOW_S3_ENDPOINT_URL"):
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

mlflow.set_tracking_uri("http://158.160.134.123:8080")
app = FastAPI()

# Загрузка моделей при старте
xgb_model = mlflow.pyfunc.load_model(model_uri="models:/XGBoostModel@production")
lgb_model = mlflow.pyfunc.load_model(model_uri="models:/LightGBMModel@production")
cat_model = mlflow.pyfunc.load_model(model_uri="models:/CatBoostModel@production")

models = {
    "XGB": xgb_model,
    "LGBM": lgb_model,
    "CatBoost": cat_model,
}

# Функция загрузки и подготовки последних данных
def get_latest_market_features():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3650)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_index = 0
    batch_size = 500
    all_candles_data = []

    while True:
        url = (
            f"http://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/"
            f"securities/SBER/candles.json?from={start_date_str}&till={end_date_str}"
            f"&interval=60&start={start_index}"
        )
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers, timeout=15)
        data = response.json()
        candles_data = data["candles"]["data"]
        if not candles_data:
            break
        all_candles_data.extend(candles_data)
        start_index += batch_size

    df = pd.DataFrame(all_candles_data, columns=data["candles"]["columns"])
    df["begin"] = pd.to_datetime(df["begin"])
    df.set_index("begin", inplace=True)
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Индикаторы
    df["EMA_15"] = ta.ema(df["Close"], length=15)
    df["EMA_75"] = ta.ema(df["Close"], length=75)
    adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    df["ADX"] = adx["ADX_14"]
    df["DI+"] = adx["DMP_14"]
    df["DI-"] = adx["DMN_14"]
    df["RSI_14"] = ta.rsi(df["Close"], length=14)
    df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # Фичи
    df["ema_diff"] = df["EMA_15"] - df["EMA_75"]
    df["ema_15_gt_ema_75"] = (df["EMA_15"] > df["EMA_75"]).astype(int)
    df["rsi_above_70"] = (df["RSI_14"] > 70).astype(int)
    df["rsi_above_80"] = (df["RSI_14"] > 80).astype(int)
    df["rsi_below_20"] = (df["RSI_14"] < 20).astype(int)
    df["rsi_below_30"] = (df["RSI_14"] < 30).astype(int)
    df["adx_strong_trend"] = (df["ADX"] > 25).astype(int)
    df["adx_weak_trend"] = (df["ADX"] < 20).astype(int)
    df["di_up_trend"] = (df["DI+"] > df["DI-"]).astype(int)
    df["di_down_trend"] = (df["DI-"] > df["DI+"]).astype(int)
    df["strong_up_trend"] = ((df["ADX"] > 25) & (df["DI+"] > df["DI-"])).astype(int)
    df["strong_down_trend"] = ((df["ADX"] > 25) & (df["DI-"] > df["DI+"])).astype(int)

    df.dropna(inplace=True)

    features = [
        "Open", "High", "Low", "Close", "Volume",
        "EMA_15", "EMA_75", "ADX", "DI+", "DI-", "RSI_14", "ATR_14",
        "ema_diff", "ema_15_gt_ema_75",
        "rsi_above_70", "rsi_above_80", "rsi_below_20", "rsi_below_30",
        "adx_strong_trend", "adx_weak_trend",
        "di_up_trend", "di_down_trend",
        "strong_up_trend", "strong_down_trend"
    ]

    return df[features].iloc[[-1]]


@app.get("/predict")
def predict():
    features_df = get_latest_market_features()
    features = features_df.to_dict(orient="records")[0]  # одна строка, как dict
    timestamp = features_df.index[-1].isoformat()

    predictions = []
    for model in models.values():
        pred = int(model.predict(features_df)[0])
        predictions.append(pred)
    vote = Counter(predictions).most_common(1)[0][0]

    return JSONResponse({
        "timestamp": timestamp,
        "prediction": vote,
        "votes": predictions,
        "features": features
    })