import requests
import pandas as pd
import numpy as np
import pandas_ta as ta

from datetime import datetime, timedelta
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
    
import os
from dotenv import load_dotenv

# Загрузить переменные из файла .env
# load_dotenv(dotenv_path="infra/.env", override=True)
   
    
def load_and_prepare_data():
    """
    Загружает данные с MOEX (SBER), рассчитывает технические индикаторы
    и возвращает DataFrame со столбцами:
    [Open, High, Low, Close, Volume, EMA_15, EMA_75, ADX, DI+, DI-, RSI_14, ATR_14]
    и индексом datetime.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Последний год
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_index = 0
    batch_size = 500
    all_candles_data = []

    while True:
        url = (
            f'http://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/'
            f'securities/SBER/candles.json?from={start_date_str}&till={end_date_str}'
            f'&interval=60&start={start_index}'
        )
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers, timeout=15)
        
        data = response.json()
        candles_data = data['candles']['data']
        if not candles_data:
            break
        all_candles_data.extend(candles_data)
        start_index += batch_size

    df = pd.DataFrame(all_candles_data, columns=data['candles']['columns'])
    df['begin'] = pd.to_datetime(df['begin'])
    df.set_index('begin', inplace=True)
    df.rename(
        columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                 'close': 'Close', 'volume': 'Volume'},
        inplace=True
    )
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # -- Рассчитываем индикаторы --
    df['EMA_15'] = ta.ema(df['Close'], length=15)
    df['EMA_75'] = ta.ema(df['Close'], length=75)

    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx['ADX_14']
    df['DI+'] = adx['DMP_14']
    df['DI-'] = adx['DMN_14']
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    df.dropna(inplace=True)
    return df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========================
# 1. Загрузка и подготовка
# ========================
df = load_and_prepare_data()

# Сортируем по времени (индексу)
df.sort_index(inplace=True)

# Рассчитываем future_return на 10 часов вперёд
df['future_return'] = df['Close'].shift(-14) / df['Close'] - 1
df.dropna(subset=['future_return'], inplace=True)

# Формируем таргет (-1, 0, 1)
df['target'] = 0
df.loc[df['future_return'] >  0.01, 'target'] = 1
df.loc[df['future_return'] < -0.01, 'target'] = -1
df['target'] = df['target'].astype(int)

# Т.к. XGBoost/CatBoost/LGB ждут классы 0..2:
# -1 → 0,  0 → 1,  1 → 2
mapping_dict = {-1: 0, 0: 1, 1: 2}
df['target_mapped'] = df['target'].map(mapping_dict)

# 📊 Дополнительные признаки тренда и силы рынка

df['ema_diff'] = df['EMA_15'] - df['EMA_75']  
df['ema_15_gt_ema_75'] = (df['EMA_15'] > df['EMA_75']).astype(int)
df["rsi_above_70"] = (df["RSI_14"] > 70).astype(int)
df["rsi_above_80"] = (df["RSI_14"] > 80).astype(int)
df['rsi_below_20'] = (df['RSI_14'] < 20).astype(int)
df['rsi_below_30'] = (df['RSI_14'] < 30).astype(int)
df["adx_strong_trend"] = (df["ADX"] > 25).astype(int)
df["adx_weak_trend"] = (df["ADX"] < 20).astype(int)
df["di_up_trend"] = (df["DI+"] > df["DI-"]).astype(int)
df["di_down_trend"] = (df["DI-"] > df["DI+"]).astype(int)
df["strong_up_trend"] = ((df["ADX"] > 25) & (df["DI+"] > df["DI-"])).astype(int)
df["strong_down_trend"] = ((df["ADX"] > 25) & (df["DI-"] > df["DI+"])).astype(int)

ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
futr_exog_cols = ['EMA_15', 'EMA_75', 'ADX', 'DI+', 'DI-', 'RSI_14', 'ATR_14']
additional_cols = [
    'ema_diff', 'ema_15_gt_ema_75',  # Разница EMA и флаг пересечения
    'rsi_above_70', 'rsi_above_80',  # RSI выше 70 и 80
    'rsi_below_20', 'rsi_below_30',  # RSI ниже 20 и 30
    'adx_strong_trend', 'adx_weak_trend',  # Сила тренда по ADX
    'di_up_trend', 'di_down_trend',  # Направление тренда по DI+ и DI-
    'strong_up_trend', 'strong_down_trend'  # Комбинированные признаки сильного тренда
]
features = ohlcv + futr_exog_cols + additional_cols

# ====================================
# 2. Деление на Train / Val / Test
# ====================================
n = len(df)
train_end = int(n * 0.7)
val_end   = int(n * 0.9)

train_df = df.iloc[:train_end]
val_df   = df.iloc[train_end:val_end]
test_df  = df.iloc[val_end:]

X_train, y_train = train_df[features], train_df['target_mapped']
X_val,   y_val   = val_df[features],   val_df['target_mapped']
X_test,  y_test  = test_df[features],  test_df['target_mapped']

# =============================
# 3. Проверка дисбаланса классов
# =============================
print("Train class distribution (mapped):")
print(y_train.value_counts(normalize=True))

# Автоматически вычислим веса классов как 1 / (доля класса) - упрощенный вариант
# Если класс 0 встречается очень часто, его вес будет меньше, чем у редких классов.
class_counts = y_train.value_counts(normalize=True).sort_index()  # сортировка по индексу (0,1,2)
class_weights = [1.0 / freq for freq in class_counts]
print("Computed class_weights:", class_weights)

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter
import mlflow
import mlflow.sklearn

# === Настройка MLflow ===
mlflow.set_tracking_uri("http://158.160.134.123:8080")
mlflow.set_experiment("ensemble_latest_5000")

# Вычисление будущей доходности
df["future_return"] = df["Close"].shift(-14) / df["Close"] - 1
df.dropna(subset=["future_return"], inplace=True)

# Классификация: {-1, 0, 1}
df["target"] = 0
df.loc[df["future_return"] > 0.01, "target"] = 1
df.loc[df["future_return"] < -0.01, "target"] = -1
df["target_mapped"] = df["target"].map({-1: 0, 0: 1, 1: 2})

# Последние 5000 наблюдений
df = df.tail(5000).copy()
X = df.drop(columns=["target", "target_mapped", "future_return"])
y = df["target_mapped"]

# === Модели ===
models = {
    "XGB": XGBClassifier(objective="multi:softmax", max_depth=3,n_estimators=408,
        subsample=0.754, colsample_bytree=0.602, num_class=3, eval_metric="mlogloss",
        random_state=42, use_label_encoder=False,
    ),
    "LGBM": LGBMClassifier(objective="multiclass", max_depth=3, n_estimators=928,
        subsample=0.724, colsample_bytree=0.658, num_class=3, random_state=42,
    ),
    "CatBoost": CatBoostClassifier(loss_function="MultiClass", random_seed=42,
        iterations=1000, learning_rate=0.05, early_stopping_rounds=100,
        use_best_model=True, verbose=0,),
}

selected_models = list(models.keys())

with mlflow.start_run(run_name="last_5000"):
    preds = {}
    for name, model in models.items():
        if name == "CatBoost":
            model.fit(X, y, eval_set=(X, y))
        else:
            model.fit(X, y)

        y_pred = model.predict(X)
        if isinstance(y_pred, np.ndarray) and y_pred.ndim == 2:
            y_pred = np.argmax(y_pred, axis=1)
        preds[name] = y_pred

        # === Логирование модели ===
        with mlflow.start_run(run_name=f"{name}_model", nested=True):
            mlflow.log_param("model_type", name)
            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", accuracy_score(y, y_pred))
            mlflow.log_metric("f1_macro", f1_score(y, y_pred, average="macro"))
            mlflow.log_metric("precision_macro", precision_score(y, y_pred, average="macro"))
            mlflow.log_metric("recall_macro", recall_score(y, y_pred, average="macro"))

            # Логирование модели
            if name == "XGB": mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name="XGBoostModel")
            elif name == "LGBM": mlflow.lightgbm.log_model(model, artifact_path="model", registered_model_name="LightGBMModel")
            elif name == "CatBoost": mlflow.catboost.log_model(model, artifact_path="model", registered_model_name="CatBoostModel")

    # === Ансамбль ===
    final_preds = []
    for i in range(len(X)):
        votes = [int(preds[name][i]) for name in selected_models]
        vote_count = Counter(votes).most_common()
        if vote_count[0][1] > 1:
            final_preds.append(vote_count[0][0])
        else:
            final_preds.append(np.random.choice([vote_count[0][0], vote_count[1][0]]))

    acc = accuracy_score(y, final_preds)
    f1 = f1_score(y, final_preds, average="macro")
    prec = precision_score(y, final_preds, average="macro")
    rec = recall_score(y, final_preds, average="macro")
    correct_frac = (np.sum((np.array(final_preds) == 0) & (y.values == 0)) + np.sum((np.array(final_preds) == 2) & (y.values == 2))) / len(y)

    mlflow.log_metric("ensemble_accuracy", acc)
    mlflow.log_metric("ensemble_f1_macro", f1)
    mlflow.log_metric("ensemble_precision_macro", prec)
    mlflow.log_metric("ensemble_recall_macro", rec)
    mlflow.log_metric("ensemble_correct_frac_0_2", correct_frac)

    df_preds = pd.DataFrame({"index": X.index, "true": y.values,"ensemble_pred": final_preds})
    pred_path = "ensemble_last_5000_predictions.csv"
    df_preds.to_csv(pred_path, index=False)
    mlflow.log_artifact(pred_path, artifact_path="ensemble_preds")

mlflow.end_run()