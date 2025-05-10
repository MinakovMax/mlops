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

# Загрузка переменных окружения
load_dotenv(dotenv_path="../infra/.env", override=True)
# Принудительно задаем переменные окружения
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

mlflow.set_tracking_uri("http://158.160.134.123:8080")

model = mlflow.pyfunc.load_model("models:/XGBoostModel@production")
print("Модель успешно загружена!")