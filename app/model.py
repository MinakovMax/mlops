import mlflow.pyfunc

# Загружаем модели по зарегистрированным именам
xgb_model = mlflow.pyfunc.load_model(model_uri="models:/XGBoostModel@production")
lgb_model = mlflow.pyfunc.load_model(model_uri="models:/LightGBMModel@production")
cat_model = mlflow.pyfunc.load_model(model_uri="models:/CatBoostModel@production")

models = {
    "XGB": xgb_model,
    "LGBM": lgb_model,
    "CatBoost": cat_model,
}