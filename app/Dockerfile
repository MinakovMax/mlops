################### builder ###################
FROM python:3.10 AS builder
WORKDIR /app

# Системные build-зависимости
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

COPY requirements.txt .
# requirements.txt теперь содержит numpy<2  и НЕТ mlflow-skinny
RUN python -m pip install --prefix=/opt/deps --no-cache-dir -r requirements.txt

COPY . .

################### runtime ###################
FROM python:3.10-slim AS runtime
WORKDIR /app

RUN pip install boto3

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/deps /opt/deps
COPY --from=builder /app /app

# Патчинг pandas_ta: заменяем устаревший импорт NaN → nan
RUN sed -i "/from numpy import NaN as npNaN/c\import numpy as np\nnpNaN = np.nan" \
    /opt/deps/lib/python3.10/site-packages/pandas_ta/momentum/squeeze_pro.py || true && \
    sed -i "/from numpy import NaN as npNaN/c\import numpy as np\nnpNaN = np.nan" \
    /opt/deps/lib/python3.10/site-packages/pandas_ta/volatility/kc.py || true

ENV PYTHONPATH=/opt/deps/lib/python3.10/site-packages
ENV PATH=/opt/deps/bin:$PATH

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9090"]
