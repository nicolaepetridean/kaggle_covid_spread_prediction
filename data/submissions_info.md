|ID| date | local score |  LB score | Model | Comments|
|--|------|:-----------:|----------:|------:|----------------:|
|1 | 03-04-2030 | 0.22674  | 0.22674 |  ARIMA Forecasting model|  no retraining, **future is used** |
|2 | 04-04-2020 | - | 1.02140 | ML models | per state/different models for different states, default params |
|3 | 04-04-2020 | - | 0.20091 | XGBoost | per country/1000 estimators, **future is used** |
|4 | 05-04-2020 | 0.337 | 0.87987 | Keras LSTM | common model for all countries|
