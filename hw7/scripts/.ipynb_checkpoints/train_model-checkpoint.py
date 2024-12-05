import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import xgboost as xgb
# from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import os
import logging
import pickle

logging.basicConfig(
    level=logging.INFO,                                
    format="%(asctime)s - %(levelname)s - %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S",                      
    filename="logs/train.log",                              
    filemode="a"                                      
)

yc_access_key = os.getenv('YC_ACCESS_KEY')
yc_secret_key = os.getenv('YC_SECRET_KEY')
bucket_name = 'otus-fin'
endpoint_url = 'https://storage.yandexcloud.net'

def main():

    # Загрузка обучающих данных
    df = pd.read_parquet(
                        f's3://otus-fin/data/train_features/ADAUSDT.parquet',
                        engine='pyarrow',
                        storage_options={
                            'key': yc_access_key,
                            'secret': yc_secret_key,
                            'client_kwargs': {'endpoint_url': endpoint_url}
                        }
                    )
  
    df['target'] = df['close'].shift(-1) # Добавил таргет, цена следующего бара

    # Актуальные фичи на данный момент
    # f_shape = ['close', 'SMA', 'rolling_mean_10', 'rolling_min_5', 'high', 'rolling_min_20', 'rolling_min_10', 'rolling_max_20', 'rolling_max_10', 'open' ]
    f_shape = ['close', 'SMA_L', 'SMA_S', 'rolling_mean_10', 'rolling_min_5', 'high', 
       'rolling_min_20', 'rolling_min_10', 'rolling_max_20', 'rolling_max_10', 'open' ]
    df = df.dropna()
    X = df.drop(columns=['target'])[f_shape]
    y = df['target']

    train_size = 0.90  # 90% для тренировки
    
    # Определяем индекс для разбиения
    split_index = int(len(df) * train_size)
    
    # Тренировочные данные: первые 90% данных
    X_train, y_train = X.iloc[:split_index], y.iloc[:split_index]
    
    # Тестовые данные: оставшиеся 10% данных
    X_val, y_val = X.iloc[split_index:], y.iloc[split_index:]
        
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    # Оценка точности 
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)


    current_date = datetime.now().strftime("%m-%d")
    file_name = f"linear_regression_model_{current_date}.pkl"
    
    logging.info(f"{file_name} MAE (linear regression): {mae}, R²: {r2}")


    with open('models/'+file_name, 'wb') as file:
        pickle.dump(model, file)   
        

if __name__ == "__main__":
    main()