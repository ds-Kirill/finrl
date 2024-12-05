import pandas as pd
import numpy as np
from time import sleep
from datetime import datetime, timedelta
import warnings
# from sklearn.ensemble import IsolationForest
import boto3
import argparse
import logging
import os
import gc
import pickle
from pybit.unified_trading import HTTP
from feature_engineering import Features
from pybit.unified_trading import WebSocket
from sklearn.linear_model import LinearRegression

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler()  # Вывод в консоль
    ]
)

yc_access_key = os.getenv('YC_ACCESS_KEY')
yc_secret_key = os.getenv('YC_SECRET_KEY')
api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')

	
bucket_name = 'otus-fin'
endpoint_url = 'https://storage.yandexcloud.net'
# f_shape_pl = ['open', 'high', 'low', 'close', 'volume', 'rolling_mean_10', 'rolling_min_5', 'rolling_min_20', 'rolling_min_10', 'rolling_max_20', 'rolling_max_10']

session = HTTP(
    testnet=False,
    api_key=api_key,
    api_secret=api_secret,
    )
    
with open('models/linear_regression_model-act.pkl', 'rb') as file:
    	loaded_model = pickle.load(file)
    	
f_shape = ['close', 'SMA_L', 'SMA_S', 'rolling_mean_10', 'rolling_min_5', 'high', 
       'rolling_min_20', 'rolling_min_10', 'rolling_max_20', 'rolling_max_10', 'open' ]

session_boto = boto3.session.Session()
s3_client = session_boto.client(
service_name='s3',
endpoint_url=endpoint_url,
aws_access_key_id=yc_access_key,
aws_secret_access_key=yc_secret_key,
)

result = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='data/train/ADAUSDT/')
parquet_files = [obj['Key'] for obj in result.get('Contents', []) if obj['Key'].endswith('.parquet')]

dfs = []

for pf in parquet_files:
    df = pd.read_parquet(
        f's3://otus-fin/{pf}',
        engine='pyarrow',
        storage_options={
            'key': yc_access_key,
            'secret': yc_secret_key,
            'client_kwargs': {'endpoint_url': endpoint_url}
            }
        )
    dfs.append(df)

combined_df = pd.concat(dfs).drop_duplicates()

del dfs
del df
gc.collect()
    
def place_order(ticker, side, qty, tpprice, slprice):
    order = session.place_order(
    category="linear",
    symbol=ticker,
    side=side,
    orderType="Market",
    # timeInForce="PostOnly",
    qty=qty,
    takeProfit= tpprice,
    stopLoss = slprice, 
    isLeverage=0,
    tpOrderType = 'Market',
    slOrderType = 'Market',
    tpslMode = 'Full'
    # orderFilter="Order",
    )

    return order

def add_features(df):
    df_features = df.copy()

    feature_engineer = Features(df_features)
    df_features = feature_engineer.momentum_features()
    df_features = feature_engineer.lag_rolling_features(lags=[1, 2, 3], windows=[5, 10, 20])
    df_features = feature_engineer.fill(True, True)
    df_features = df_features.dropna()

    return df_features

def make_prediction(df):
    predictions = loaded_model.predict(df)

    return predictions[-1]


def is_pos(ticker):
    pos = session.get_positions(
    category="linear",
    symbol=ticker,
    )

    if len(pos['result']['list'][0]['side']) == 0: #, 
        return False
    else:
        return True

def main(): #ticker    

    def handle_message(message):
        
        global combined_df
        
        if message.get('data') and message['data'][0].get('confirm') == True:
            print (message)
            try:
                cols = ['datetime', 'open', 'high',	'low', 'close',	'volume']
                datetime = message['data'][0]['start']
                openn = message['data'][0]['open']
                high = message['data'][0]['high']
                low = message['data'][0]['low']
                close = message['data'][0]['close']
                volume = message['data'][0]['volume']       
                ndf = pd.DataFrame([[datetime, openn, high, low, close, volume]], columns=cols)
            except (TypeError, IndexError, AttributeError) as e:
                print (e)
    
    
            ndf['datetime'] = pd.to_datetime(ndf['datetime'], unit='ms')
            ndf.set_index('datetime', inplace=True)
            for col in ndf.columns:
                ndf[col] =  pd.to_numeric(ndf[col])
            combined_df = pd.concat([combined_df, ndf]) # подклеил новые данные и отправил предсказывать    
            if is_pos("ADAUSDT") == False:
                df_features = add_features(combined_df[-105:])
                prediction = make_prediction(df_features[f_shape])
                if df_features['SMA_S'].iloc[-1] < df_features['SMA_L'].iloc[-1]:
                    if df_features['high'].iloc[-2] < df_features['SMA_S'].iloc[-2]: # проверить ХАЙ в датасете, возможно его там нет
                        if (df_features['close'].iloc[-1] > df_features['close'].iloc[-2]) & (prediction > df_features['close'].iloc[-1]):
                            tpprice = float(close) * 1.31
                            slprice = float(df_features['close'][-4:].min()) * 0.98
                            try:
                                order = place_order("ADAUSDT", "Buy", 20, tpprice, slprice)
                                print(f"BUY!!! {order}")
                            except (TypeError, IndexError, AttributeError) as e:
                                print (e)
            else:
                print("POS!!", session.get_positions(category="linear", symbol="ADAUSDT",))

    ws_spot = WebSocket(testnet=False, channel_type="linear")
        
    ws_spot.kline_stream(
        interval=5,
        symbol="ADAUSDT",
        callback=handle_message
    )
    
    while True:
        sleep(30)

if __name__ == "__main__":
    main()
