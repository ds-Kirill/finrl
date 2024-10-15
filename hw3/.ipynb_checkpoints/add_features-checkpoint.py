import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import IsolationForest
import boto3
import argparse
import logging
import os
import gc

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler()  # Вывод в консоль
    ]
)

yc_access_key = os.getenv('YC_ACCESS_KEY')
yc_secret_key = os.getenv('YC_SECRET_KEY')
bucket_name = 'otus-fin'
endpoint_url = 'https://storage.yandexcloud.net'
parent_folder = 'data/train/'
new_folder = 'data/train_features'

def main():
    session = boto3.session.Session()
    s3_client = session.client(
    service_name='s3',
    endpoint_url=endpoint_url,
    aws_access_key_id=yc_access_key,
    aws_secret_access_key=yc_secret_key,
    )

    # получаем список папок в трейн = сисок активов
    result = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=parent_folder, Delimiter='/')
    folders = [folder.get('Prefix') for folder in result.get('CommonPrefixes', [])]

    for folder in folders:
        symbol = folder.split('/')[-2]
        logging.info(f"Processing {symbol}")
        
        #список всех файлов с данными по активу
        result = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder)
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

        # данные все собраны можно делать фичи

        from feature_engineering import Features
        feature_engineer = Features(combined_df)

        combined_df = feature_engineer.price_based_features()
        combined_df = feature_engineer.volume_based_features()
        combined_df = feature_engineer.volatility_features(window=15)
        combined_df = feature_engineer.momentum_features(window=15)
        combined_df = feature_engineer.trend_features(window=14)
        combined_df = feature_engineer.lag_rolling_features(lags=[1, 2, 3], windows=[5, 10, 20])
        combined_df = feature_engineer.statistical_features(window=20)
        combined_df = feature_engineer.time_based_features()
        combined_df = feature_engineer.fill(True, True)  # ffill and bfill

        combined_df.to_parquet(
                        f's3://{bucket_name}/{new_folder}/{symbol}.parquet',
                        engine='pyarrow',
                        storage_options={
                            'key': yc_access_key,
                            'secret': yc_secret_key,
                            'client_kwargs': {'endpoint_url': endpoint_url}
                        }
                    )
        
if __name__ == "__main__":
    main()
