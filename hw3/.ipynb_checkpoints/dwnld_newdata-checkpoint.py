import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import argparse
import logging
import os

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

def get_dates():
    current_time = datetime.utcnow()
    # Округляем до 5 минут в меньшую сторону
    rounded_time = (current_time - timedelta(minutes=current_time.minute % 5,
                                            seconds=current_time.second,
                                            microseconds=current_time.microsecond))
    next_timeframe = rounded_time + timedelta(minutes=5)
        
    with open('date.txt', 'r+') as file:
        date_str = file.read().strip()
        file.seek(0)
        file.truncate()
        file.write(next_timeframe.strftime('%Y-%m-%dT%H:%M:%SZ'))
        
    return date_str, rounded_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
def main(symbols, exchange='bybit', timeframe='5m'):
    
    start_date, end_date = get_dates()
    
    object_name = f'{end_date}.parquet'
    '''
    загрузка данных с пом библ ccxt
    по умолчанию лимит 1000 баров, пока подстроенно для 5минутных таймфреймов по 3 дня 
    
    '''
    if exchange == 'bybit':
        exchange = ccxt.bybit()  # Можно использовать любую биржу, которая поддерживается ccxt

    start_date = exchange.parse8601(start_date)  # Укажите начальную дату и время в формате ISO 8601
    end_date = exchange.parse8601(end_date)
    if not start_date or not end_date:
        logging.info(f"Неправильные даты")
        return

    if len(symbols) == 0:
        logging.info(f"Нет тикерв")
        return

    since = start_date
    for symbol in symbols:
        logging.info(f"Processing {symbol}")
        dfs = []
        since = start_date
        while since < end_date:
            try:                
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=864) # 864 - 3 дня по 24 часа по 5 минут
            except Exception as e:
                logging.error(f"Ошибка при обработке символа {symbol}: {e}")
                break
                
            since = ohlcv[-1][0]  + 300000 # это 5 минут

            df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            # df.set_index('timestamp', inplace=True)
            dfs.append(df)
            time.sleep(0.2)
        
        df_ticker = pd.concat(dfs, ignore_index=True)
        df_ticker.set_index('datetime', inplace=True)        
        df_ticker.dropna(inplace=True)
        # df_ticker.index.name = None
        df_ticker.to_parquet(
                        f's3://{bucket_name}/data/train/{symbol}/{object_name}',
                        engine='pyarrow',
                        storage_options={
                            'key': yc_access_key,
                            'secret': yc_secret_key,
                            'client_kwargs': {'endpoint_url': 'https://storage.yandexcloud.net'}
                        }
                    )
    
    # Преобразование данных в DataFrame для удобства

if __name__ == "__main__":
    # Создаем парсер аргументов
    parser = argparse.ArgumentParser(description="Загрузка данных в с биржи через ccxt, выгрузка в файлы паркет")

    # Добавляем аргументы
    parser.add_argument("symbols", nargs='+', type=str, help="Список тикеров крипто с юсдт DASHUSDT")
    #parser.add_argument("--start_date", type=str, help="Формат: YYYY-MM-DD 00:00:00")
    #parser.add_argument("--end_date", type=str, help="Формат: YYYY-MM-DD 00:00:00")
    parser.add_argument("--exchange", type=str, default="bybit", help="Биржа, пока только bybit")
    parser.add_argument("--timeframe", type=str, default="5m", help="по умолчанию 5m")

    # Парсим аргументы
    args = parser.parse_args()

    # Передаем аргументы в основную функцию
    main(args.symbols) #, args.start_date, args.end_date