import ccxt
import pandas as pd
import time
import argparse

def main(symbols, start_date, end_date, exchange='bybit', timeframe='5m'):
    '''
    загрузка данных с пом библ ccxt
    по умолчанию лимит 1000 баров, пока подстроенно для 5минутных таймфреймов по 3 дня 
    
    '''
    if exchange == 'bybit':
        exchange = ccxt.bybit()  # Можно использовать любую биржу, которая поддерживается ccxt

    start_date = exchange.parse8601(start_date)  # Укажите начальную дату и время в формате ISO 8601
    end_date = exchange.parse8601(end_date)
    if not start_date or not end_date:
        print("Неправильные даты")
        return

    if len(symbols) == 0:
        print("Нет тикерв")
        return

    since = start_date
    for symbol in symbols:
        dfs = []
        while since < end_date:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=864) # 864 - 3 дня по 24 часа по 5 минут
            since = ohlcv[-1][0]  + 300000 # это 5 минут

            df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            # df.set_index('timestamp', inplace=True)
            dfs.append(df)
            time.sleep(0.1)
        
        df_ticker = pd.concat(dfs, ignore_index=True)
        df_ticker.set_index('datetime', inplace=True)        
        df_ticker.dropna(inplace=True)
        # df_ticker.index.name = None
        df_ticker.to_parquet(f'data/{symbol}.parquet', engine='pyarrow')
    
    # Преобразование данных в DataFrame для удобства

if __name__ == "__main__":
    # Создаем парсер аргументов
    parser = argparse.ArgumentParser(description="Загрузка данных в с биржи через ccxt, выгрузка в файлы паркет")

    # Добавляем аргументы
    parser.add_argument("symbols", nargs='+', type=str, help="Список тикеров крипто с юсдт DASHUSDT")
    parser.add_argument("--start_date", type=str, help="Формат: YYYY-MM-DD 00:00:00")
    parser.add_argument("--end_date", type=str, help="Формат: YYYY-MM-DD 00:00:00")
    parser.add_argument("--exchange", type=str, default="bybit", help="Биржа, пока только bybit")
    parser.add_argument("--timeframe", type=str, default="5m", help="по умолчанию 5m")

    # Парсим аргументы
    args = parser.parse_args()

    # Передаем аргументы в основную функцию
    main(args.symbols, args.start_date, args.end_date)