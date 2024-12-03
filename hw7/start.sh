#!/bin/bash
echo "Обновляем данные..."
source /home/ubuntu/miniconda3/bin/activate algo_trade && python /home/ubuntu/scripts/dwnld_newdata.py DASHUSDT FILUSDT APTUSDT DOTUSDT SOLUSDT XRPUSDT ETHUSDT APEUSDT ADAUSDT BNXUSDT SUIUSDT BTCUSD

echo "Создаем фичи и датасет..."
source /home/ubuntu/miniconda3/bin/activate myenv && python /home/ubuntu/scripts/add_features.py

echo "Обучаем модель..."
source /home/ubuntu/miniconda3/bin/activate myenv && python /home/ubuntu/scripts/train_model.py

echo "Запускаем торгового робота..."
source /home/ubuntu/miniconda3/bin/activate myenv && python /home/ubuntu/scripts/trader.py

