#!/bin/bash
echo "Обновляем данные..."
python scripts/dwnld_newdata.py DASHUSDT FILUSDT APTUSDT DOTUSDT SOLUSDT XRPUSDT ETHUSDT APEUSDT ADAUSDT BNXUSDT SUIUSDT BTCUSD

echo "Создаем фичи и датасет..."
python scripts/add_features.py

echo "Обучаем модель..."
python scripts/train_model.py

echo "Запускаем торгового робота..."
python scripts/trader.py

