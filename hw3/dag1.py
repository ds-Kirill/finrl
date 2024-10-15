from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
from airflow.models import Variable

default_args = {
    'start_date': datetime(2024, 1, 1),
    'retries': 3,
}

yc_access_key = Variable.get("YC_ACCESS_KEY")
yc_secret_key = Variable.get("YC_SECRET_KEY")

with DAG('python_download', default_args=default_args, schedule_interval='@weekly') as dag:

    run_script = BashOperator(
        task_id='run_script',
        bash_command='source /home/ubuntu/miniconda3/bin/activate myenv && python /home/ubuntu/trading/scripts/dwnld_newdata.py DASHUSDT FILUSDT APTUSDT DOTUSDT SOLUSDT XRPUSDT ETHUSDT APEUSDT ADAUSDT BNXUSDT SUIUSDT BTCUSD',
        env={
            'YC_ACCESS_KEY': yc_access_key,
            'YC_SECRET_KEY': yc_secret_key
        }
    )

    run_script
