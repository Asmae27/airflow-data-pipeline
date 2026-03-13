import sys
import os
sys.path.append(os.path.dirname(__file__))

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from scripts.fetch_data import simulate_solar_data
from scripts.preprocess import preprocess_data
from scripts.train_q_learning import run_q_learning
from scripts.decision import make_decision
from scripts.train_prophet import train_prophet
from scripts.predict_prophet import predict_prophet

default_args = {
    'owner': 'ensam',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

def fetch_and_simulate(**context):
    exec_date = context['logical_date']
    simulate_solar_data(exec_date)



with DAG(
    dag_id='solar_ai_pipeline_2',
    default_args=default_args,
    description='Pipeline intelligent de chauffe-eau solaire',
    schedule='@daily',
    catchup=True,
    tags=['solar', 'ai', 'prophet']
) as dag:

    task_fetch = PythonOperator(
        task_id='fetch_data',
        python_callable=fetch_and_simulate,
             )

    task_preprocess = PythonOperator(
        task_id='preprocess',
        python_callable=preprocess_data,
    )

    train_prophet_task = PythonOperator(
        task_id='train_prophet',
        python_callable=train_prophet,
        
    )


    predict_prophet_task = PythonOperator(
        task_id='predict_prophet',
        python_callable=predict_prophet,
        
    )

    

    task_q_learning = PythonOperator(
        task_id='train_rl',
        python_callable=run_q_learning,
        
    )

    task_decide = PythonOperator(
        task_id='decide',
        python_callable=make_decision,
        
    )

    task_fetch >> task_preprocess >> train_prophet_task >> predict_prophet_task >> task_q_learning >> task_decide