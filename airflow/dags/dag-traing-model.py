from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator
from datetime import datetime

with DAG(
    dag_id="trigger_model", start_date=datetime(2023, 1, 1), schedule_interval="@daily"
) as dag:
    trigger_model_api = SimpleHttpOperator(
        task_id="call_model_api",
        method="GET",
        http_conn_id="model_api_conn",  # Defined in Airflow Connections
        endpoint="/train",
        headers={"Content-Type": "application/json"},
    )
