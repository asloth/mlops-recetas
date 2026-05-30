from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator
from datetime import datetime, timedelta

# Define default args with retry configuration
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id="trigger_model", 
    default_args=default_args,
    start_date=datetime(2025, 1, 1), 
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,  # Prevent multiple instances running simultaneously
) as dag:
    
    trigger_model_api = SimpleHttpOperator(
        task_id="call_model_api",
        method="GET",
        http_conn_id="model_api_conn",  # Defined in Airflow Connections
        endpoint="/train",
        headers={"Content-Type": "application/json"},
        # Use extra_options for timeout configuration
        extra_options={'timeout': 600},  # 10 minute timeout
        # Task-level timeout as backup
        #execution_timeout=timedelta(hours=2),
    )
    
    task_compare_and_deploy = SimpleHttpOperator(
        task_id="task_compare_and_deploy",
        method="POST",
        http_conn_id="prodserver",  # Defined in Airflow Connections
        endpoint="/evaluateanddeploy",
        headers={"Content-Type": "application/json"},
        # This task depends on the previous one succeeding
        depends_on_past=False,
    )
    
    trigger_model_api >> task_compare_and_deploy
