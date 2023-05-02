# mlflow

The monitoring, metadata store and model registry platform

### Usage
    Always import mlflow Client from utils
        from analytics.services.mlflow.utils import MlflowClient
        client = MlflowClient()
    
    1. Basic logging 
        with mlflow.start_run(experiment_id=<id>, run_name=<optional>) as run:
            client.log_param("parameter","test")
            client.log_metric("the_answer",42.0)
            
            client.log_artifact("./1_data_sample.csv")
    
    2. Logging artifacts in folders
        client.log_artifact("./output/image.png")
    
    3. Nested runs
        with mlflow.start_run(experiment_id=1, run_name="top_lever_run") as run:
            with mlflow.start_run(experiment_id=1, run_name="subrun1",nested=True) as subrun1:
                client.log_param("p1","red")

    4. Query every run in an experiment
        mlflow.search_runs(experiment_ids=[<experiment name>])
    
    5. Correct run metric
        with mlflow.start_run(run_id="your_run_id") as run:
            clinet.log_param("p1","your_corrected_value")
    



