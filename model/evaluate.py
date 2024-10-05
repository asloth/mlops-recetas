import mlflow
from mlflow.tracking import MlflowClient

from fastapi import FastAPI


def compare_and_deploy():

    remote_server_uri = "http://mlflow-server:5000"  # set to your server URI
    mlflow.set_tracking_uri(remote_server_uri)
    client = MlflowClient()
    # Retrieve all versions of your registered model "LLM"
    registered_model_name = "PhiModel"
    versions = client.search_model_versions(f"name='{registered_model_name}'")

    # Compare the versions based on a specific metric (e.g., accuracy)
    model_performance = []

    print("Obtaining versions performance:")
    for version in versions:
        run_id = version.run_id
        print(run_id)
        run_info = client.get_run(run_id)
        tl = run_info.data.metrics["train_loss"]
        model_performance.append(
            {"version": version.version, "train_loss": tl, "run_id": run_id}
        )

    print("Sorting")
    # Sort by the metric (e.g., accuracy)
    model_performance = sorted(
        model_performance, key=lambda x: x["train_loss"], reverse=False
    )

    print(model_performance)
    print("Getting the best version")
    # Get the best version
    best_model = model_performance[0]
    print(
        f"Best model version: {best_model['version']} with train_loss: {best_model['train_loss']}"
    )

    best_version = best_model["version"]

    # Transition the best version to production
    client.transition_model_version_stage(
        name=registered_model_name, version=best_version, stage="Production"
    )

    print(f"Model version {best_version} is now in production.")


app = FastAPI()


@app.post("/evaluateanddeploy")
async def trainn():
    compare_and_deploy()
    return {"message": "Deployed"}


@app.get("/")
async def root():
    return {"message": "Holi soy el servicio que evalua y despliega"}
