import os
import requests
import json

import plotly.express as px
import mlflow
from analytics.services.mlflow.utils import MlflowClient
from analytics.config import MLFLOW_USER, MLFLOW_PASSWORD, MLFLOW_TRACKING_URI


os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_USER
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_PASSWORD


def basic_example():
    client = MlflowClient()
    # experiment_id = client.create_experiment("New Experiment")
    df = px.data.iris()

    # sample CSV file
    df.to_csv("1_data_sample.csv")

    # sample pandas HTML file
    df.to_html("2_data_sample.html")

    # sample image
    r = requests.get(
        "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png")
    with open("3_image_sample.png", 'wb') as f:
        f.write(r.content)

    # sample gif
    r = requests.get("https://media1.giphy.com/media/bU3YVJAAXckCI/giphy.gif")
    with open("4_gif_sample.gif", 'wb') as f:
        f.write(r.content)

    # sample plotly plot - HTML
    fig = px.scatter(df, x="sepal_width", y="sepal_length",
                     color="species", marginal_y="rug", marginal_x="histogram")
    fig.write_html("5_plot_sample.html")

    # sample geojson
    with open("6_map_sample.geojson", "w+") as f:
        data = requests.get(
            "https://gist.githubusercontent.com/wavded/1200773/raw/e122cf709898c09758aecfef349964a8d73a83f3/sample.json").json()
        f.write(json.dumps(data))

    # get experiment
    experiment = client.get_experiment_by_name("New Experiment")

    # log files to mlflow experiment
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:

        client.log_param(run.info.run_id, "parameter", "test")
        client.log_metric(run.info.run_id, "the_answer", 42.0)

        client.log_artifact(run.info.run_id, "./1_data_sample.csv")
        client.log_artifact(run.info.run_id, "./2_data_sample.html")
        client.log_artifact(run.info.run_id, "./3_image_sample.png")
        client.log_artifact(run.info.run_id, "./4_gif_sample.gif")
        client.log_artifact(run.info.run_id, "./5_plot_sample.html")
        client.log_artifact(run.info.run_id, "./6_map_sample.geojson")


def download_example(run_id):
    client = MlflowClient()
    # get experiment
    experiment = client.get_experiment_by_name("New Experiment")

    # Download artifacts
    client = MlflowClient()
    local_dir = "artifacts"
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)
    local_path = client.download_artifacts(
        run_id, "2_data_sample.html", local_dir)
    print("Artifacts downloaded in: {}".format(local_path))


if __name__ == '__main__':
    # basic_example()
    download_example('d61b26c8e0d2401b814ce9fccc71516b')
