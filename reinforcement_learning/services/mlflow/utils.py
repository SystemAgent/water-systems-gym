from mlflow.tracking import MlflowClient
from mlflow.tracking._tracking_service.client import TrackingServiceClient
from mlflow.utils.uri import add_databricks_profile_info_to_artifact_uri
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository


sftp_artifact_uri = 'sftp://developer@prod4.yatrusanalytics.com/home/developer/mlflow/artifact_storage'


def my_get_artifact_repo(self, run_id):
    print('calling art repo')
    run = self.get_run(run_id)
    current_artifact_uri = run.info.artifact_uri
    final_artifact_uri = sftp_artifact_uri + \
        current_artifact_uri.replace('/home/stageai/artifact_storage', '')
    artifact_uri = add_databricks_profile_info_to_artifact_uri(
        final_artifact_uri, self.tracking_uri
    )
    print(final_artifact_uri)
    return get_artifact_repository(artifact_uri)


TrackingServiceClient._get_artifact_repo = my_get_artifact_repo
