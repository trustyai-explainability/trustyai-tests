import pytest
import yaml
from ocp_resources.configmap import ConfigMap
from ocp_resources.namespace import Namespace
from ocp_resources.resource import get_client
from ocp_resources.service_account import ServiceAccount

from resources.inference_service import InferenceService
from resources.serving_runtime import ServingRuntime
from resources.storage.minio_pod import MinioPod
from resources.storage.minio_secret import MinioSecret
from resources.storage.minio_service import MinioService
from resources.trustyai_service import TrustyAIService
from tests.utils import wait_for_model_pods_registered
from utilities.constants import (
    TRUSTYAI_SERVICE,
    MINIO_IMAGE,
    OVMS_RUNTIME,
    OVMS,
    OPENVINO_MODEL_FORMAT,
    ONNX,
    OVMS_QUAY_IMAGE,
)


@pytest.fixture(scope="session")
def client():
    yield get_client()


@pytest.fixture(scope="session")
def model_namespace(client):
    with Namespace(
        client=client,
        name="test-namespace",
        label={"modelmesh-enabled": "true"},
        delete_timeout=600,
    ) as ns:
        ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=120)
        yield ns


@pytest.fixture(scope="session")
def modelmesh_serviceaccount(client, model_namespace):
    with ServiceAccount(client=client, name="modelmesh-serving-sa", namespace=model_namespace.name):
        yield


@pytest.fixture(scope="session")
def cluster_monitoring_config(client):
    config_yaml = yaml.dump({"enableUserWorkload": "true"})
    with ConfigMap(
        name="cluster-monitoring-config",
        namespace="openshift-monitoring",
        data={"config.yaml": config_yaml},
    ) as cm:
        yield cm


@pytest.fixture(scope="session")
def user_workload_monitoring_config(client):
    config_yaml = yaml.dump({"prometheus": {"logLevel": "debug", "retention": "15d"}})
    with ConfigMap(
        name="user-workload-monitoring-config",
        namespace="openshift-user-workload-monitoring",
        data={"config.yaml": config_yaml},
    ) as cm:
        yield cm


@pytest.fixture(scope="session")
def trustyai_service(
    client,
    model_namespace,
    modelmesh_serviceaccount,
):
    with TrustyAIService(
        name=TRUSTYAI_SERVICE,
        namespace=model_namespace.name,
        storage_format="PVC",
        storage_folder="/inputs",
        storage_size="1Gi",
        data_filename="data.csv",
        data_format="CSV",
        metrics_schedule_interval="5s",
        client=client,
    ) as trusty:
        yield trusty


@pytest.fixture(scope="session")
def minio_service(client, model_namespace):
    with MinioService(
        name="minio",
        port=9000,
        target_port=9000,
        namespace=model_namespace.name,
        client=client,
    ) as ms:
        yield ms


@pytest.fixture(scope="session")
def minio_pod(client, model_namespace):
    with MinioPod(client=client, name="minio", namespace=model_namespace.name, image=MINIO_IMAGE) as mp:
        yield mp


@pytest.fixture(scope="session")
def minio_secret(client, model_namespace):
    with MinioSecret(
        client=client,
        name="aws-connection-minio-data-connection",
        namespace=model_namespace.name,
        # Dummy AWS values
        aws_access_key_id="VEhFQUNDRVNTS0VZ",
        aws_default_region="dXMtc291dGg=",
        aws_s3_bucket="bW9kZWxtZXNoLWV4YW1wbGUtbW9kZWxz",
        aws_s3_endpoint="aHR0cDovL21pbmlvOjkwMDA=",
        aws_secret_access_key="VEhFU0VDUkVUS0VZ",
    ) as ms:
        yield ms


@pytest.fixture(scope="session")
def minio_data_connection(minio_service, minio_pod, minio_secret):
    yield


@pytest.fixture(scope="class")
def ovms_runtime(client, minio_data_connection, model_namespace):
    supported_model_formats = [
        {"name": OPENVINO_MODEL_FORMAT, "version": "opset1", "autoSelect": True},
        {"name": ONNX, "version": "1"},
    ]
    containers = [
        {
            "name": OVMS,
            "image": OVMS_QUAY_IMAGE,
            "args": [
                "--port=8001",
                "--rest_port=8888",
                "--config_path=/models/model_config_list.json",
                "--file_system_poll_wait_seconds=0",
                "--grpc_bind_address=127.0.0.1",
                "--rest_bind_address=127.0.0.1",
            ],
            "resources": {
                "requests": {"cpu": "500m", "memory": "1Gi"},
                "limits": {"cpu": "5", "memory": "1Gi"},
            },
        }
    ]

    with ServingRuntime(
        client=client,
        name=OVMS_RUNTIME,
        namespace=model_namespace.name,
        supported_model_formats=supported_model_formats,
        protocol_versions="grpc-v1",
        multi_model=True,
        containers=containers,
        grpc_endpoint=8085,
        grpc_data_endpoint=8001,
        server_type=OVMS,
        runtime_mgmt_port=8888,
        mem_buffer_bytes=134217728,
        model_loading_timeout_millis=90000,
        enable_route=True,
        label={
            "name": "modelmesh-serving-ovms-1.x-SR",
        },
    ) as ovms:
        yield ovms


@pytest.fixture(scope="class")
def onnx_loan_model_alpha(client, model_namespace, ovms_runtime):
    with InferenceService(
        client=client,
        name="demo-loan-nn-onnx-alpha",
        namespace=model_namespace.name,
        storage_name="aws-connection-minio-data-connection",
        storage_path="onnx/loan_model_alpha_august.onnx",
        model_format_name=ONNX,
        serving_runtime=OVMS_RUNTIME,
        deployment_mode=InferenceService.DeploymentMode.MODEL_MESH,
    ) as inference_service:
        wait_for_model_pods_registered(client=client, namespace=model_namespace)
        yield inference_service
