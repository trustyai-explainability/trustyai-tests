import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.serving_runtime import ServingRuntime

from trustyai_tests.tests.constants import KSERVE_API_GROUP
from trustyai_tests.tests.utils import wait_for_model_pods_registered

SKLEARN = "sklearn"
XGBOOST = "xgboost"
MLSERVER = "mlserver"
MLSERVER_RUNTIME_NAME = f"{MLSERVER}-1.x"
MLSERVER_QUAY_IMAGE = "quay.io/aaguirre/mlserver:1.3.2"  # TODO: Change it to TrustyAI specific quay


@pytest.fixture(scope="class")
def mlserver_runtime(client, minio_data_connection, model_namespace):
    supported_model_formats = [
        {"name": SKLEARN, "version": "0", "autoselect": "true"},
        {"name": XGBOOST, "version": "1", "autoselect": "true"},
        {"name": "lightgbm", "version": "3", "autoselect": "true"},
    ]
    containers = [
        {
            "name": MLSERVER,
            "image": MLSERVER_QUAY_IMAGE,
            "env": [
                {"name": "MLSERVER_MODELS_DIR", "value": "/models/_mlserver_models/"},
                {"name": "MLSERVER_GRPC_PORT", "value": "8001"},
                {"name": "MLSERVER_HTTP_PORT", "value": "8002"},
                {"name": "MLSERVER_LOAD_MODELS_AT_STARTUP", "value": "false"},
                {"name": "MLSERVER_MODEL_NAME", "value": "dummy-model-fixme"},
                {"name": "MLSERVER_HOST", "value": "127.0.0.1"},
                {"name": "MLSERVER_GRPC_MAX_MESSAGE_LENGTH", "value": "-1"},
            ],
            "resources": {"requests": {"cpu": "500m", "memory": "1Gi"}, "limits": {"cpu": "5", "memory": "1Gi"}},
        }
    ]

    with ServingRuntime(
        client=client,
        name=MLSERVER_RUNTIME_NAME,
        namespace=model_namespace.name,
        containers=containers,
        supported_model_formats=supported_model_formats,
        multi_model=True,
        protocol_versions=["grpc-v2"],
        grpc_endpoint="port:8085",
        grpc_data_endpoint="port:8001",
        built_in_adapter={
            "serverType": MLSERVER,
            "runtimeManagementPort": 8001,
            "memBufferBytes": 134217728,
            "modelLoadingTimeoutMillis": 90000,
        },
        annotations={"enable-route": "true"},
        label={"name": f"modelmesh-serving-{MLSERVER_RUNTIME_NAME}-SR"},
    ) as mlserver:
        yield mlserver


@pytest.fixture(scope="class")
def gaussian_credit_model(client, model_namespace, minio_data_connection, mlserver_runtime):
    with InferenceService(
        client=client,
        name="gaussian-credit-model",
        namespace=model_namespace.name,
        predictor={
            "model": {
                "modelFormat": {"name": XGBOOST},
                "runtime": mlserver_runtime.name,
                "storage": {"key": minio_data_connection.name, "path": f"{SKLEARN}/gaussian_credit_model.json"},
            }
        },
        annotations={f"{KSERVE_API_GROUP}/deploymentMode": "ModelMesh"},
    ) as inference_service:
        wait_for_model_pods_registered(client=client, namespace=model_namespace)
        yield inference_service
