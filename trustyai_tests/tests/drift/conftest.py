import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime

from trustyai_tests.tests.constants import KSERVE_API_GROUP
from trustyai_tests.tests.minio import MinioSecret

SKLEARN: str = "sklearn"
XGBOOST: str = "xgboost"
MLSERVER: str = "mlserver"
MLSERVER_RUNTIME_NAME: str = f"{MLSERVER}-1.x"
MLSERVER_QUAY_IMAGE: str = "quay.io/aaguirre/mlserver:1.3.2"  # TODO: Change it to TrustyAI specific quay


@pytest.fixture(scope="class")
def mlserver_runtime(
    client: DynamicClient, minio_data_connection: MinioSecret, model_namespace: Namespace
) -> ServingRuntime:
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
def gaussian_credit_model(
    client: DynamicClient,
    model_namespace: Namespace,
    minio_data_connection: MinioSecret,
    mlserver_runtime: ServingRuntime,
) -> InferenceService:
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
        inference_service.wait_for_condition(
            condition=inference_service.Condition.READY, status=inference_service.Condition.Status.TRUE, timeout=5 * 60
        )
        yield inference_service
