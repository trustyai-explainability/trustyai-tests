import pytest

from ocp_resources.inference_service import InferenceService
from ocp_resources.serving_runtime import ServingRuntime

from trustyai_tests.tests.constants import (
    OPENVINO_MODEL_FORMAT,
    KSERVE_API_GROUP,
)

ONNX = "onnx"
OVMS = "ovms"
OVMS_RUNTIME_NAME = f"{OVMS}-1.x"
OVMS_QUAY_IMAGE = "quay.io/opendatahub/openvino_model_server:stable"


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
        name=OVMS_RUNTIME_NAME,
        namespace=model_namespace.name,
        containers=containers,
        supported_model_formats=supported_model_formats,
        multi_model=True,
        protocol_versions=["grpc-v1"],
        grpc_endpoint="port:8085",
        grpc_data_endpoint="port:8001",
        built_in_adapter={
            "serverType": OVMS,
            "runtimeManagementPort": 8888,
            "memBufferBytes": 134217728,
            "modelLoadingTimeoutMillis": 90000,
        },
        annotations={"enable-route": "true"},
        label={
            "name": f"modelmesh-serving-{OVMS_RUNTIME_NAME}-SR",
        },
    ) as ovms:
        yield ovms


@pytest.fixture(scope="class")
def onnx_loan_model_alpha(client, model_namespace, minio_data_connection, ovms_runtime):
    with InferenceService(
        client=client,
        name="demo-loan-nn-onnx-alpha",
        namespace=model_namespace.name,
        predictor={
            "model": {
                "modelFormat": {"name": ONNX},
                "runtime": ovms_runtime.name,
                "storage": {"key": minio_data_connection.name, "path": "onnx/loan_model_alpha_august.onnx"},
            }
        },
        annotations={f"{KSERVE_API_GROUP}/deploymentMode": "ModelMesh"},
    ) as inference_service:
        yield inference_service


@pytest.fixture(scope="class")
def onnx_loan_model_beta(client, model_namespace, minio_data_connection, ovms_runtime):
    with InferenceService(
        client=client,
        name="demo-loan-nn-onnx-beta",
        namespace=model_namespace.name,
        predictor={
            "model": {
                "modelFormat": {"name": ONNX},
                "runtime": ovms_runtime.name,
                "storage": {"key": minio_data_connection.name, "path": "onnx/loan_model_beta_august.onnx"},
            }
        },
        annotations={f"{KSERVE_API_GROUP}/deploymentMode": "ModelMesh"},
    ) as inference_service:
        yield inference_service
