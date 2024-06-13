import pytest

from trustyai_tests.resources.inference_service import InferenceService
from trustyai_tests.resources.serving_runtime import ServingRuntime
from trustyai_tests.tests.utils import wait_for_model_pods_registered
from trustyai_tests.utilities.constants import (
    ONNX,
    OVMS,
    OPENVINO_MODEL_FORMAT,
    OVMS_QUAY_IMAGE,
    GRPC,
    OVMS_RUNTIME_NAME,
)


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
        supported_model_formats=supported_model_formats,
        protocol_versions=f"{GRPC}-v1",
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
        storage_name=minio_data_connection.name,
        storage_path="onnx/loan_model_alpha_august.onnx",
        model_format_name=ONNX,
        serving_runtime=ovms_runtime.name,
        deployment_mode=InferenceService.DeploymentMode.MODEL_MESH,
    ) as inference_service:
        wait_for_model_pods_registered(client=client, namespace=model_namespace)
        yield inference_service
