import pytest

from ocp_resources.inference_service import InferenceService
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.trustyai_service import TrustyAIService

from trustyai_tests.tests.constants import (
    OPENVINO_MODEL_FORMAT,
    KSERVE_API_GROUP,
    TRUSTYAI_SERVICE,
    MINIO_DATA_CONNECTION_NAME,
)
from trustyai_tests.tests.fairness.utils import deploy_namespace_with_minio
from trustyai_tests.tests.utils import wait_for_modelmesh_pods_registered

ONNX = "onnx"
OVMS = "ovms"
OVMS_RUNTIME_NAME = f"{OVMS}-1.x"
OVMS_QUAY_IMAGE = "quay.io/opendatahub/openvino_model_server:stable"
ONNX_LOAN_MODEL_ALPHA_PATH = "onnx/loan_model_alpha_august.onnx"


def create_ovms_runtime(namespace):
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

    return ServingRuntime(
        name=OVMS_RUNTIME_NAME,
        namespace=namespace.name,
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
    )


@pytest.fixture(scope="class")
def ovms_runtime(minio_data_connection, model_namespace):
    with create_ovms_runtime(model_namespace) as ovms:
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
        inference_service.wait_for_condition(
            condition=inference_service.Condition.READY, status=inference_service.Condition.Status.TRUE, timeout=5 * 60
        )
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
        inference_service.wait_for_condition(
            condition=inference_service.Condition.READY, status=inference_service.Condition.Status.TRUE, timeout=5 * 60
        )
        yield inference_service


@pytest.fixture(scope="class")
def model_namespaces_with_minio():
    namespaces = []

    for i in range(2):
        namespace = deploy_namespace_with_minio(name=f"test-namespace-{i}")
        namespaces.append(namespace)

    yield namespaces
    for namespace in namespaces:
        namespace.delete(wait=True)


@pytest.fixture(scope="class")
def trustyai_services_in_namespaces(model_namespaces_with_minio):
    trustyai_services = []

    for namespace in model_namespaces_with_minio:
        trustyai_service = TrustyAIService(
            name=TRUSTYAI_SERVICE,
            namespace=namespace.name,
            storage={"format": "PVC", "folder": "/inputs", "size": "1Gi"},
            data={"filename": "data.csv", "format": "CSV"},
            metrics={"schedule": "5s"},
        )
        trustyai_service.deploy()
        trustyai_services.append(trustyai_service)
    yield trustyai_services
    for trustyai_service in trustyai_services:
        trustyai_service.delete(wait=True)


@pytest.fixture(scope="class")
def ovms_runtimes_in_namespaces(model_namespaces_with_minio):
    ovms_runtimes = []

    for namespace in model_namespaces_with_minio:
        ovms_runtime = create_ovms_runtime(namespace=namespace)
        ovms_runtime.deploy()
        ovms_runtimes.append(ovms_runtime)
    yield ovms_runtimes
    for ovms_runtime in ovms_runtimes:
        ovms_runtime.delete(wait=True)


@pytest.fixture(scope="class")
def onnx_loan_models_in_namespaces(model_namespaces_with_minio, ovms_runtimes_in_namespaces):
    inference_services = []
    for namespace, ovms_runtime in zip(model_namespaces_with_minio, ovms_runtimes_in_namespaces):
        inference_service = InferenceService(
            name="demo-loan-nn-onnx-alpha",
            namespace=namespace.name,
            predictor={
                "model": {
                    "modelFormat": {"name": ONNX},
                    "runtime": ovms_runtime.name,
                    "storage": {"key": MINIO_DATA_CONNECTION_NAME, "path": ONNX_LOAN_MODEL_ALPHA_PATH},
                }
            },
            annotations={f"{KSERVE_API_GROUP}/deploymentMode": "ModelMesh"},
        )
        inference_service.deploy()
        wait_for_modelmesh_pods_registered(namespace=namespace)
        inference_services.append(inference_service)
    yield inference_services
    for inference_service in inference_services:
        inference_service.delete(wait=True)
