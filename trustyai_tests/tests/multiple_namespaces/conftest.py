from typing import Any, Generator

import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.trustyai_service import TrustyAIService

from trustyai_tests.tests.constants import (
    KSERVE_API_GROUP,
    MINIO_DATA_CONNECTION_NAME,
    TRUSTYAI_SERVICE,
    ONNX,
    ONNX_LOAN_MODEL_ALPHA_PATH,
)
from trustyai_tests.tests.multiple_namespaces.utils import deploy_namespace_with_minio
from trustyai_tests.tests.utils import wait_for_modelmesh_pods_registered, create_ovms_runtime


@pytest.fixture(scope="class")
def model_namespaces_with_minio() -> Generator[list[Namespace], Any, None]:
    namespaces = []

    for i in range(2):
        namespace = deploy_namespace_with_minio(name=f"test-namespace-{i}")
        namespaces.append(namespace)

    yield namespaces
    for namespace in namespaces:
        namespace.delete(wait=True)


@pytest.fixture(scope="class")
def trustyai_services_in_namespaces(
    request,
    model_namespaces_with_minio: list[Namespace],
) -> Generator[list[TrustyAIService], Any, None]:
    trustyai_services = []
    storage_type = request.param["storage_type"]
    metrics = {"schedule": "5s"}

    for namespace in model_namespaces_with_minio:
        if storage_type == "pvc":
            trustyai_service = TrustyAIService(
                name=TRUSTYAI_SERVICE,
                namespace=namespace.name,
                storage={"format": "PVC", "folder": "/inputs", "size": "1Gi"},
                data={"filename": "data.csv", "format": "CSV"},
                metrics=metrics,
            )
        else:  # run with db
            trustyai_service = TrustyAIService(
                name=TRUSTYAI_SERVICE,
                namespace=namespace.name,
                storage={"format": "DATABASE", "databaseConfigurations": "db-credentials"},
                metrics=metrics,
            )
        trustyai_service.deploy()
        trustyai_services.append(trustyai_service)
    yield trustyai_services
    for trustyai_service in trustyai_services:
        trustyai_service.delete(wait=True)


@pytest.fixture(scope="class")
def ovms_runtimes_in_namespaces(
    model_namespaces_with_minio: Any,
) -> Generator[list[ServingRuntime], Any, None]:
    ovms_runtimes = []

    for namespace in model_namespaces_with_minio:
        ovms_runtime = create_ovms_runtime(namespace=namespace)
        ovms_runtime.deploy()
        ovms_runtimes.append(ovms_runtime)
    yield ovms_runtimes
    for ovms_runtime in ovms_runtimes:
        ovms_runtime.delete(wait=True)


@pytest.fixture(scope="class")
def onnx_loan_models_in_namespaces(
    model_namespaces_with_minio: Any, ovms_runtimes_in_namespaces: Any
) -> Generator[list[InferenceService], Any, None]:
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
