import pytest
from kubernetes.dynamic import DynamicClient

from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from trustyai_tests.tests.constants import (
    KSERVE_API_GROUP,
    ONNX,
)
from trustyai_tests.tests.utils import create_ovms_runtime, create_kserve_ovms_runtime


@pytest.fixture(scope="class")
def ovms_runtime(minio_data_connection: Secret, model_namespace: Namespace) -> ServingRuntime:
    with create_ovms_runtime(model_namespace) as ovms:
        yield ovms


@pytest.fixture(scope="class")
def ovms_runtime_kserve(minio_data_connection: Secret, model_namespace: Namespace) -> ServingRuntime:
    with create_kserve_ovms_runtime(model_namespace) as ovms:
        yield ovms


@pytest.fixture(scope="class")
def onnx_loan_model_alpha(
    client: DynamicClient, model_namespace: Namespace, minio_data_connection: Secret, ovms_runtime: ServingRuntime
) -> InferenceService:
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
            condition=inference_service.Condition.READY, status=inference_service.Condition.Status.TRUE, timeout=10 * 60
        )
        yield inference_service


@pytest.fixture(scope="class")
def onnx_loan_model_alpha_kserve(
    client: DynamicClient,
    model_namespace: Namespace,
    minio_data_connection: Secret,
    ovms_runtime_kserve: ServingRuntime,
) -> InferenceService:
    with InferenceService(
        client=client,
        name="demo-loan-nn-onnx-alpha",
        namespace=model_namespace.name,
        predictor={
            "maxReplicas": 1,
            "minReplicas": 1,
            "model": {
                "modelFormat": {"name": ONNX, "version": "1"},
                "resources": {"limits": {"cpu": "2", "memory": "8Gi"}, "requests": {"cpu": "1", "memory": "4Gi"}},
                "runtime": ovms_runtime_kserve.name,
                "storage": {"key": minio_data_connection.name, "path": "ovms/loan_model_alpha"},
            },
        },
        annotations={
            "openshift.io/display-name": "demo-loan-nn-onnx-alpha",
            # "security.opendatahub.io/enable-auth": "true",
            "serving.knative.openshift.io/enablePassthrough": "true",
            "sidecar.istio.io/inject": "true",
            "sidecar.istio.io/rewriteAppHTTPProbers": "true",
        },
    ) as inference_service:
        inference_service.wait_for_condition(
            condition=inference_service.Condition.READY, status=inference_service.Condition.Status.TRUE, timeout=10 * 60
        )
        yield inference_service


@pytest.fixture(scope="class")
def onnx_loan_model_beta(
    client: DynamicClient, model_namespace: Namespace, minio_data_connection: Secret, ovms_runtime: ServingRuntime
) -> InferenceService:
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
            condition=inference_service.Condition.READY, status=inference_service.Condition.Status.TRUE, timeout=10 * 60
        )
        yield inference_service
