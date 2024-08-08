import pytest
from kubernetes.dynamic import DynamicClient

from ocp_resources.inference_service import InferenceService


@pytest.fixture(scope="class")
def bank_churn_model_lime(client: DynamicClient, model_namespace: str) -> InferenceService:
    with InferenceService(
        client=client,
        name="explainer-test-lime",
        namespace=model_namespace,
        predictor={
            "model": {
                "modelFormat": {"name": "sklearn"},
                "protocolVersion": "v2",
                "runtime": "kserve-sklearnserver",
                "storageUri": "https://github.com/trustyai-explainability/model-collection/raw/bank-churn/model.joblib",
            },
        },
        explainer={"containers": [{"name": "explainer", "image": "quay.io/trustyai/trustyai-kserve-explainer:latest"}]},
    ) as inference_service:
        yield inference_service


@pytest.fixture(scope="class")
def bank_churn_model_shap(client: DynamicClient, model_namespace: str) -> InferenceService:
    with InferenceService(
        client=client,
        name="explainer-test-shap",
        namespace=model_namespace,
        predictor={
            "model": {
                "modelFormat": {"name": "sklearn"},
                "protocolVersion": "v2",
                "runtime": "kserve-sklearnserver",
                "storageUri": "https://github.com/trustyai-explainability/model-collection/raw/bank-churn/model.joblib",
            },
        },
        explainer={
            "containers": [
                {
                    "name": "explainer",
                    "image": "quay.io/trustyai/trustyai-kserve-explainer:latest",
                    "env": {"name": "SHAP", "value": "SHAP"},
                }
            ]
        },
    ) as inference_service:
        yield inference_service
