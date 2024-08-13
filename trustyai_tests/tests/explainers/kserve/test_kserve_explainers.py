import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace

from trustyai_tests.tests.utils import verify_model_prediction, verify_saliency_explanation


@pytest.mark.kubernetes
class TestExplainers:
    def test_bank_churn_model_prediction(self, model_namespace: Namespace, model: InferenceService) -> None:
        verify_model_prediction(model_namespace=model_namespace, model=model)

    def test_request_lime_shap(
        self, model_namespace, bank_churn_model_lime_shap, explainer_type=["SHAP", "LIME"]
    ) -> None:
        verify_saliency_explanation(
            model_namespace=model_namespace, model=bank_churn_model_lime_shap, explainer_type=explainer_type
        )

    def test_request_lime(
        self, model_namespace: Namespace, bank_churn_model_lime: InferenceService, explainer_type=["LIME"]
    ) -> None:
        verify_saliency_explanation(
            model_namespace=model_namespace, model=bank_churn_model_lime, explainer_type=explainer_type
        )

    def test_request_shap(
        self, model_namespace: Namespace, bank_churn_model_shap: InferenceService, explainer_type=["SHAP"]
    ) -> None:
        verify_saliency_explanation(
            model_namespace=model_namespace, model=bank_churn_model_shap, explainer_type=explainer_type
        )
