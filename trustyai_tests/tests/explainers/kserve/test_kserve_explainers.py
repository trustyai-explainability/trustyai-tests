import pytest

from trustyai_tests.tests.utils import (
    verify_model_prediction,
    verify_saliency_explanation
)

@pytest.mark.kubernetes
class TestExplainers:
    def test_bank_churn_model_prediction(self, model_namespace, model):
        verify_model_prediction(
            model_namespace, model
        )

    def test_request_lime(self, model_namespace, bank_churn_model_lime):
        verify_saliency_explanation(
            model_namespace=model_namespace,
            model=bank_churn_model_lime
        )

    def test_request_shap(self, model_namespace, bank_churn_model_shap):
        verify_saliency_explanation(
            model_namespace=model_namespace,
            model=bank_churn_model_shap,
        )
