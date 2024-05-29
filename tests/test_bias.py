import logging

from tests.utils import send_data_to_inference_service, verify_trustyai_model_metadata

logger = logging.getLogger(__name__)


class TestBiasMetrics:
    def test_loan_model_metadata(
        self, model_namespace, trustyai_service, onnx_loan_model_alpha
    ):
        input_data_path = f"./model_data/{onnx_loan_model_alpha.name}"
        send_data_to_inference_service(
            inference_service=onnx_loan_model_alpha,
            namespace=model_namespace,
            data_path=input_data_path,
        )

        verify_trustyai_model_metadata(
            namespace=model_namespace,
            model=onnx_loan_model_alpha,
            data_path=input_data_path,
            expected_percentage_observations=0.5,
        )

    # TODO: Add tests for the rest of the bias metrics
