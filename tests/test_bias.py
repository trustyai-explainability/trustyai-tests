import pytest

from tests.utils import send_data_to_inference_service


class TestBiasMetrics:
    def test_loan_model_metadata(client, model_namespace, trustyai_service, onnx_loan_model_alpha):
        send_data_to_inference_service(client=client,
                                       inference_service=onnx_loan_model_alpha,
                                       namespace=model_namespace,
                                       data_path="./data/training")