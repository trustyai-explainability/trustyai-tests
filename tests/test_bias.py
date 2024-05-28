import http
import logging
import pytest
import json

from tests.utils import send_data_to_inference_service, get_trustyai_model_metadata

logger = logging.getLogger(__name__)

class TestBiasMetrics:
    def test_get_loan_model_metadata(self, model_namespace, trustyai_service, onnx_loan_model_alpha):
        send_data_to_inference_service(inference_service=onnx_loan_model_alpha,
                                       namespace=model_namespace,
                                       data_path="./model_data/bias_loan")

        response = get_trustyai_model_metadata(namespace=model_namespace)
        assert response.status_code == http.HTTPStatus.OK
        content = json.loads(response.content)
        pretty_content = json.dumps(content, indent=4)
        logger.info(pretty_content)
