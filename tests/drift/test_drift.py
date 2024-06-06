import http
import logging
from tests.utils import (
    verify_trustyai_model_metadata,
    upload_data_to_trustyai_service,
    request_meanshift,
    request_kstest,
    request_fouriermmd,
    request_approxkstest,
)

logger = logging.getLogger(__name__)


class TestDriftMetrics:
    def test_gaussian_credit_model_metadata(self, model_namespace, trustyai_service, gaussian_credit_model):
        input_data_path = f"./model_data/{gaussian_credit_model.name}"
        response = upload_data_to_trustyai_service(
            namespace=model_namespace,
            data_path=input_data_path,
        )
        assert response.status_code == http.HTTPStatus.OK
        assert response.content == b"1000 datapoints successfully added to gaussian-credit-model data."

        verify_trustyai_model_metadata(
            namespace=model_namespace,
            model=gaussian_credit_model,
            data_path=input_data_path,
            expected_percentage_observations=0.3,
        )

    def test_request_meanshift(self, model_namespace, trustyai_service, gaussian_credit_model):
        response = request_meanshift(namespace=model_namespace, model=gaussian_credit_model)

        print(response.text)
        print(response.status_code)

    def test_request_approxkstest(self, model_namespace, trustyai_service, gaussian_credit_model):
        response = request_approxkstest(namespace=model_namespace, model=gaussian_credit_model)

        print(response.text)
        print(response.status_code)

    def test_request_fouriermmd(self, model_namespace, trustyai_service, gaussian_credit_model):
        response = request_fouriermmd(namespace=model_namespace, model=gaussian_credit_model)

        print(response.text)
        print(response.status_code)

    def test_request_kstest(self, model_namespace, trustyai_service, gaussian_credit_model):
        response = request_kstest(namespace=model_namespace, model=gaussian_credit_model)

        print(response.text)
        print(response.status_code)
