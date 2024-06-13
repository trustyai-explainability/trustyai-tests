import http

from trustyai_tests.tests.utils import (
    verify_trustyai_model_metadata,
    send_data_to_inference_service,
    verify_trustyai_metric_prometheus,
    verify_metric_request,
    verify_metric_scheduling,
    upload_data_to_trustyai_service,
)
from trustyai_tests.utilities.constants import (
    TRUSTYAI_MEANSHIFT_ENDPOINT,
    TRUSTYAI_MEANSHIFT,
    TRUSTYAI_FOURIERMMD,
    TRUSTYAI_KSTEST,
    TRUSTYAI_APPROXKSTEST,
    MODEL_DATA_PATH,
    TRUSTYAI_MEANSHIFT_SCHEDULE_ENDPOINT,
    TRUSTYAI_FOURIERMMD_ENDPOINT,
    TRUSTYAI_FOURIERMMD_SCHEDULE_ENDPOINT,
    TRUSTYAI_KSTEST_ENDPOINT,
    TRUSTYAI_KSTEST_SCHEDULE_ENDPOINT,
    TRUSTYAI_APPROXKSTEST_ENDPOINT,
    TRUSTYAI_APPROXKSTEST_SCHEDULE_ENDPOINT,
)


class TestDriftMetrics:
    """
    Verifies the different input data drift metrics available in TrustyAI.
    Drift metrics: Meanshift, FourierMMD, KSTest, and ApproxKSTest.

    1. Send data to the model (gaussian_credit_model).
    2. Upload training data for TrustyAI (used as baseline to calculate the drift metrics).
    3. For each metric:
        3.1. Send a basic request and verify the response.
        3.2. Send a schedule request and verify the response.
        3.3. Verify that the metric has reached Prometheus.
    """

    def test_gaussian_credit_model_metadata(self, model_namespace, trustyai_service, gaussian_credit_model):
        path = f"{MODEL_DATA_PATH}/{gaussian_credit_model.name}"

        send_data_to_inference_service(
            inference_service=gaussian_credit_model,
            namespace=model_namespace,
            data_path=f"{path}/data_batches",
        )

        response = upload_data_to_trustyai_service(
            namespace=model_namespace,
            data_path=f"{path}/training_data.json",
        )
        assert response.status_code == http.HTTPStatus.OK

        verify_trustyai_model_metadata(
            namespace=model_namespace,
            model=gaussian_credit_model,
            data_path=path,
            expected_percentage_observations=0.3,
        )

    def test_request_meanshift(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_metric_request(
            namespace=model_namespace,
            model=gaussian_credit_model,
            endpoint=TRUSTYAI_MEANSHIFT_ENDPOINT,
            expected_metric_name=TRUSTYAI_MEANSHIFT.upper(),
        )

    def test_schedule_meanshift(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_metric_scheduling(
            namespace=model_namespace,
            model=gaussian_credit_model,
            endpoint=TRUSTYAI_MEANSHIFT_SCHEDULE_ENDPOINT,
            expected_metric_name=TRUSTYAI_MEANSHIFT.upper(),
        )

    def test_meanshift_prometheus_query(self, model_namespace, gaussian_credit_model):
        verify_trustyai_metric_prometheus(
            namespace=model_namespace,
            model=gaussian_credit_model,
            prometheus_query=f'trustyai_{TRUSTYAI_MEANSHIFT.lower()}{{namespace="{model_namespace.name}"}}',
            metric_name=TRUSTYAI_MEANSHIFT,
        )

    def test_request_fouriermmd(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_metric_request(
            namespace=model_namespace,
            model=gaussian_credit_model,
            endpoint=TRUSTYAI_FOURIERMMD_ENDPOINT,
            expected_metric_name=TRUSTYAI_FOURIERMMD.upper(),
        )

    def test_schedule_fouriermmd(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_metric_scheduling(
            namespace=model_namespace,
            model=gaussian_credit_model,
            endpoint=TRUSTYAI_FOURIERMMD_SCHEDULE_ENDPOINT,
            expected_metric_name=TRUSTYAI_FOURIERMMD.upper(),
        )

    def test_fouriermmd_prometheus_query(self, model_namespace, gaussian_credit_model):
        verify_trustyai_metric_prometheus(
            namespace=model_namespace,
            model=gaussian_credit_model,
            prometheus_query=f'trustyai_{TRUSTYAI_FOURIERMMD.lower()}{{namespace="{model_namespace.name}"}}',
            metric_name=TRUSTYAI_FOURIERMMD,
        )

    def test_request_kstest(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_metric_request(
            namespace=model_namespace,
            model=gaussian_credit_model,
            endpoint=TRUSTYAI_KSTEST_ENDPOINT,
            expected_metric_name=TRUSTYAI_KSTEST.upper(),
        )

    def test_schedule_kstest_scheduling_request(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_metric_scheduling(
            namespace=model_namespace,
            model=gaussian_credit_model,
            endpoint=TRUSTYAI_KSTEST_SCHEDULE_ENDPOINT,
            expected_metric_name=TRUSTYAI_KSTEST.upper(),
        )

    def test_kstest_prometheus_query(self, model_namespace, gaussian_credit_model):
        verify_trustyai_metric_prometheus(
            namespace=model_namespace,
            model=gaussian_credit_model,
            prometheus_query=f'trustyai_{TRUSTYAI_KSTEST.lower()}{{namespace="{model_namespace.name}"}}',
            metric_name=TRUSTYAI_KSTEST,
        )

    def test_request_approxkstest(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_metric_request(
            namespace=model_namespace,
            model=gaussian_credit_model,
            endpoint=TRUSTYAI_APPROXKSTEST_ENDPOINT,
            expected_metric_name=TRUSTYAI_APPROXKSTEST.upper(),
        )

    def test_schedule_approxkstest(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_metric_scheduling(
            namespace=model_namespace,
            model=gaussian_credit_model,
            endpoint=TRUSTYAI_APPROXKSTEST_SCHEDULE_ENDPOINT,
            expected_metric_name=TRUSTYAI_APPROXKSTEST.upper(),
        )

    def test_approxkstest_prometheus_query(self, model_namespace, gaussian_credit_model):
        verify_trustyai_metric_prometheus(
            namespace=model_namespace,
            model=gaussian_credit_model,
            prometheus_query=f'trustyai_{TRUSTYAI_APPROXKSTEST.lower()}{{namespace="{model_namespace.name}"}}',
            metric_name=TRUSTYAI_APPROXKSTEST,
        )
