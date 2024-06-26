import http
from trustyai_tests.tests.drift.utils import get_drift_metric_endpoint, DriftMetrics
from trustyai_tests.tests.utils import (
    verify_trustyai_model_metadata,
    send_data_to_inference_service,
    verify_trustyai_metric_prometheus,
    verify_metric_request,
    verify_metric_scheduling,
    upload_data_to_trustyai_service,
)
from trustyai_tests.constants import (
    MODEL_DATA_PATH,
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
            endpoint=get_drift_metric_endpoint(metric=DriftMetrics.TRUSTYAI_MEANSHIFT.value),
            expected_metric_name=DriftMetrics.TRUSTYAI_MEANSHIFT.value.upper(),
            json_data={"modelId": gaussian_credit_model.name, "referenceTag": "TRAINING"},
        )

    def test_schedule_meanshift(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_metric_scheduling(
            namespace=model_namespace,
            model=gaussian_credit_model,
            endpoint=get_drift_metric_endpoint(metric=DriftMetrics.TRUSTYAI_MEANSHIFT.value, schedule=True),
            json_data={"modelId": gaussian_credit_model.name, "referenceTag": "TRAINING"},
        )

    def test_meanshift_prometheus_query(self, model_namespace, gaussian_credit_model):
        verify_trustyai_metric_prometheus(
            namespace=model_namespace,
            model=gaussian_credit_model,
            prometheus_query=f'trustyai_{DriftMetrics.TRUSTYAI_MEANSHIFT.value}{{namespace="{model_namespace.name}"}}',
            metric_name=DriftMetrics.TRUSTYAI_MEANSHIFT.value,
        )

    def test_request_fouriermmd(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_metric_request(
            namespace=model_namespace,
            model=gaussian_credit_model,
            endpoint=get_drift_metric_endpoint(metric=DriftMetrics.TRUSTYAI_FOURIERMMD.value),
            expected_metric_name=DriftMetrics.TRUSTYAI_FOURIERMMD.value.upper(),
            json_data={"modelId": gaussian_credit_model.name, "referenceTag": "TRAINING"},
        )

    def test_schedule_fouriermmd(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_metric_scheduling(
            namespace=model_namespace,
            model=gaussian_credit_model,
            endpoint=get_drift_metric_endpoint(metric=DriftMetrics.TRUSTYAI_FOURIERMMD.value, schedule=True),
            json_data={"modelId": gaussian_credit_model.name, "referenceTag": "TRAINING"},
        )

    def test_fouriermmd_prometheus_query(self, model_namespace, gaussian_credit_model):
        verify_trustyai_metric_prometheus(
            namespace=model_namespace,
            model=gaussian_credit_model,
            prometheus_query=f'trustyai_{DriftMetrics.TRUSTYAI_FOURIERMMD.value}{{namespace="{model_namespace.name}"}}',
            metric_name=DriftMetrics.TRUSTYAI_FOURIERMMD.value,
        )

    def test_request_kstest(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_metric_request(
            namespace=model_namespace,
            model=gaussian_credit_model,
            endpoint=get_drift_metric_endpoint(metric=DriftMetrics.TRUSTYAI_KSTEST.value),
            expected_metric_name=DriftMetrics.TRUSTYAI_KSTEST.value.upper(),
            json_data={"modelId": gaussian_credit_model.name, "referenceTag": "TRAINING"},
        )

    def test_schedule_kstest_scheduling_request(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_metric_scheduling(
            namespace=model_namespace,
            model=gaussian_credit_model,
            endpoint=get_drift_metric_endpoint(metric=DriftMetrics.TRUSTYAI_KSTEST.value, schedule=True),
            json_data={"modelId": gaussian_credit_model.name, "referenceTag": "TRAINING"},
        )

    def test_kstest_prometheus_query(self, model_namespace, gaussian_credit_model):
        verify_trustyai_metric_prometheus(
            namespace=model_namespace,
            model=gaussian_credit_model,
            prometheus_query=f'trustyai_{DriftMetrics.TRUSTYAI_KSTEST.value}{{namespace="{model_namespace.name}"}}',
            metric_name=DriftMetrics.TRUSTYAI_KSTEST.value,
        )

    def test_request_approxkstest(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_metric_request(
            namespace=model_namespace,
            model=gaussian_credit_model,
            endpoint=get_drift_metric_endpoint(metric=DriftMetrics.TRUSTYAI_APPROXKSTEST.value),
            expected_metric_name=DriftMetrics.TRUSTYAI_APPROXKSTEST.value.upper(),
            json_data={"modelId": gaussian_credit_model.name, "referenceTag": "TRAINING"},
        )

    def test_schedule_approxkstest(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_metric_scheduling(
            namespace=model_namespace,
            model=gaussian_credit_model,
            endpoint=get_drift_metric_endpoint(metric=DriftMetrics.TRUSTYAI_APPROXKSTEST.value, schedule=True),
            json_data={"modelId": gaussian_credit_model.name, "referenceTag": "TRAINING"},
        )

    def test_approxkstest_prometheus_query(self, model_namespace, gaussian_credit_model):
        verify_trustyai_metric_prometheus(
            namespace=model_namespace,
            model=gaussian_credit_model,
            prometheus_query=f"trustyai_{DriftMetrics.TRUSTYAI_APPROXKSTEST.value}"
            f'{{namespace="{model_namespace.name}"}}',
            metric_name=DriftMetrics.TRUSTYAI_APPROXKSTEST.value,
        )
