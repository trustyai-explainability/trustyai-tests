from trustyai_tests.tests.metrics import get_metric_endpoint, Metric
from trustyai_tests.tests.utils import (
    send_data_to_inference_service,
    verify_trustyai_model_metadata,
    verify_metric_request,
    verify_metric_scheduling,
    apply_trustyai_name_mappings,
    verify_trustyai_metric_prometheus,
)
from trustyai_tests.constants import MODEL_DATA_PATH


IS_MALE_IDENTIFYING = "Is Male-Identifying?"
WILL_DEFAULT = "Will Default?"


class TestFairnessMetrics:
    """
    Verifies the different fairness metrics available in TrustyAI.
    Fairness metrics: Statistical Parity Difference (SPD) and Disparate Impact Ratio (DIR).

    1. Send data to the model (onnx_loan_model_alpha).
    2. Send data to the model.
    3. Apply name mappings.
    4. For each metric:
        4.1. Send a basic request and verify the response.
        4.2. Send a schedule request and verify the response.
        4.3. Verify that the metric has reached Prometheus.
    """

    def test_loan_model_metadata(self, model_namespace, trustyai_service, onnx_loan_model_alpha):
        input_data_path = f"{MODEL_DATA_PATH}/{onnx_loan_model_alpha.name}"
        send_data_to_inference_service(
            inference_service=onnx_loan_model_alpha,
            namespace=model_namespace,
            data_path=input_data_path,
        )

        apply_trustyai_name_mappings(
            namespace=model_namespace,
            inference_service=onnx_loan_model_alpha,
            input_mappings={
                "customer_data_input-0": "Number of Children",
                "customer_data_input-1": "Total Income",
                "customer_data_input-2": "Number of Total Family Members",
                "customer_data_input-3": IS_MALE_IDENTIFYING,
                "customer_data_input-4": "Owns Car?",
                "customer_data_input-5": "Owns Realty?",
                "customer_data_input-6": "Is Partnered?",
                "customer_data_input-7": "Is Employed?",
                "customer_data_input-8": "Live with Parents?",
                "customer_data_input-9": "Age",
                "customer_data_input-10": "Length of Employment?",
            },
            output_mappings={"predict": WILL_DEFAULT},
        )

        verify_trustyai_model_metadata(
            namespace=model_namespace,
            model=onnx_loan_model_alpha,
            data_path=input_data_path,
            expected_percentage_observations=0.3,
        )

    def test_request_spd(self, model_namespace, trustyai_service, onnx_loan_model_alpha):
        verify_metric_request(
            namespace=model_namespace,
            model=onnx_loan_model_alpha,
            endpoint=get_metric_endpoint(metric=Metric.SPD),
            expected_metric_name=Metric.SPD.value.upper(),
            json_data={
                "modelId": onnx_loan_model_alpha.name,
                "protectedAttribute": IS_MALE_IDENTIFYING,
                "privilegedAttribute": 1.0,
                "unprivilegedAttribute": 0.0,
                "outcomeName": WILL_DEFAULT,
                "favorableOutcome": 0,
                "batchSize": 5000,
            },
        )

    def test_schedule_spd(self, model_namespace, trustyai_service, onnx_loan_model_alpha):
        verify_metric_scheduling(
            namespace=model_namespace,
            model=onnx_loan_model_alpha,
            endpoint=get_metric_endpoint(metric=Metric.SPD, schedule=True),
            json_data={
                "modelId": onnx_loan_model_alpha.name,
                "protectedAttribute": IS_MALE_IDENTIFYING,
                "privilegedAttribute": 1.0,
                "unprivilegedAttribute": 0.0,
                "outcomeName": WILL_DEFAULT,
                "favorableOutcome": 0,
                "batchSize": 5000,
            },
        )

    def test_spd_prometheus_query(self, model_namespace, onnx_loan_model_alpha):
        verify_trustyai_metric_prometheus(
            namespace=model_namespace,
            model=onnx_loan_model_alpha,
            prometheus_query=f"trustyai_{Metric.SPD.value}" f'{{namespace="{model_namespace.name}"}}',
            metric_name=Metric.SPD.value,
        )

    def test_request_dir(self, model_namespace, trustyai_service, onnx_loan_model_alpha):
        verify_metric_request(
            namespace=model_namespace,
            model=onnx_loan_model_alpha,
            endpoint=get_metric_endpoint(metric=Metric.DIR),
            expected_metric_name=Metric.DIR.value.upper(),
            json_data={
                "modelId": onnx_loan_model_alpha.name,
                "protectedAttribute": IS_MALE_IDENTIFYING,
                "privilegedAttribute": 1.0,
                "unprivilegedAttribute": 0.0,
                "outcomeName": WILL_DEFAULT,
                "favorableOutcome": 0,
                "batchSize": 5000,
            },
        )

    def test_schedule_dir(self, model_namespace, trustyai_service, onnx_loan_model_alpha):
        verify_metric_scheduling(
            namespace=model_namespace,
            model=onnx_loan_model_alpha,
            endpoint=get_metric_endpoint(metric=Metric.DIR, schedule=True),
            json_data={
                "modelId": onnx_loan_model_alpha.name,
                "protectedAttribute": IS_MALE_IDENTIFYING,
                "privilegedAttribute": 1.0,
                "unprivilegedAttribute": 0.0,
                "outcomeName": WILL_DEFAULT,
                "favorableOutcome": 0,
                "batchSize": 5000,
            },
        )

    def test_dir_prometheus_query(self, model_namespace, onnx_loan_model_alpha):
        verify_trustyai_metric_prometheus(
            namespace=model_namespace,
            model=onnx_loan_model_alpha,
            prometheus_query=f"trustyai_{Metric.DIR.value}" f'{{namespace="{model_namespace.name}"}}',
            metric_name=Metric.DIR.value,
        )
