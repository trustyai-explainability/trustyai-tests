from trustyai_tests.tests.constants import MODEL_DATA_PATH
from trustyai_tests.tests.metrics import get_metric_endpoint, Metric
from trustyai_tests.tests.utils import (
    send_data_to_inference_service,
    verify_trustyai_model_metadata,
    verify_metric_request,
    verify_metric_scheduling,
    apply_trustyai_name_mappings,
    verify_trustyai_metric_prometheus,
    wait_for_model_pods_registered,
)


IS_MALE_IDENTIFYING = "Is Male-Identifying?"
WILL_DEFAULT = "Will Default?"
INPUT_MAPPINGS = {
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
}
OUTPUT_MAPPINGS = {"predict": WILL_DEFAULT}
INPUT_DATA_PATH = f"{MODEL_DATA_PATH}/loan-nn-onnx"


def get_json_data(inference_service):
    return {
        "modelId": inference_service.name,
        "protectedAttribute": IS_MALE_IDENTIFYING,
        "privilegedAttribute": 1.0,
        "unprivilegedAttribute": 0.0,
        "outcomeName": WILL_DEFAULT,
        "favorableOutcome": 0,
        "batchSize": 5000,
    }


class TestFairnessMetrics:
    """
    Verifies the different fairness metrics available in TrustyAI.
    Fairness metrics: Statistical Parity Difference (SPD) and Disparate Impact Ratio (DIR).

    1. Send data to the inference_service (onnx_loan_model_alpha).
    2. Send data to the inference_service.
    3. Apply name mappings.
    4. For each metric:
        4.1. Send a basic request and verify the response.
        4.2. Send a schedule request and verify the response.
        4.3. Verify that the metric has reached Prometheus.
    """

    def test_loan_model_metadata(self, model_namespace, trustyai_service, onnx_loan_model_alpha, onnx_loan_model_beta):
        wait_for_model_pods_registered(namespace=model_namespace)

        for model in [onnx_loan_model_alpha, onnx_loan_model_beta]:
            send_data_to_inference_service(
                inference_service=model,
                namespace=model_namespace,
                data_path=INPUT_DATA_PATH,
            )

            apply_trustyai_name_mappings(
                namespace=model_namespace,
                inference_service=model,
                input_mappings=INPUT_MAPPINGS,
                output_mappings=OUTPUT_MAPPINGS,
            )

            verify_trustyai_model_metadata(
                namespace=model_namespace,
                model=model,
                data_path=INPUT_DATA_PATH,
                expected_percentage_observations=0.3,
            )

    def test_request_spd(self, model_namespace, trustyai_service, onnx_loan_model_alpha, onnx_loan_model_beta):
        for model in [onnx_loan_model_alpha, onnx_loan_model_beta]:
            verify_metric_request(
                namespace=model_namespace,
                model=model,
                endpoint=get_metric_endpoint(metric=Metric.SPD),
                expected_metric_name=Metric.SPD.value.upper(),
                json_data=get_json_data(model),
            )

    def test_schedule_spd(self, model_namespace, trustyai_service, onnx_loan_model_alpha, onnx_loan_model_beta):
        for model in [onnx_loan_model_alpha, onnx_loan_model_beta]:
            verify_metric_scheduling(
                namespace=model_namespace,
                model=model,
                endpoint=get_metric_endpoint(metric=Metric.SPD, schedule=True),
                json_data=get_json_data(model),
            )

    def test_spd_prometheus_query(self, model_namespace, onnx_loan_model_alpha, onnx_loan_model_beta):
        for model in [onnx_loan_model_alpha, onnx_loan_model_beta]:
            verify_trustyai_metric_prometheus(
                namespace=model_namespace,
                model=model,
                prometheus_query=f"trustyai_{Metric.SPD.value}" f'{{namespace="{model_namespace.name}"}}',
                metric_name=Metric.SPD.value,
            )

    def test_request_dir(self, model_namespace, trustyai_service, onnx_loan_model_alpha, onnx_loan_model_beta):
        for model in [onnx_loan_model_alpha, onnx_loan_model_beta]:
            verify_metric_request(
                namespace=model_namespace,
                model=model,
                endpoint=get_metric_endpoint(metric=Metric.DIR),
                expected_metric_name=Metric.DIR.value.upper(),
                json_data=get_json_data(model),
            )

    def test_schedule_dir(self, model_namespace, trustyai_service, onnx_loan_model_alpha, onnx_loan_model_beta):
        for model in [onnx_loan_model_alpha, onnx_loan_model_beta]:
            verify_metric_scheduling(
                namespace=model_namespace,
                model=model,
                endpoint=get_metric_endpoint(metric=Metric.DIR, schedule=True),
                json_data=get_json_data(model),
            )

    def test_dir_prometheus_query(self, model_namespace, onnx_loan_model_alpha, onnx_loan_model_beta):
        for inference_service in [onnx_loan_model_alpha, onnx_loan_model_beta]:
            verify_trustyai_metric_prometheus(
                namespace=model_namespace,
                model=inference_service,
                prometheus_query=f"trustyai_{Metric.DIR.value}" f'{{namespace="{model_namespace.name}"}}',
                metric_name=Metric.DIR.value,
            )


class TestMultipleNamespaces:
    """
    Tests that TrustyAI Operator can handle multiple namespaces.
    Creates three namespaces, deploys TrustyAIService on each, as well as an InferenceService,
    sends data to the InferenceService and sends several fairness metric requests.
    """

    def test_multiple_namespaces(
        self,
        model_namespaces_with_minio,
        trustyai_services_in_namespaces,
        ovms_runtimes_in_namespaces,
        onnx_loan_models_in_namespaces,
    ):
        for namespace, inference_service in zip(model_namespaces_with_minio, onnx_loan_models_in_namespaces):
            send_data_to_inference_service(
                inference_service=inference_service,
                namespace=namespace,
                data_path=INPUT_DATA_PATH,
                num_batches=3,
            )

            apply_trustyai_name_mappings(
                namespace=namespace,
                inference_service=inference_service,
                input_mappings=INPUT_MAPPINGS,
                output_mappings=OUTPUT_MAPPINGS,
            )

            verify_trustyai_model_metadata(
                namespace=namespace,
                model=inference_service,
                data_path=INPUT_DATA_PATH,
                expected_percentage_observations=0.1,
            )

            for metric in (Metric.SPD, Metric.DIR):
                verify_metric_scheduling(
                    namespace=namespace,
                    model=inference_service,
                    endpoint=get_metric_endpoint(metric=metric, schedule=True),
                    json_data=get_json_data(inference_service),
                )

                verify_trustyai_metric_prometheus(
                    namespace=namespace,
                    model=inference_service,
                    prometheus_query=f"trustyai_{metric.value}" f'{{namespace="{namespace.name}"}}',
                    metric_name=metric.value,
                )
