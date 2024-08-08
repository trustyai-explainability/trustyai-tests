from typing import Any

import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.trustyai_service import TrustyAIService

from trustyai_tests.tests.constants import MODEL_DATA_PATH
from trustyai_tests.tests.metrics import get_metric_endpoint, Metric
from trustyai_tests.tests.utils import (
    send_data_to_inference_service,
    verify_trustyai_model_metadata,
    verify_metric_request,
    verify_metric_scheduling,
    apply_trustyai_name_mappings,
    verify_trustyai_metric_prometheus,
    wait_for_modelmesh_pods_registered,
)


IS_MALE_IDENTIFYING: str = "Is Male-Identifying?"
WILL_DEFAULT: str = "Will Default?"
INPUT_MAPPINGS: dict[str, str] = {
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
OUTPUT_MAPPINGS: dict[str, str] = {"predict": WILL_DEFAULT}
INPUT_DATA_PATH: str = f"{MODEL_DATA_PATH}/loan-nn-onnx"


def get_json_data(inference_service: InferenceService) -> dict[str, Any]:
    return {
        "modelId": inference_service.name,
        "protectedAttribute": IS_MALE_IDENTIFYING,
        "privilegedAttribute": 1.0,
        "unprivilegedAttribute": 0.0,
        "outcomeName": WILL_DEFAULT,
        "favorableOutcome": 0,
        "batchSize": 5000,
    }


@pytest.mark.openshift
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

    def test_loan_model_metadata(
        self,
        model_namespace: Namespace,
        trustyai_service: TrustyAIService,
        onnx_loan_model_alpha: InferenceService,
        onnx_loan_model_beta: InferenceService,
    ) -> None:
        wait_for_modelmesh_pods_registered(namespace=model_namespace)

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
            )

    def test_request_spd(
        self,
        model_namespace: Namespace,
        trustyai_service: TrustyAIService,
        onnx_loan_model_alpha: InferenceService,
        onnx_loan_model_beta: InferenceService,
    ) -> None:
        for model in [onnx_loan_model_alpha, onnx_loan_model_beta]:
            verify_metric_request(
                namespace=model_namespace,
                model=model,
                endpoint=get_metric_endpoint(metric=Metric.SPD),
                expected_metric_name=Metric.SPD.value.upper(),
                json_data=get_json_data(model),
            )

    def test_schedule_spd(
        self,
        model_namespace: Namespace,
        trustyai_service: TrustyAIService,
        onnx_loan_model_alpha: InferenceService,
        onnx_loan_model_beta: InferenceService,
    ) -> None:
        for model in [onnx_loan_model_alpha, onnx_loan_model_beta]:
            verify_metric_scheduling(
                namespace=model_namespace,
                model=model,
                endpoint=get_metric_endpoint(metric=Metric.SPD, schedule=True),
                json_data=get_json_data(model),
            )

    def test_spd_prometheus_query(
        self,
        model_namespace: Namespace,
        onnx_loan_model_alpha: InferenceService,
        onnx_loan_model_beta: InferenceService,
    ) -> None:
        for model in [onnx_loan_model_alpha, onnx_loan_model_beta]:
            verify_trustyai_metric_prometheus(
                namespace=model_namespace,
                model=model,
                prometheus_query=f"trustyai_{Metric.SPD.value}" f'{{namespace="{model_namespace.name}"}}',
                metric_name=Metric.SPD.value,
            )

    def test_request_dir(
        self,
        model_namespace: Namespace,
        trustyai_service: TrustyAIService,
        onnx_loan_model_alpha: InferenceService,
        onnx_loan_model_beta: InferenceService,
    ) -> None:
        for model in [onnx_loan_model_alpha, onnx_loan_model_beta]:
            verify_metric_request(
                namespace=model_namespace,
                model=model,
                endpoint=get_metric_endpoint(metric=Metric.DIR),
                expected_metric_name=Metric.DIR.value.upper(),
                json_data=get_json_data(model),
            )

    def test_schedule_dir(
        self,
        model_namespace: Namespace,
        trustyai_service: TrustyAIService,
        onnx_loan_model_alpha: InferenceService,
        onnx_loan_model_beta: InferenceService,
    ) -> None:
        for model in [onnx_loan_model_alpha, onnx_loan_model_beta]:
            verify_metric_scheduling(
                namespace=model_namespace,
                model=model,
                endpoint=get_metric_endpoint(metric=Metric.DIR, schedule=True),
                json_data=get_json_data(model),
            )

    def test_dir_prometheus_query(
        self,
        model_namespace: Namespace,
        onnx_loan_model_alpha: InferenceService,
        onnx_loan_model_beta: InferenceService,
    ) -> None:
        for inference_service in [onnx_loan_model_alpha, onnx_loan_model_beta]:
            verify_trustyai_metric_prometheus(
                namespace=model_namespace,
                model=inference_service,
                prometheus_query=f"trustyai_{Metric.DIR.value}" f'{{namespace="{model_namespace.name}"}}',
                metric_name=Metric.DIR.value,
            )


@pytest.mark.openshift
class TestMultipleNamespaces:
    """
    Tests that TrustyAI Operator can handle multiple namespaces.
    Creates three namespaces, deploys TrustyAIService on each, as well as an InferenceService,
    sends data to the InferenceService and sends several fairness metric requests.
    """

    def test_multiple_namespaces(
        self,
        model_namespaces_with_minio: list(Namespace),
        trustyai_services_in_namespaces: list(TrustyAIService),
        ovms_runtimes_in_namespaces: list(ServingRuntime),
        onnx_loan_models_in_namespaces: list(InferenceService),
    ):
        num_batches = 3
        for namespace, inference_service in zip(model_namespaces_with_minio, onnx_loan_models_in_namespaces):
            send_data_to_inference_service(
                inference_service=inference_service,
                namespace=namespace,
                data_path=INPUT_DATA_PATH,
                num_batches=num_batches,
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
                num_batches=num_batches,
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
