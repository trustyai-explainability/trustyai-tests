from typing import Any

import pytest

from trustyai_tests.tests.fairness.test_fairness import get_json_data, INPUT_DATA_PATH, INPUT_MAPPINGS, OUTPUT_MAPPINGS
from trustyai_tests.tests.metrics import get_metric_endpoint, Metric
from trustyai_tests.tests.utils import (
    verify_trustyai_model_metadata,
    verify_metric_scheduling,
    verify_trustyai_metric_prometheus,
    send_data_to_inference_service,
    apply_trustyai_name_mappings,
)


@pytest.mark.parametrize(
    "trustyai_services_in_namespaces",
    [pytest.param({"storage_type": "pvc"}, id="pvc"), pytest.param({"storage_type": "db"}, id="db")],
    indirect=True,
)
@pytest.mark.openshift
@pytest.mark.heavy
class TestMultipleNamespaces:
    """
    Tests that TrustyAI Operator can handle multiple namespaces.
    Creates three namespaces, deploys TrustyAIService on each, as well as an InferenceService,
    sends data to the InferenceService and sends several fairness metric requests.
    """

    def test_multiple_namespaces(
        self,
        model_namespaces_with_minio: Any,
        trustyai_services_in_namespaces: Any,
        ovms_runtimes_in_namespaces: Any,
        onnx_loan_models_in_namespaces: Any,
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
                    endpoint=get_metric_endpoint(metric=metric, schedule=True),
                    json_data=get_json_data(inference_service),
                )

                verify_trustyai_metric_prometheus(
                    namespace=namespace,
                    model=inference_service,
                    prometheus_query=f"trustyai_{metric.value}" f'{{namespace="{namespace.name}"}}',
                    metric_name=metric.value,
                )
