import http
import logging


from tests.drift.utils import (
    verify_meanshift_request,
    verify_approxkstest_request,
    verify_fouriermmd_request,
    verify_kstest_request,
    upload_data_to_trustyai_service,
)
from tests.utils import (
    verify_trustyai_model_metadata,
)

logger = logging.getLogger(__name__)
PROMETHEUS_K8S = "prometheus-k8s"
OPENSHIFT_MONITORING = "openshift-monitoring"


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

    def test_meanshift(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_meanshift_request(namespace=model_namespace, model=gaussian_credit_model)

        """
        input_data_path = "./model_data/data_batches"
        send_data_to_inference_service(
            inference_service=gaussian_credit_model,
            namespace=model_namespace,
            data_path=input_data_path,
        )
        time.sleep(180)

        prom = Prometheus(
            namespace="openshift-user-workload-monitoring",
            resource_name="federate",
            verify_ssl=False,
            bearer_token=get_prometheus_user_workload_token())
        result = prom.query(query="trustyai_meanshift{namespace='test-namespace'}")

        print(result)
        """

    def test_approxkstest(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_approxkstest_request(namespace=model_namespace, model=gaussian_credit_model)

    def test_fouriermmd(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_fouriermmd_request(namespace=model_namespace, model=gaussian_credit_model)

    def test_kstest(self, model_namespace, trustyai_service, gaussian_credit_model):
        verify_kstest_request(namespace=model_namespace, model=gaussian_credit_model)


"""

def get_prometheus_k8s_token(duration="1800s"):
    token_command = f"oc create token prometheus-k8s -n openshift-monitoring --duration={duration}"
    try:
        result = subprocess.run(token_command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise AssertionError(f"Command {token_command} failed to execute: {e.stderr}")


def get_prometheus_user_workload_token():
    try:
        # Get the secret name
        secret_command = "oc get secret -n openshift-user-workload-monitoring
         | grep prometheus-user-workload-token | head -n 1 | awk '{print $1}'"
        secret_result = subprocess.run(secret_command, shell=True, check=True, capture_output=True, text=True)
        secret_name = secret_result.stdout.strip()

        # Get the token from the secret
        token_command = f"oc get secret {secret_name} -n openshift-user-workload-monitoring -o json"
        token_result = subprocess.run(token_command, shell=True, check=True, capture_output=True, text=True)
        token_data = json.loads(token_result.stdout)
        token_base64 = token_data['data']['token']
        token = base64.b64decode(token_base64).decode('utf-8')

        # Get the Thanos Querier host
        host_command = "oc get route thanos-querier -n openshift-monitoring -o json"
        host_result = subprocess.run(host_command, shell=True, check=True, capture_output=True, text=True)
        host_data = json.loads(host_result.stdout)
        thanos_querier_host = host_data['spec']['host']

        return token
    except subprocess.CalledProcessError as e:
        raise AssertionError(f"Command failed to execute: {e.stderr}")
"""
