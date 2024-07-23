import http
import json
import logging
import os
import subprocess
from time import time, sleep

import kubernetes
import requests
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_utilities.monitoring import Prometheus

from trustyai_tests.tests.constants import (
    TRUSTYAI_SERVICE,
)

logger = logging.getLogger(__name__)


class TrustyAIPodNotFoundError(Exception):
    pass


class ModelInputData:
    def __init__(self, name, num_features, num_observations, data_type):
        self.name = name
        self.num_features = num_features
        self.num_observations = num_observations
        self.data_type = data_type


class TrustyAIModelMetadata:
    def __init__(
        self,
        input_tensor_name,
        output_tensor_name,
        num_observations,
        model_name,
        num_features,
    ):
        self.input_tensor_name = input_tensor_name
        self.output_tensor_name = output_tensor_name
        self.num_observations = num_observations
        self.model_name = model_name
        self.num_features = num_features


def get_ocp_token(namespace):
    return subprocess.check_output(["oc", "create", "token", "test-user", "-n", namespace.name]).decode().strip()


def get_trustyai_pod(namespace):
    for pod in Pod.get(namespace=namespace.name):
        if TRUSTYAI_SERVICE in pod.name:
            return pod

    raise TrustyAIPodNotFoundError(f"No TrustyAI pod found in namespace {namespace.name}")


def get_trustyai_service_route(namespace):
    return Route(namespace=namespace.name, name=TRUSTYAI_SERVICE, ensure_exists=True)


def get_trustyai_model_metadata(namespace):
    return send_trustyai_service_request(
        namespace=namespace,
        endpoint="/info",
        method=http.HTTPMethod.GET,
    )


def send_trustyai_service_request(namespace, endpoint, method, data=None, json=None):
    trustyai_service_route = get_trustyai_service_route(namespace=namespace)
    token = get_ocp_token(namespace=namespace)

    url = f"https://{trustyai_service_route.host}{endpoint}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    if method == http.HTTPMethod.GET:
        return requests.get(url=url, headers=headers, verify=False)
    elif method == http.HTTPMethod.POST:
        return requests.post(url=url, headers=headers, data=data, json=json, verify=False)
    raise ValueError(f"Unsupported HTTP method: {method}")


def verify_trustyai_model_metadata(namespace, model, data_path, expected_percentage_observations):
    response = get_trustyai_model_metadata(namespace=namespace)
    assert (
        response.status_code == http.HTTPStatus.OK
    ), f"Expected status code {http.HTTPStatus.OK}, but got {response.status_code}"
    model_input_data = parse_input_data(data_path=data_path)
    model_metadata_list = parse_trustyai_model_metadata(model_metadata=response.content)

    model_metadata = next((m for m in model_metadata_list if m.model_name == model.name), None)

    assert (
        model_metadata.model_name == model.name
    ), f"Expected model name '{model.name}', but got '{model_metadata.model_name}'"
    assert model_metadata.num_observations > model_input_data.num_observations * expected_percentage_observations, (
        f"Expected number of observations to be greater than "
        f"{model_input_data.num_observations * expected_percentage_observations},"
        f" but got {model_metadata.num_observations}"
    )
    assert (
        model_metadata.num_features == model_input_data.num_features
    ), f"Expected number of features '{model_input_data.num_features}', but got '{model_metadata.num_features}'"


def parse_trustyai_model_metadata(model_metadata):
    json_data = json.loads(model_metadata)

    model_metadata_list = []

    for item in json_data:
        try:
            data = item["data"]
        except (IndexError, KeyError) as exp:
            raise ValueError(f"Invalid JSON data format. {exp}")

        input_tensor_name = data["inputTensorName"]
        output_tensor_name = data["outputTensorName"]
        num_observations = data["observations"]
        model_name = data["modelId"]

        model_metadata_list.append(
            TrustyAIModelMetadata(
                input_tensor_name=input_tensor_name,
                output_tensor_name=output_tensor_name,
                num_observations=num_observations,
                model_name=model_name,
                num_features=len(data["inputSchema"]["items"]),
            )
        )

    return model_metadata_list


def parse_input_data(data_path):
    total_observations = 0
    name = ""
    num_features = 0
    data_type = ""

    for filename in os.listdir(data_path):
        if filename.endswith(".json"):
            file_path = os.path.join(data_path, filename)

            with open(file_path, "r") as file:
                data = json.load(file)

            # Check if "inputs" is directly available or nested inside "request"
            inputs = data.get("inputs", data.get("request", {}).get("inputs"))

            if inputs:
                for input_data in inputs:
                    name = name or input_data["name"]
                    num_features = num_features or input_data["shape"][1]
                    data_type = data_type or input_data["datatype"]

                    num_observations = input_data["shape"][0]
                    total_observations += num_observations

    return ModelInputData(
        name=name,
        num_features=num_features,
        num_observations=total_observations,
        data_type=data_type,
    )


def wait_for_modelmesh_pods_registered(namespace):
    """Wait for modelmesh pods to be registered by TrustyAIService"""
    pods_with_env_var = False
    all_pods_running = False
    timeout = 60 * 10
    start_time = time()
    while not pods_with_env_var or not all_pods_running:  # TODO: Consider using TimeoutSampler in the future
        if time() - start_time > timeout:
            raise TimeoutError("Not all model pods are ready in time")

        model_pods = [pod for pod in Pod.get(namespace=namespace.name) if "modelmesh-serving" in pod.name]

        pods_with_env_var = False
        all_pods_running = True
        for pod in model_pods:
            try:
                has_env_var = False
                for container in pod.instance.spec.containers:
                    if container.env is not None and any(env.name == "MM_PAYLOAD_PROCESSORS" for env in container.env):
                        has_env_var = True
                        break

                if has_env_var:
                    pods_with_env_var = True
                    if pod.status != Pod.Status.RUNNING:
                        all_pods_running = False
                        break
            except kubernetes.dynamic.exceptions.NotFoundError:
                # Ignore the error if the pod is not found (deleted during the process)
                continue

        if not pods_with_env_var or not all_pods_running:
            sleep(5)


def send_data_to_inference_service(
    namespace, inference_service, data_path, max_retries=5, retry_delay=1, num_batches=None
):
    inference_route = Route(namespace=namespace.name, name=inference_service.name)
    token = get_ocp_token(namespace=namespace)

    files_processed = 0
    for root, _, files in os.walk(data_path):
        for file_name in files:
            if num_batches is not None and files_processed >= num_batches:
                logger.info(f"Reached the specified number of batches ({num_batches}). Stopping processing.")
                return

            file_path = os.path.join(root, file_name)
            with open(file_path, "r") as file:
                data = file.read()

            url = f"https://{inference_route.host}{inference_route.instance.spec.path}/infer"
            headers = {"Authorization": f"Bearer {token}"}

            retry_count = 0
            while retry_count < max_retries:
                try:
                    response = requests.post(url=url, headers=headers, data=data, verify=False)
                    response.raise_for_status()
                    if response.status_code == 200:
                        logger.info(f"Successfully sent data for file: {file_name}")
                        break
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error sending data for file: {file_name}. Error: {str(e)}")
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"Retrying in {retry_delay} second(s)...")
                        sleep(retry_delay)
                sleep(5)
            else:
                logger.error(f"Maximum retries reached for file: {file_name}")

            files_processed += 1


def upload_data_to_trustyai_service(namespace, data_path):
    with open(f"{data_path}", "r") as file:
        data = file.read()

    logger.info(msg="Uploading data to TrustyAI Service.")
    return send_trustyai_service_request(
        namespace=namespace, endpoint="/data/upload", method=http.HTTPMethod.POST, data=data
    )


def verify_metric_request(namespace, model, endpoint, expected_metric_name, json_data):
    """
    Send a basic metric request to TrustyAI and validates the response.

    :param namespace (Namespace): Namespace where TrustyAIService and the corresponding InferenceService live.
    :param model (InferenceService): InferenceService for which the metric has been calculated.
    :param endpoint (str): TrustyAI endpoint to request the metric.
    :param expected_metric_name (str): Metric name used to validate the response
    """

    logger.info(f"Sending TrustyAI metric request: {endpoint}")
    response = send_trustyai_service_request(
        namespace=namespace,
        endpoint=endpoint,
        method=http.HTTPMethod.POST,
        json=json_data,
    )
    response_data = json.loads(response.text)
    logger.info(msg=f"Response: {json.dumps(json.loads(response.text), indent=2)}")

    assert response.status_code == http.HTTPStatus.OK, f"Unexpected status code: {response.status_code}"
    assert response_data["timestamp"] != "", "Timestamp is empty"
    assert response_data["type"] == "metric", "Incorrect type"
    assert response_data["value"] != "", "Value is empty"
    assert response_data["namedValues"] != "", "Named values are empty"
    assert response_data["specificDefinition"] != "", "Specific definition is empty"
    assert (
        response_data["name"] == expected_metric_name
    ), f"Wrong name: {response_data['name']}, expected: {expected_metric_name}"
    assert response_data["id"] != "", "ID is empty"
    assert response_data["thresholds"] != "", "Thresholds are empty"


def verify_metric_scheduling(namespace, model, endpoint, json_data):
    """
    Send a request to schedule a metric to TrustyAI and validates the response.

    :param namespace (Namespace): Namespace where TrustyAIService and the corresponding InferenceService live.
    :param model (InferenceService): InferenceService for which the metric has been calculated.
    :param endpoint (str): TrustyAI endpoint to request the metric.
    :param expected_metric_name (str): Metric name used to validate the response
    """

    logger.info(f"Sending TrustyAI metric request: {endpoint}")
    response = send_trustyai_service_request(
        namespace=namespace,
        endpoint=endpoint,
        method=http.HTTPMethod.POST,
        json=json_data,
    )
    response_data = json.loads(response.text)

    logger.info(msg=f"Response: {json.dumps(json.loads(response.text), indent=2)}")

    assert response.status_code == http.HTTPStatus.OK, f"Unexpected status code: {response.status_code}"
    assert response_data["requestId"] != "", "Request ID is empty"
    assert response_data["timestamp"] != "", "Timestamp is empty"


def verify_trustyai_metric_prometheus(namespace, model, prometheus_query, metric_name, max_retries=20, retry_delay=2):
    """
    Sends a query to Prometheus for a specific TrustyAI metric and verifies the result for a specific model.

    :param namespace (Namespace): Namespace where TrustyAIService and InferenceService live
    :param model (InferenceService): InferenceService for which the metric has been calculated
    :param prometheus_query (str): Prometheus query
    :param metric_name (str): Name of the metric
    :param max_retries (int): Maximum number of retries
    :param retry_delay (int): Delay between retries in seconds
    """

    prom_token = get_prometheus_token()
    prom = Prometheus(verify_ssl=False, bearer_token=prom_token)

    logger.info(f"Sending Prometheus query: {prometheus_query}")

    retry_count = 0
    while retry_count < max_retries:
        result = prom.query(query=prometheus_query)

        if result["status"] == "success" and len(result["data"]["result"]) > 0:
            break

        retry_count += 1
        if retry_count < max_retries:
            logger.info(f"No Prometheus data for the metric {metric_name}. Retrying in {retry_delay} second(s)...")
            sleep(retry_delay)
        else:
            assert False, f"No Prometheus data for the metric {metric_name} after {max_retries} retries."

    logger.info(msg=json.dumps(result, indent=4))

    assert (
        result["status"] == "success"
    ), f"Unexpected status in result. Expected: 'success', Actual: {result['status']}"

    # Find the item corresponding to the specified model
    model_data = next((item for item in result["data"]["result"] if item["metric"]["model"] == model.name), None)

    assert model_data is not None, f"No data found for model: {model.name}"

    expected_metric_name = f"trustyai_{metric_name.lower()}"
    assert (
        model_data["metric"]["__name__"] == expected_metric_name
    ), f"Incorrect metric name. Expected: {expected_metric_name}, Actual: {model_data['metric']['__name__']}"
    assert model_data["metric"]["batch_size"] != "", "Batch size is empty"
    assert (
        model_data["metric"]["job"] == TRUSTYAI_SERVICE
    ), f"Incorrect job name. Expected: {TRUSTYAI_SERVICE}, Actual: {model_data['metric']['job']}"

    expected_metric_name_upper = metric_name.upper()
    assert model_data["metric"]["metricName"] == expected_metric_name_upper, (
        f"Incorrect metric name capitalization. "
        f"Expected: {expected_metric_name_upper}, Actual: {model_data['metric']['metricName']}"
    )
    assert (
        model_data["metric"]["model"] == model.name
    ), f"Incorrect model name. Expected: {model.name}, Actual: {model_data['metric']['model']}"
    assert model_data["metric"]["namespace"] == namespace.name, (
        f"Incorrect namespace. " f"Expected: {namespace.name}, Actual: {model_data['metric']['namespace']}"
    )

    expected_pod_name = get_trustyai_pod(namespace=namespace).name
    assert (
        model_data["metric"]["pod"] == expected_pod_name
    ), f"Incorrect pod name. Expected: {expected_pod_name}, Actual: {model_data['metric']['pod']}"
    assert (
        model_data["metric"]["request"] != ""
    ), "Request is empty"  # TODO: Try to find a way to get the requestId for this
    assert (
        model_data["metric"]["service"] == TRUSTYAI_SERVICE
    ), f"Incorrect service name. Expected: {TRUSTYAI_SERVICE}, Actual: {model_data['metric']['service']}"
    assert model_data["value"] != "", "Value is empty"


def get_prometheus_token(duration="1800s"):
    token_command = f"oc create token prometheus-k8s -n openshift-monitoring --duration={duration}"
    try:
        result = subprocess.run(token_command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command {token_command} failed to execute: {e.stderr}")


def apply_trustyai_name_mappings(namespace, inference_service, input_mappings, output_mappings):
    data = {"modelId": inference_service.name, "inputMapping": input_mappings, "outputMapping": output_mappings}

    response = send_trustyai_service_request(
        namespace=namespace, endpoint="/info/names", method=http.HTTPMethod.POST, json=data
    )
    assert response.status_code == http.HTTPStatus.OK, f"Wrong status code: {response.status_code}"
