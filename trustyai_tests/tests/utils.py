import http
import json
import logging
import os
import subprocess
from time import time, sleep
from typing import Any

import kubernetes
import requests
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_utilities.monitoring import Prometheus

from trustyai_tests.tests.constants import (
    TRUSTYAI_SERVICE,
    MODEL_DATA_PATH,
    KNATIVE_API_GROUP,
    ODH_OPERATOR,
    RHOAI_OPERATOR,
)

logger: logging.Logger = logging.getLogger(__name__)


class TrustyAIPodNotFoundError(Exception):
    pass


class ModelInputData:
    def __init__(self, name: str, num_features: int, num_observations: int, data_type: str):
        self.name = name
        self.num_features = num_features
        self.num_observations = num_observations
        self.data_type = data_type


class TrustyAIModelMetadata:
    def __init__(
        self,
        input_tensor_name: str,
        output_tensor_name: str,
        num_observations: int,
        model_name: str,
        num_features: int,
    ):
        self.input_tensor_name = input_tensor_name
        self.output_tensor_name = output_tensor_name
        self.num_observations = num_observations
        self.model_name = model_name
        self.num_features = num_features


def is_odh_or_rhoai():
    operators = {ODH_OPERATOR, RHOAI_OPERATOR}
    for csv in ClusterServiceVersion.get():
        for operator in operators:
            if operator in csv.name:
                return operator
    raise RuntimeError("Neither ODH nor RHOAI operators are installed.")


def get_ocp_token(namespace: Namespace) -> str:
    return subprocess.check_output(["oc", "create", "token", "test-user", "-n", namespace.name]).decode().strip()


def get_trustyai_pod(namespace: Namespace) -> Pod:
    for pod in Pod.get(namespace=namespace.name):
        if TRUSTYAI_SERVICE in pod.name:
            return pod

    raise TrustyAIPodNotFoundError(f"No TrustyAI pod found in namespace {namespace.name}")


def get_trustyai_service_route(namespace: Namespace) -> Route:
    return Route(namespace=namespace.name, name=TRUSTYAI_SERVICE, ensure_exists=True)


def get_trustyai_model_metadata(namespace: Namespace) -> Any:
    return send_trustyai_service_request(
        namespace=namespace,
        endpoint="/info",
        method="GET",
    )


def send_trustyai_service_request(
    namespace: Namespace, endpoint: str, method: str, data: Any = None, json: Any = None
) -> Any:
    trustyai_service_route = get_trustyai_service_route(namespace=namespace)
    token = get_ocp_token(namespace=namespace)

    url = f"https://{trustyai_service_route.host}{endpoint}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    if method == "GET":
        return requests.get(url=url, headers=headers, verify=False)
    elif method == "POST":
        return requests.post(url=url, headers=headers, data=data, json=json, verify=False)
    raise ValueError(f"Unsupported HTTP method: {method}")


def verify_trustyai_model_metadata(
    namespace: Namespace, model: InferenceService, data_path: str, num_batches: int = None
):
    response = get_trustyai_model_metadata(namespace=namespace)
    logger.info(msg=response.text)
    logger.info(msg=response.content)
    logger.info(msg=response.json)
    assert (
        response.status_code == http.HTTPStatus.OK
    ), f"Expected status code {http.HTTPStatus.OK}, but got {response.status_code}"

    model_input_data = parse_input_data(data_path=data_path, num_batches=num_batches)
    model_metadata_list = parse_trustyai_model_metadata(model_metadata=response.content)

    model_metadata = next((m for m in model_metadata_list if m.model_name == model.name), None)

    if model_metadata is None:
        raise ValueError(f"No metadata found for model '{model.name}'")

    assert (
        model_metadata.model_name == model.name
    ), f"Expected model name '{model.name}', but got '{model_metadata.model_name}'"
    assert model_metadata.num_observations == model_input_data.num_observations, (
        f"Expected number of observations to be "
        f"{model_input_data.num_observations},"
        f" but got {model_metadata.num_observations}"
    )
    assert (
        model_metadata.num_features == model_input_data.num_features
    ), f"Expected number of features '{model_input_data.num_features}', but got '{model_metadata.num_features}'"


def parse_trustyai_model_metadata(model_metadata: Any) -> list[TrustyAIModelMetadata]:
    if isinstance(model_metadata, bytes):
        model_metadata = model_metadata.decode("utf-8")

    if isinstance(model_metadata, str):
        json_data = json.loads(model_metadata)
    else:
        json_data = model_metadata

    logger.info(f"Model metadata: {json_data}")

    model_metadata_list = []

    # Handle both list and dict inputs
    if isinstance(json_data, list):
        items = json_data
    elif isinstance(json_data, dict):
        items = json_data.items()
    else:
        raise ValueError("Invalid input format. Expected list or dict.")

    for item in items:
        if isinstance(item, dict):
            # Handle the first input format
            model_id = item.get("data", {}).get("modelId")
            data = item.get("data", {})
        else:
            # Handle the second input format
            model_id, item_data = item
            data = item_data.get("data", {})

        try:
            input_tensor_name = data["inputTensorName"]
            output_tensor_name = data["outputTensorName"]
            num_observations = data["observations"]
            model_name = model_id or "unknown"
            num_features = len(data["inputSchema"]["items"])
        except KeyError as exp:
            raise ValueError(f"Invalid JSON data format. {exp}")

        model_metadata_list.append(
            TrustyAIModelMetadata(
                input_tensor_name=input_tensor_name,
                output_tensor_name=output_tensor_name,
                num_observations=num_observations,
                model_name=model_name,
                num_features=num_features,
            )
        )

    return model_metadata_list


def parse_input_data(data_path: str, num_batches: int = None) -> ModelInputData:
    total_observations = 0
    name = ""
    num_features = 0
    data_type = ""
    processed_files = 0

    for root, dirs, files in os.walk(data_path):
        for filename in files:
            if filename.endswith(".json"):
                if num_batches is not None and processed_files >= num_batches:
                    break

                file_path = os.path.join(root, filename)

                with open(file_path, "r") as file:
                    data = json.load(file)

                inputs = data.get("inputs", data.get("request", {}).get("inputs"))

                if inputs:
                    for input_data in inputs:
                        name = name or input_data["name"]
                        num_features = num_features or input_data["shape"][1]
                        data_type = data_type or input_data["datatype"]

                        num_observations = input_data["shape"][0]
                        total_observations += num_observations

                processed_files += 1

        if num_batches is not None and processed_files >= num_batches:
            break

    return ModelInputData(
        name=name,
        num_features=num_features,
        num_observations=total_observations,
        data_type=data_type,
    )


def wait_for_modelmesh_pods_registered(namespace: Namespace) -> None:
    """Wait for modelmesh pods to be registered by TrustyAIService"""
    pods_with_env_var = False
    all_pods_running = False
    timeout = 60 * 20
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

    sleep(120)


def send_data_to_inference_service(
    namespace: Namespace,
    inference_service: InferenceService,
    data_path: str,
    max_retries: int = 5,
    retry_delay: int = 1,
    num_batches: int = None,
) -> None:
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
                    logger.info(msg=response.text)
                    logger.info(msg=response.content)
                    logger.info(msg=response.json)

                    response.raise_for_status()
                    if response.status_code == 200:
                        logger.info(f"Successfully sent data for file: {file_name}")
                        break
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error sending data for file: {file_name}. Error: {str(e)}")
                    retry_count += 1
                    if retry_count < max_retries:
                        sleep(retry_delay)
            else:
                logger.error(f"Maximum retries reached for file: {file_name}")

            files_processed += 1
            sleep(30)


def upload_data_to_trustyai_service(namespace: Namespace, data_path: str) -> Any:
    with open(f"{data_path}", "r") as file:
        data = file.read()

    logger.info(msg="Uploading data to TrustyAI Service.")
    response = send_trustyai_service_request(namespace=namespace, endpoint="/data/upload", method="POST", data=data)
    logger.info(msg=response.text)
    logger.info(msg=response.content)
    logger.info(msg=response.json)
    return response


def verify_metric_request(namespace: Namespace, endpoint: str, expected_metric_name: str, json_data: Any) -> None:
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
        method="POST",
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


def verify_metric_scheduling(namespace: Namespace, endpoint: str, json_data: Any) -> None:
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
        method="POST",
        json=json_data,
    )
    response_data = json.loads(response.text)

    logger.info(msg=f"Response: {json.dumps(json.loads(response.text), indent=2)}")

    assert response.status_code == http.HTTPStatus.OK, f"Unexpected status code: {response.status_code}"
    assert response_data["requestId"] != "", "Request ID is empty"
    assert response_data["timestamp"] != "", "Timestamp is empty"


def verify_trustyai_metric_prometheus(
    namespace: Namespace,
    model: InferenceService,
    prometheus_query: str,
    metric_name: str,
    max_retries: int = 20,
    retry_delay: int = 5,
) -> None:
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


def get_prometheus_token(duration: int = "1800s") -> str:
    token_command = f"oc create token prometheus-k8s -n openshift-monitoring --duration={duration}"
    try:
        result = subprocess.run(token_command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command {token_command} failed to execute: {e.stderr}")


def apply_trustyai_name_mappings(
    namespace: Namespace, inference_service: InferenceService, input_mappings: Any, output_mappings: Any
) -> None:
    data = {"modelId": inference_service.name, "inputMapping": input_mappings, "outputMapping": output_mappings}

    response = send_trustyai_service_request(namespace=namespace, endpoint="/info/names", method="POST", json=data)
    logger.info(msg=response.text)
    logger.info(msg=response.content)
    logger.info(msg=response.json)
    assert response.status_code == http.HTTPStatus.OK, f"Wrong status code: {response.status_code}"


def get_kserve_route(model_namespace: str, model: InferenceService) -> Any:
    """
    Gets the hostname of a model deployed on KServe.

    :param model_namespace (str): Namespace where the model lives.
    :param model (InferenceService): Name of model that is deployed.
    """
    try:
        k8s_client = kubernetes.config.load_incluster_config()
    except kubernetes.config.ConfigException:
        k8s_client = kubernetes.config.load_kube_config()
    dyn_client = kubernetes.dynamic.DynamicClient(
        client=kubernetes.client.api_client.ApiClient(configuration=k8s_client)
    )

    route = Route(
        namespace=model_namespace,
        name=model,
        client=dyn_client,
        context="kind-kind",
        api_group=KNATIVE_API_GROUP,
        ensure_exists=True,
    )
    return route.instance.status.url


def verify_model_prediction(model_namespace: str, model: InferenceService) -> None:
    """
    Verifies output of KServe explainers' "predict" endpoint.

    :param model_namespace (str): Namespace where the model lives.
    :param model (InferenceService): Name of the predictor that is deployed
    """
    data = json.loads(f"{MODEL_DATA_PATH}/bank-inference_data.json")
    service_hostname = get_kserve_route(model=model, model_namespace=model_namespace)
    response = requests.post(
        f"http://localhost:8080/v1/models/{model}:predict",
        data=data,
        headers={
            "Host": service_hostname.status.url,
            "Content-Type": "application/json",
        },
        timeout=10,
    )
    assert response.status_code == http.HTTPStatus.OK
    assert (list(response.json().keys()))[0] == "predictions", f"Unexpected type: {list(response.json().keys())[0]}"
    assert len(response.json()["predictions"]) != 0, "Predictions is empty."


def verify_saliency_explanation(model_namespace: str, model: InferenceService) -> None:
    """
    Verifies output of KServe explainers' "explain" endpoint.

     param model_namespace (str): Namespace where model lives.
    :param model (InferenceService): Name of explainer that is deployed
    """
    data = json.loads(f"{MODEL_DATA_PATH}/bank-inference_data.json")
    service_hostname = get_kserve_route(model_namespace=model_namespace, model=model)
    response = requests.post(
        f"http://localhost:8080/v1/models/{model}:explain",
        data=data,
        headers={
            "Host": service_hostname.status.url,
            "Content-Type": "application/json",
        },
        timeout=10,
    )
    assert response.status_code == http.HTTPStatus.OK
    assert response.json()["type"] == "explanation", f"Unexpected type: {response.json()['type']}"
    assert len(response.json()["saliences"]) != 0
    for item in response.json()["saliencies"]["outputs-0"]:
        assert list(item.keys()) == ["name", "score", "confidence"], f"Unexpected saliency results: {list(item.keys())}"
