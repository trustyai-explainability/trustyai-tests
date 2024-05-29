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

from utils.constants import (
    MM_PAYLOAD_PROCESSORS,
    INFERENCE_ENDPOINT,
    TRUSTYAI_SERVICE,
    TRUSTYAI_MODEL_METADATA_ENDPOINT,
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


def get_ocp_token():
    token = subprocess.check_output(["oc", "whoami", "-t"]).decode().strip()
    return token


def get_trustyai_pod(client, namespace):
    pod = next(
        (
            pod
            for pod in Pod.get(dyn_client=client, namespace=namespace.name)
            if TRUSTYAI_SERVICE in pod.name
        ),
        None,
    )
    if pod is None:
        raise TrustyAIPodNotFoundError(
            f"No TrustyAI pod found in namespace {namespace.name}"
        )
    return pod


def get_trustyai_service_route(namespace):
    try:
        return Route(namespace=namespace.name, name=TRUSTYAI_SERVICE)
    except kubernetes.dynamic.exceptions.NotFoundError as e:
        raise ValueError(
            f"Route '{TRUSTYAI_SERVICE}' not found in namespace '{namespace.name}'"
        ) from e
    except Exception as e:
        raise ValueError(
            f"Unexpected error retrieving route '{TRUSTYAI_SERVICE}' from namespace '{namespace.name}': {str(e)}"
        ) from e


def get_trustyai_model_metadata(namespace):
    return send_trustyai_service_request(
        namespace=namespace,
        endpoint=TRUSTYAI_MODEL_METADATA_ENDPOINT,
        method=http.HTTPMethod.GET,
    )


def send_trustyai_service_request(namespace, endpoint, method, data=None):
    trustyai_service_route = get_trustyai_service_route(namespace=namespace)
    token = get_ocp_token()

    url = f"https://{trustyai_service_route.host}{endpoint}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    response = None
    if method == http.HTTPMethod.GET:
        response = requests.get(url=url, headers=headers, verify=False)
    elif method == http.HTTPMethod.POST:
        response = requests.post(url=url, headers=headers, json=data, verify=False)

    return response


def verify_trustyai_model_metadata(
    namespace, model, data_path, expected_percentage_observations
):
    response = get_trustyai_model_metadata(namespace=namespace)
    assert (
        response.status_code == http.HTTPStatus.OK
    ), f"Expected status code {http.HTTPStatus.OK}, but got {response.status_code}"

    model_input_data = parse_input_data(data_path=data_path)
    model_metadata = parse_trustyai_model_metadata(model_metadata=response.content)

    assert (
        model_metadata.model_name == model.name
    ), f"Expected model name '{model.name}', but got '{model_metadata.model_name}'"
    assert (
        model_metadata.input_tensor_name == model_input_data.name
    ), f"Expected input tensor name '{model_input_data}', but got '{model_metadata.input_tensor_name}'"

    assert (
        model_metadata.num_observations
        > model_input_data.num_observations * expected_percentage_observations
    ), (
        f"Expected number of observations to be greater than "
        f"{model_input_data.num_observations * expected_percentage_observations},"
        f" but got {model_metadata.num_observations}"
    )
    assert (
        model_metadata.num_features == model_input_data.num_features
    ), f"Expected number of features '{model_input_data.num_features}', but got '{model_metadata.num_features}'"


def parse_trustyai_model_metadata(model_metadata):
    json_data = json.loads(model_metadata)

    if isinstance(json_data, list) and len(json_data) > 0:
        data = json_data[0]["data"]
    else:
        raise ValueError("Invalid JSON data format")

    input_tensor_name = data["inputTensorName"]
    output_tensor_name = data["outputTensorName"]
    num_observations = data["observations"]
    model_name = data["modelId"]

    model_metadata = TrustyAIModelMetadata(
        input_tensor_name=input_tensor_name,
        output_tensor_name=output_tensor_name,
        num_observations=num_observations,
        model_name=model_name,
        num_features=len(data["inputSchema"]["items"]),
    )
    return model_metadata


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

            for input_data in data["inputs"]:
                if not name:
                    name = input_data["name"]
                if not num_features:
                    num_features = input_data["shape"][1]
                if not data_type:
                    data_type = input_data["datatype"]

                num_observations = input_data["shape"][0]
                total_observations += num_observations

    model_input_data = ModelInputData(
        name=name,
        num_features=num_features,
        num_observations=total_observations,
        data_type=data_type,
    )
    return model_input_data


def wait_for_model_pods_registered(client, namespace):
    """Wait for model pods to be registered by TrustyAIService"""
    pods_with_env_var = False
    all_pods_running = False
    timeout = 60 * 3
    start_time = time()
    while not pods_with_env_var or not all_pods_running:
        if time() - start_time > timeout:
            raise TimeoutError("Not all model pods are ready in time")

        model_pods = [
            pod
            for pod in Pod.get(dyn_client=client, namespace=namespace.name)
            if "modelmesh-serving" in pod.name
        ]

        pods_with_env_var = False
        all_pods_running = True
        for pod in model_pods:
            try:
                has_env_var = False
                for container in pod.instance.spec.containers:
                    if container.env is not None and any(
                        env.name == MM_PAYLOAD_PROCESSORS for env in container.env
                    ):
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
    namespace, inference_service, data_path, max_retries=5, retry_delay=1
):
    inference_route = Route(namespace=namespace.name, name=inference_service.name)
    token = get_ocp_token()

    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                data = file.read()

            url = f"https://{inference_route.host}{inference_route.instance.spec.path}{INFERENCE_ENDPOINT}"
            headers = {"Authorization": f"Bearer {token}"}

            retry_count = 0
            while (
                retry_count < max_retries
            ):  # Consider using TimeoutSampler in the future
                try:
                    response = requests.post(
                        url=url, headers=headers, data=data, verify=False
                    )
                    response.raise_for_status()
                    if response.status_code == 200:
                        logger.info(f"Successfully sent data for file: {file_name}")
                        break
                except requests.exceptions.RequestException as e:
                    logger.error(
                        f"Error sending data for file: {file_name}. Error: {str(e)}"
                    )
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"Retrying in {retry_delay} second(s)...")
                        sleep(retry_delay)
            else:
                logger.error(f"Maximum retries reached for file: {file_name}")
