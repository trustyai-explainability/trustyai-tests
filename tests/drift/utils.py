import http
import json
import logging
from tests.utils import send_trustyai_service_request
from utilities.constants import (
    TRUSTYAI_UPLOAD_ENDPOINT,
    TRUSTYAI_MEANSHIFT_ENDPOINT,
    TRUSTYAI_FOURIERMMD_ENDPOINT,
    TRUSTYAI_KSTEST_ENDPOINT,
    TRUSTYAI_APPROXKSTEST_ENDPOINT,
)

logger = logging.getLogger(__name__)


def upload_data_to_trustyai_service(namespace, data_path, max_retries=5, retry_delay=1):
    with open(f"{data_path}/training_data.json", "r") as file:
        data = file.read()

    return send_trustyai_service_request(
        namespace=namespace, endpoint=TRUSTYAI_UPLOAD_ENDPOINT, method=http.HTTPMethod.POST, data=data
    )


def verify_meanshift_request(namespace, model):
    response = send_trustyai_service_request(
        namespace=namespace,
        endpoint=f"{TRUSTYAI_MEANSHIFT_ENDPOINT}",
        method=http.HTTPMethod.POST,
        json={"modelId": model.name, "referenceTag": "TRAINING"},
    )
    response_data = json.loads(response.text)
    logger.info(msg="=====================")
    logger.info(msg="Meanshift:")
    logger.info(msg=json.dumps(json.loads(response.text), indent=2))
    logger.info(msg=response.status_code)
    assert response.status_code == http.HTTPStatus.OK
    assert response_data["name"] == "MEANSHIFT"


def verify_fouriermmd_request(namespace, model):
    response = send_trustyai_service_request(
        namespace=namespace,
        endpoint=TRUSTYAI_FOURIERMMD_ENDPOINT,
        method=http.HTTPMethod.POST,
        json={"modelId": model.name, "referenceTag": "TRAINING"},
    )
    response_data = json.loads(response.text)

    logger.info(msg="=====================")
    logger.info(msg="fouriermmd:")
    logger.info(msg=json.dumps(json.loads(response.text), indent=2))
    logger.info(msg=response.status_code)
    assert response.status_code == http.HTTPStatus.OK
    assert response_data["name"] == "FOURIERMMD"


def verify_kstest_request(namespace, model):
    response = send_trustyai_service_request(
        namespace=namespace,
        endpoint=TRUSTYAI_KSTEST_ENDPOINT,
        method=http.HTTPMethod.POST,
        json={"modelId": model.name, "referenceTag": "TRAINING"},
    )
    response_data = json.loads(response.text)

    logger.info(msg="=====================")
    logger.info(msg="kstest:")
    logger.info(msg=json.dumps(json.loads(response.text), indent=2))
    logger.info(msg=response.status_code)
    assert response.status_code == http.HTTPStatus.OK
    assert response_data["name"] == "KSTEST"


def verify_approxkstest_request(namespace, model):
    response = send_trustyai_service_request(
        namespace=namespace,
        endpoint=TRUSTYAI_APPROXKSTEST_ENDPOINT,
        method=http.HTTPMethod.POST,
        json={"modelId": model.name, "referenceTag": "TRAINING"},
    )
    response_data = json.loads(response.text)

    logger.info(msg="=====================")
    logger.info(msg="approxkstest:")
    logger.info(msg=json.dumps(json.loads(response.text), indent=2))
    logger.info(msg=response.status_code)
    assert response.status_code == http.HTTPStatus.OK
    assert response_data["name"] == "APPROXKSTEST"
