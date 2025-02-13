import os
from time import sleep
from typing import Any, Generator, Optional

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ConflictError
from ocp_resources.config_map import ConfigMap
from ocp_resources.maria_db import MariaDB
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import get_client
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ocp_resources.trustyai_service import TrustyAIService

from trustyai_tests.tests.constants import (
    TRUSTYAI_SERVICE,
    ODH_OPERATOR,
)
from trustyai_tests.tests.minio import create_minio_secret, create_minio_pod, create_minio_service
from trustyai_tests.tests.utils import (
    wait_for_mariadb_pods,
    log_namespace_pods,
    log_namespace_events,
    log_namespace_logs,
    per_test_artifacting_logic,
)
from trustyai_tests.tests.utils import logger, is_odh_or_rhoai, wait_for_trustyai_pod_running


@pytest.fixture(autouse=True)
def test_log(request):
    name = request.node.nodeid
    spacing = "=" * len(name)
    padding = "=" * (130 - len(name))
    logger.info(
        f"\n"
        f"=============={spacing}=============={padding}\n"
        f"======= Test '{name}' STARTED ===={padding}\n"
        f"=============={spacing}=============={padding}\n"
    )
    yield
    logger.info(f"\n======= Test '{name}' COMPLETED =={padding}\n\n")


def pytest_addoption(parser):
    parser.addoption(
        "--use-modelmesh-image", action="store_true", default=False, help="Include modelMeshImage in the ConfigMap"
    )


@pytest.fixture(scope="session")
def use_modelmesh_image(request):
    return request.config.getoption("--use-modelmesh-image")


@pytest.fixture(scope="session")
def client() -> DynamicClient:
    yield get_client()


@pytest.fixture(autouse=True)
def test_logging(request, client):
    per_test_artifacting_logic(request, client, ["pre-test"])
    yield
    per_test_artifacting_logic(request, client, ["post-test"])


@pytest.fixture(autouse=True, scope="session")
def modelmesh_configmap(use_modelmesh_image) -> Optional[ConfigMap]:
    operator = is_odh_or_rhoai()
    namespace = Namespace(
        name="opendatahub" if operator == ODH_OPERATOR else "redhat-ods-applications", ensure_exists=True
    )

    config_data = {
        "podsPerRuntime": 1,
    }

    if use_modelmesh_image:
        config_data["modelMeshImage"] = {"name": "quay.io/opendatahub/modelmesh", "tag": "fast"}

    with ConfigMap(
        name="model-serving-config",
        namespace=namespace.name,
        data={"config.yaml": yaml.dump(config_data)},
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def model_namespace(client: DynamicClient) -> Namespace:
    with Namespace(
        client=client,
        name="test-namespace",
        delete_timeout=600,
        annotations={
            "openshift.io/description": "",
            "openshift.io/display-name": "",
            "openshift.io/requester": "htpasswd-cluster-admin-user",
        },
        label={"modelmesh-enabled": "true"},
    ) as ns:
        ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=120)
        user_name = "test-user"
        service_account = ServiceAccount(name=user_name, namespace=ns.name)
        service_account.deploy()
        role_binding = RoleBinding(
            name="test-user-view",
            namespace=ns.name,
            subjects_kind="ServiceAccount",
            subjects_name=user_name,
            role_ref_kind="ClusterRole",
            role_ref_name="view",
        )
        role_binding.deploy()
        yield ns


@pytest.fixture(scope="class")
def db_credentials(model_namespace):
    with Secret(
        name="db-credentials",
        namespace=model_namespace.name,
        string_data={
            "databaseKind": "mariadb",
            "databaseName": "trustyai_database",
            "databaseUsername": "quarkus",
            "databasePassword": "quarkus",
            "databaseService": "mariadb",
            "databasePort": "3306",
            "databaseGeneration": "update",
        },
    ) as db_credentials:
        yield db_credentials


@pytest.fixture(scope="class")
def mariadb(model_namespace, db_credentials) -> MariaDB:
    with MariaDB(yaml_file="trustyai_tests/manifests/mariadb.yaml") as mariadb:
        wait_for_mariadb_pods(mariadb=mariadb)
        sleep(60)
        yield mariadb


@pytest.fixture(scope="class")
def modelmesh_serviceaccount(client: DynamicClient, model_namespace: Namespace) -> Any:
    with ServiceAccount(client=client, name="modelmesh-serving-sa", namespace=model_namespace.name):
        yield


@pytest.fixture(scope="session")
def cluster_monitoring_config(client: DynamicClient) -> ConfigMap:
    config_yaml = yaml.dump({"enableUserWorkload": "true"})
    name = "cluster-monitoring-config"
    namespace = "openshift-monitoring"
    try:
        with ConfigMap(
            name=name,
            namespace=namespace,
            data={"config.yaml": config_yaml},
        ) as cm:
            yield cm
    except ConflictError:
        yield ConfigMap(name=name, namespace=namespace)


@pytest.fixture(scope="session")
def user_workload_monitoring_config(client: DynamicClient) -> ConfigMap:
    config_yaml = yaml.dump({"prometheus": {"logLevel": "debug", "retention": "15d"}})
    name = "user-workload-monitoring-config"
    namespace = "openshift-user-workload-monitoring"
    try:
        with ConfigMap(
            name=name,
            namespace=namespace,
            data={"config.yaml": config_yaml},
        ) as cm:
            yield cm
    except ConflictError:
        yield ConfigMap(name=name, namespace=namespace)


@pytest.fixture(scope="class")
def trustyai_service_pvc(
    client: DynamicClient,
    model_namespace: Namespace,
    modelmesh_serviceaccount: Any,
    cluster_monitoring_config: ConfigMap,
    user_workload_monitoring_config: ConfigMap,
) -> TrustyAIService:
    with TrustyAIService(
        client=client,
        name=TRUSTYAI_SERVICE,
        namespace=model_namespace.name,
        storage={"format": "PVC", "folder": "/inputs", "size": "1Gi"},
        data={"filename": "data.csv", "format": "CSV"},
        metrics={"schedule": "5s"},
    ) as trusty:
        wait_for_trustyai_pod_running(namespace=model_namespace)
        yield trusty


@pytest.fixture(scope="class")
def trustyai_service_db(
    client: DynamicClient,
    model_namespace: Namespace,
    mariadb,
    modelmesh_serviceaccount: Any,
    cluster_monitoring_config: ConfigMap,
    user_workload_monitoring_config: ConfigMap,
) -> TrustyAIService:
    with TrustyAIService(
        client=client,
        name=TRUSTYAI_SERVICE,
        namespace=model_namespace.name,
        storage={"format": "DATABASE", "databaseConfigurations": "db-credentials"},
        metrics={"schedule": "5s"},
    ) as trusty:
        wait_for_trustyai_pod_running(namespace=model_namespace)
        yield trusty


@pytest.fixture(scope="class")
def minio_service(client: DynamicClient, model_namespace: Namespace) -> Generator[Service, Any, None]:
    with create_minio_service(namespace=model_namespace) as minio_service:
        yield minio_service


@pytest.fixture(scope="class")
def minio_pod(client: DynamicClient, model_namespace: Namespace) -> Generator[Pod, Any, None]:
    with create_minio_pod(namespace=model_namespace) as minio_pod:
        yield minio_pod


@pytest.fixture(scope="class")
def minio_secret(client: DynamicClient, model_namespace: Namespace) -> Generator[Secret, Any, None]:
    with create_minio_secret(namespace=model_namespace) as minio_secret:
        yield minio_secret


@pytest.fixture(scope="class")
def minio_data_connection(minio_service: Service, minio_pod: Pod, minio_secret: Secret) -> Generator[Secret, Any, None]:
    yield minio_secret
