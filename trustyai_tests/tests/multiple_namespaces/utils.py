from ocp_resources.namespace import Namespace
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount

from trustyai_tests.tests.minio import create_minio_service, create_minio_pod, create_minio_secret


def deploy_namespace_with_minio(name: str) -> Namespace:
    namespace = Namespace(name=name, label={"modelmesh-enabled": "true"})
    namespace.deploy()
    namespace.wait_for_status(status=Namespace.Status.ACTIVE)
    deploy_service_account(namespace=namespace)
    deploy_minio(namespace=namespace)

    return namespace


def deploy_minio(namespace: Namespace) -> Secret:
    minio_service = create_minio_service(namespace=namespace)
    minio_pod = create_minio_pod(namespace=namespace)
    minio_secret = create_minio_secret(namespace=namespace)

    for resource in [minio_service, minio_pod, minio_secret]:
        resource.deploy()

    return minio_secret


def deploy_service_account(namespace: Namespace) -> None:
    user_name = "test-user"
    service_account = ServiceAccount(name=user_name, namespace=namespace.name)
    service_account.deploy()
    role_binding = RoleBinding(
        name="test-user-view",
        namespace=namespace.name,
        subjects_kind="ServiceAccount",
        subjects_name=user_name,
        role_ref_kind="ClusterRole",
        role_ref_name="view",
    )
    role_binding.deploy()
