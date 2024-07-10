from ocp_resources.namespace import Namespace

from trustyai_tests.tests.minio import MinioService, MinioPod, MinioSecret


def deploy_namespace_with_minio(name):
    namespace = Namespace(name=name, label={"modelmesh-enabled": "true"})
    namespace.deploy()
    namespace.wait_for_status(status=Namespace.Status.ACTIVE)

    deploy_minio(namespace=namespace)

    return namespace


def deploy_minio(namespace):
    minio_service = MinioService(name="minio", port=9000, target_port=9000, namespace=namespace.name)
    minio_pod = MinioPod(
        name="minio", namespace=namespace.name, image="quay.io/trustyai/modelmesh-minio-examples:gauss"
    )
    minio_secret = MinioSecret(
        name="aws-connection-minio-data-connection",
        namespace=namespace.name,
        aws_access_key_id="VEhFQUNDRVNTS0VZ",
        aws_default_region="dXMtc291dGg=",
        aws_s3_bucket="bW9kZWxtZXNoLWV4YW1wbGUtbW9kZWxz",
        aws_s3_endpoint="aHR0cDovL21pbmlvOjkwMDA=",
        aws_secret_access_key="VEhFU0VDUkVUS0VZ",
    )

    for resource in [minio_service, minio_pod, minio_secret]:
        resource.deploy()

    return minio_secret
