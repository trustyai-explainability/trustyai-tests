from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service import Service

from trustyai_tests.tests.constants import OPENDATAHUB_IO, MINIO_DATA_CONNECTION_NAME


def create_minio_pod(namespace: Namespace) -> Pod:
    name = "minio"
    containers = [
        {
            "args": [
                "server",
                "/data1",
            ],
            "env": [
                {
                    "name": "MINIO_ACCESS_KEY",
                    "value": "THEACCESSKEY",
                },
                {
                    "name": "MINIO_SECRET_KEY",
                    "value": "THESECRETKEY",
                },
            ],
            "image": "quay.io/trustyai/modelmesh-minio-examples@"
            "sha256:e8360ec33837b347c76d2ea45cd4fea0b40209f77520181b15e534b101b1f323",
            "name": name,
        }
    ]

    return Pod(
        name=name,
        namespace=namespace.name,
        containers=containers,
        label={"app": "minio", "maistra.io/expose-route": "true"},
        annotations={"sidecar.istio.io/inject": "true"},
    )


def create_minio_service(namespace: Namespace) -> Service:
    return Service(
        name="minio",
        namespace=namespace.name,
        ports=[
            {
                "name": "minio-client-port",
                "port": 9000,
                "protocol": "TCP",
                "targetPort": 9000,
            }
        ],
        selector={
            "app": "minio",
        },
    )


def create_minio_secret(namespace: Namespace) -> Secret:
    return Secret(
        name=MINIO_DATA_CONNECTION_NAME,
        namespace=namespace.name,
        data_dict={
            "AWS_ACCESS_KEY_ID": "VEhFQUNDRVNTS0VZ",
            "AWS_DEFAULT_REGION": "dXMtc291dGg=",
            "AWS_S3_BUCKET": "bW9kZWxtZXNoLWV4YW1wbGUtbW9kZWxz",
            "AWS_S3_ENDPOINT": "aHR0cDovL21pbmlvOjkwMDA=",
            "AWS_SECRET_ACCESS_KEY": "VEhFU0VDUkVUS0VZ",
        },
        label={
            f"{OPENDATAHUB_IO}/dashboard": "true",
            f"{OPENDATAHUB_IO}/managed": "true",
        },
        annotations={
            f"{OPENDATAHUB_IO}/connection-type": "s3",
            "openshift.io/display-name": "Minio Data Connection",
        },
    )
