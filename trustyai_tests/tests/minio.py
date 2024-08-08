from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service import Service

from trustyai_tests.tests.constants import OPENDATAHUB_IO


class MinioPod(Pod):
    def __init__(
        self,
        image: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image = image

    def to_dict(self):
        super().to_dict()
        self.res["metadata"]["labels"] = {
            "app": "minio",
            "maistra.io/expose-route": "true",
        }
        self.res["metadata"]["annotations"] = {
            "sidecar.istio.io/inject": "true",
        }
        self.res["spec"] = {
            "containers": [
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
                    "image": self.image,
                    "name": self.name,
                }
            ]
        }


class MinioSecret(Secret):
    def __init__(
        self,
        aws_access_key_id: str,
        aws_default_region: str,
        aws_s3_bucket: str,
        aws_s3_endpoint: str,
        aws_secret_access_key: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.aws_access_key_id = aws_access_key_id
        self.aws_default_region = aws_default_region
        self.aws_s3_bucket = aws_s3_bucket
        self.aws_s3_endpoint = aws_s3_endpoint
        self.aws_secret_access_key = aws_secret_access_key

    def to_dict(self):
        super().to_dict()

        self.res["metadata"]["labels"] = {
            f"{OPENDATAHUB_IO}/dashboard": "true",
            f"{OPENDATAHUB_IO}/managed": "true",
        }
        self.res["metadata"]["annotations"] = {
            f"{OPENDATAHUB_IO}/connection-type": "s3",
            "openshift.io/display-name": "Minio Data Connection",
        }
        self.res["data"] = {
            # Dummy values
            "AWS_ACCESS_KEY_ID": self.aws_access_key_id,
            "AWS_DEFAULT_REGION": self.aws_default_region,
            "AWS_S3_BUCKET": self.aws_s3_bucket,
            "AWS_S3_ENDPOINT": self.aws_s3_endpoint,
            "AWS_SECRET_ACCESS_KEY": self.aws_secret_access_key,
        }


class MinioService(Service):
    def __init__(
        self,
        port: int,
        target_port: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.port = port
        self.target_port = target_port

    def to_dict(self):
        super().to_dict()

        self.res["spec"] = {
            "ports": [
                {
                    "name": "minio-client-port",
                    "port": self.port,
                    "protocol": "TCP",
                    "targetPort": self.target_port,
                }
            ],
            "selector": {
                "app": "minio",
            },
        }
