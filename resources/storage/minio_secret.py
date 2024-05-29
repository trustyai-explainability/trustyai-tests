from ocp_resources.secret import Secret

from utilities.constants import OPENDATAHUB_IO


class MinioSecret(Secret):
    def __init__(
        self,
        aws_access_key_id,
        aws_default_region,
        aws_s3_bucket,
        aws_s3_endpoint,
        aws_secret_access_key,
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
