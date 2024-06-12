from ocp_resources.pod import Pod


class MinioPod(Pod):
    def __init__(
        self,
        image,
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
