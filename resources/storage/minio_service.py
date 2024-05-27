from ocp_resources.service import Service


class MinioService(Service):
    def __init__(
            self,
            name,
            port,
            target_port,
            namespace,
            client,
            **kwargs,
    ):
        super().__init__(name=name, namespace=namespace, client=client, **kwargs)
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
