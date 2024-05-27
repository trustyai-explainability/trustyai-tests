from ocp_resources.resource import NamespacedResource

from utils.constants import KSERVE_API_GROUP


class ServingRuntime(NamespacedResource):
    """
    ServingRuntime object
    """

    api_group = KSERVE_API_GROUP

    def __init__(
        self,
        name=None,
        namespace=None,
        supported_model_formats=None,
        protocol_versions=None,
        multi_model=None,
        containers=None,
        grpc_endpoint=None,
        grpc_data_endpoint=None,
        server_type=None,
        runtime_mgmt_port=None,
        mem_buffer_bytes=None,
        model_loading_timeout_millis=None,
        enable_route=None,
        yaml_file=None,
        client=None,
        **kwargs,
    ):
        """
        ServingRuntime object

        Args:
            name (str): ServingRuntime name.
            namespace (str): ServingRuntime namespace.
            supported_model_formats (List(dict)): Model formats supported by the serving runtime.
            protocol_versions (List(str)): List of protocols versions used by the serving runtime.
            multi_model (bool): Specifies if the model server can serve multiple models.
            containers (List(dict)): Containers of the serving runtime.
            grpc_endpoint (int): Port of the gRPC endpoint.
            grpc_data_endpoint (int): Port of the gRPC data endpoint.
            server_type (str): Type of the model server.
            runtime_mgmt_port (int): Runtime management port for the model server.
            mem_buffer_bytes (int): Memory buffer bytes.
            model_loading_timeout_millis (int): Model loading timeout in milliseconds
            enable_route (bool): Determines if a route is enabled for the model server.
            yaml_file (yaml): yaml file for the resource.
            client (DynamicClient): DynamicClient to use.
        """
        super().__init__(name=name, namespace=namespace, yaml_file=yaml_file, client=client, **kwargs)
        self.supported_model_formats = supported_model_formats
        self.protocol_versions = protocol_versions,
        self.multi_model = multi_model,
        self.containers = containers
        self.grpc_endpoint = grpc_endpoint
        self.grpc_data_endpoint = grpc_data_endpoint
        self.server_type = server_type
        self.runtime_mgmt_port = runtime_mgmt_port
        self.mem_buffer_bytes = mem_buffer_bytes
        self.model_loading_timeout_millis = model_loading_timeout_millis
        self.enable_route = enable_route

    def to_dict(self):
        super().to_dict()

        if self.enable_route:
            self.res["metadata"].setdefault("annotations", {}).update({
                "enable-route": "true"
            })

        if self.supported_model_formats:
            self.res.setdefault("spec", {})["supportedModelFormats"] = self.supported_model_formats

        if self.protocol_versions:
            self.res.setdefault("spec", {})["protocolVersions"] = self.protocol_versions

        if self.multi_model:
            self.res.setdefault("spec", {})["multiModel"] = True

        if self.grpc_endpoint:
            self.res.setdefault("spec", {})["grpcEndpoint"] = f"port:{self.grpc_endpoint}"

        if self.grpc_data_endpoint:
            self.res.setdefault("spec", {})["grpcDataEndpoint"] = f"port:{self.grpc_data_endpoint}"

        if self.containers:
            self.res.setdefault("spec", {})["containers"] = self.containers

        if self.server_type:
            self.res.setdefault("spec", {}).setdefault("builtInAdapter", {}).update({
                "serverType": self.server_type
            })

        if self.runtime_mgmt_port:
            self.res.setdefault("spec", {}).setdefault("builtInAdapter", {}).update({
                "runtimeManagementPort": self.runtime_mgmt_port
            })

        if self.mem_buffer_bytes:
            self.res.setdefault("spec", {}).setdefault("builtInAdapter", {}).update({
                "memBufferBytes": self.mem_buffer_bytes
            })

        if self.model_loading_timeout_millis:
            self.res.setdefault("spec", {}).setdefault("builtInAdapter", {}).update({
                "modelLoadingTimeoutMillis": self.model_loading_timeout_millis
            })