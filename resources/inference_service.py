from ocp_resources.resource import NamespacedResource

from utils.constants import KSERVE_API_GROUP


class InferenceService(NamespacedResource):
    """
    InferenceService object
    """

    api_group = KSERVE_API_GROUP

    class DeploymentMode:
        """
        DeploymentMode object
        """

        MODEL_MESH = "ModelMesh"

    def __init__(
        self,
        name=None,
        namespace=None,
        storage_name=None,
        storage_path=None,
        model_format_name=None,
        serving_runtime=None,
        deployment_mode=None,
        yaml_file=None,
        client=None,
        **kwargs,
    ):
        """
        InferenceService object

        Args:
            name (str): InferenceService name.
            namespace (str): InferenceService namespace.
            storage_name (str): Name of the data connection (i.e. S3-compatible storage) where the model is located.
            storage_path (str): Path in the data connection where the model is.
            model_format_name (str): Name of the format of the model.
            serving_runtime (str): Name of the serving runtime used to deploy and execute the model.
            deployment_mode (DeploymentMode): Method used to deploy the model in the cluster.
            yaml_file (yaml): yaml file for the resource.
            client (DynamicClient): DynamicClient to use.
        """
        super().__init__(
            name=name,
            namespace=namespace,
            yaml_file=yaml_file,
            client=client,
            **kwargs,
        )
        self.storage_name = storage_name
        self.storage_path = storage_path
        self.model_format_name = model_format_name
        self.serving_runtime = serving_runtime
        self.deployment_mode = deployment_mode

    def to_dict(self) -> None:
        super().to_dict()

        if self.deployment_mode:
            self.res.setdefault("metadata", {}).setdefault("annotations", {}).update({
                f"{KSERVE_API_GROUP}/deploymentMode": self.deployment_mode
            })

        if self.model_format_name:
            self.res.setdefault("spec", {}).setdefault("predictor", {}).setdefault("model", {}).setdefault(
                "modelFormat", {})["name"] = self.model_format_name

        if self.serving_runtime:
            self.res.setdefault("spec", {}).setdefault("predictor", {}).setdefault("model", {})[
                "runtime"] = self.serving_runtime

        if self.storage_name:
            self.res.setdefault("spec", {}).setdefault("predictor", {}).setdefault("model", {}).setdefault("storage",
                                                                                                           {})[
                "key"] = self.storage_name

        if self.storage_path:
            self.res.setdefault("spec", {}).setdefault("predictor", {}).setdefault("model", {}).setdefault("storage",
                                                                                                           {})[
                "path"] = self.storage_path

