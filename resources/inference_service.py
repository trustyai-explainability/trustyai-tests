from ocp_resources.resource import NamespacedResource

from utilities.constants import KSERVE_API_GROUP


# TODO: Move this to openshift-python-wrapper once we are confident
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
        storage_name=None,
        storage_path=None,
        model_format_name=None,
        serving_runtime=None,
        deployment_mode=None,
        **kwargs,
    ):
        """
        InferenceService object

        Args:
            storage_name (str): Name of the data connection (i.e. S3-compatible storage) where the model is located.
            storage_path (str): Path in the data connection where the model is.
            model_format_name (str): Name of the format of the model.
            serving_runtime (str): Name of the serving runtime used to deploy and execute the model.
            deployment_mode (DeploymentMode): Method used to deploy the model in the cluster.
        """
        super().__init__(
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
            self.res["metadata"]["annotations"] = {f"{KSERVE_API_GROUP}/deploymentMode": self.deployment_mode}

        self.res["spec"] = {}
        _spec = self.res["spec"]

        if self.model_format_name:
            _spec.setdefault("predictor", {}).setdefault("model", {}).setdefault("modelFormat", {})["name"] = (
                self.model_format_name
            )

        if self.serving_runtime:
            _spec.setdefault("predictor", {}).setdefault("model", {})["runtime"] = self.serving_runtime

        if self.storage_name:
            _spec.setdefault("predictor", {}).setdefault("model", {}).setdefault("storage", {})["key"] = (
                self.storage_name
            )

        if self.storage_path:
            _spec.setdefault("predictor", {}).setdefault("model", {}).setdefault("storage", {})["path"] = (
                self.storage_path
            )
