from ocp_resources.resource import NamespacedResource

from trustyai_tests.utilities.constants import TRUSTYAI_API_GROUP, TRUSTYAI_SERVICE_IMAGE


# TODO: Move this to openshift-python-wrapper once we are confident
class TrustyAIService(NamespacedResource):
    """
    TrustyAIService object
    """

    api_group = TRUSTYAI_API_GROUP

    def __init__(
        self,
        replicas=1,
        image=TRUSTYAI_SERVICE_IMAGE,
        tag="latest",
        storage_format=None,
        storage_folder=None,
        storage_size=None,
        data_filename=None,
        data_format=None,
        metrics_schedule_interval=None,
        **kwargs,
    ):
        """
        TrustyAIService object

        Args:
            replicas (int, default: 1): Number of replicas for the TrustyAIService.
            image (str): Pull url of the TrustyAIService.
            tag (str): Tag of the image.
            storage_format (str): Format for the TrustyAIService storage.
            storage_folder (str): Folder for the TrustyAIService storage.
            storage_size (str): Size for the TrustyAIService storage.
            data_filename (str): File where the TrustyAIService data is stored.
            data_format (str): Format of the file where the TrustyAIService data is stored.
            metrics_schedule_interval (str): Time interval in seconds for TrustyAIService metrics.
        """
        super().__init__(**kwargs)
        self.replicas = replicas
        self.image = image
        self.tag = tag
        self.storage_format = storage_format
        self.storage_folder = storage_folder
        self.storage_size = storage_size
        self.data_filename = data_filename
        self.data_format = data_format
        self.metrics_schedule_interval = metrics_schedule_interval

    def to_dict(self):
        super().to_dict()

        self.res["spec"] = {}
        _spec = self.res["spec"]

        _spec["replicas"] = self.replicas
        _spec["image"] = self.image
        _spec["tag"] = self.tag

        if self.storage_format:
            _spec.setdefault("storage", {})["format"] = self.storage_format

        if self.storage_folder:
            _spec.setdefault("storage", {})["folder"] = self.storage_folder

        if self.storage_size:
            _spec.setdefault("storage", {})["size"] = self.storage_size

        if self.data_filename:
            _spec.setdefault("data", {})["filename"] = self.data_filename

        if self.data_format:
            _spec.setdefault("data", {})["format"] = self.data_format

        if self.metrics_schedule_interval:
            _spec.setdefault("metrics", {})["schedule"] = self.metrics_schedule_interval
