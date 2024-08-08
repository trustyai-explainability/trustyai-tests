from enum import Enum, auto


class MetricType(Enum):
    FAIRNESS = auto()
    DRIFT = auto()


class Metric(Enum):
    # Fairness metrics
    SPD = ("spd", MetricType.FAIRNESS)
    DIR = ("dir", MetricType.FAIRNESS)

    # Drift metrics
    MEANSHIFT = ("meanshift", MetricType.DRIFT)
    FOURIERMMD = ("fouriermmd", MetricType.DRIFT)
    KSTEST = ("kstest", MetricType.DRIFT)
    APPROXKSTEST = ("approxkstest", MetricType.DRIFT)

    def __init__(self, value: tuple[str, MetricType], metric_type: MetricType):
        self._value_ = value
        self.metric_type = metric_type


def get_metric_endpoint(metric: Metric, schedule: bool = False) -> str:
    base_endpoint = "/metrics/group/fairness" if metric.metric_type == MetricType.FAIRNESS else "/metrics/drift"
    endpoint = f"{base_endpoint}/{metric.value}"

    if schedule:
        endpoint += "/request"

    return endpoint
