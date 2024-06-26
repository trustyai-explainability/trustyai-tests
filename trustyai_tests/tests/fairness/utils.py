from enum import Enum


class FairnessMetrics(Enum):
    SPD = "spd"
    DIR = "dir"


def get_fairness_metric_endpoint(metric, schedule=False):
    endpoint = f"/metrics/group/fairness/{metric}"
    if schedule:
        endpoint += "/request"

    return endpoint
