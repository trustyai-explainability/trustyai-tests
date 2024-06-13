from enum import Enum


class DriftMetrics(Enum):
    TRUSTYAI_MEANSHIFT = "meanshift"
    TRUSTYAI_FOURIERMMD = "fouriermmd"
    TRUSTYAI_KSTEST = "kstest"
    TRUSTYAI_APPROXKSTEST = "approxkstest"


def get_drift_metric_endpoint(metric, schedule=False):
    endpoint = f"/metrics/drift/{metric}"
    if schedule:
        endpoint += "/request"

    return endpoint
