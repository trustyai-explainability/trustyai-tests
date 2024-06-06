# Serving Runtimes
OVMS = "ovms"
OVMS_RUNTIME_NAME = f"{OVMS}-1.x"
OVMS_QUAY_IMAGE = "quay.io/opendatahub/openvino_model_server:stable"
OPENVINO_MODEL_FORMAT = "openvino_ir"

MLSERVER = "mlserver"
MLSERVER_RUNTIME_NAME = f"{MLSERVER}-1.x"
MLSERVER_QUAY_IMAGE = "quay.io/aaguirre/mlserver:1.3.2"

# Model Format
ONNX = "onnx"
SKLEARN = "sklearn"
XGBOOST = "xgboost"

# API Groups
KSERVE_API_GROUP = "serving.kserve.io"
TRUSTYAI_API_GROUP = "trustyai.opendatahub.io"

# TrustyAI
TRUSTYAI_SERVICE = "trustyai-service"
TRUSTYAI_SERVICE_IMAGE = "quay.io/trustyaiservice/trustyai-service"
MM_PAYLOAD_PROCESSORS = "MM_PAYLOAD_PROCESSORS"

# TrustyAI Endpoints
TRUSTYAI_SPD_ENDPOINT = "/metrics/group/fairness/spd/"
TRUSTYAI_NAMES_ENDPOINT = "/info/names"
TRUSTYAI_MODEL_METADATA_ENDPOINT = "/info"
TRUSTYAI_UPLOAD_ENDPOINT = "/data/upload"
TRUSTYAI_METRICS_DRIFT = "/metrics/drift"
TRUSTYAI_MEANSHIFT_ENDPOINT = f"{TRUSTYAI_METRICS_DRIFT}/meanshift"
TRUSTYAI_FOURIERMMD_ENDPOINT = f"{TRUSTYAI_METRICS_DRIFT}/fouriermmd"
TRUSTYAI_KSTEST_ENDPOINT = f"{TRUSTYAI_METRICS_DRIFT}/kstest"
TRUSTYAI_APPROXKSTEST_ENDPOINT = f"{TRUSTYAI_METRICS_DRIFT}/approxkstest"

# InferenceService
INFERENCE_ENDPOINT = "/infer"

# Minio
MINIO_IMAGE = "quay.io/trustyai/modelmesh-minio-examples:gauss"

# Protocol Versions
GRPC = "grpc"

# OpenDataHub
OPENDATAHUB_IO = "opendatahub.io"
