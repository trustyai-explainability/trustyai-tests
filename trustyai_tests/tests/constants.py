TRUSTYAI_SERVICE: str = "trustyai-service"

OPENVINO_MODEL_FORMAT: str = "openvino_ir"

KSERVE_API_GROUP: str = "serving.kserve.io"

TRUSTYAI_API_GROUP: str = "trustyai.opendatahub.io"

OPENDATAHUB_IO: str = "opendatahub.io"

MODEL_DATA_PATH: str = "./trustyai_tests/model_data"

MINIO_DATA_CONNECTION_NAME: str = "aws-connection-minio-data-connection"

KNATIVE_API_GROUP: str = "serving.knative.dev"

ODH_OPERATOR: str = "opendatahub-operator"

RHOAI_OPERATOR: str = "rhods-operator"

ONNX: str = "onnx"
OVMS: str = "ovms"
OVMS_RUNTIME_NAME: str = f"{OVMS}-1.x"
OVMS_QUAY_IMAGE: str = (
    "quay.io/opendatahub/openvino_model_server@sha256:564664371d3a21b9e732a5c1b4b40bacad714a5144c0a9aaf675baec4a04b148"
)
ONNX_LOAN_MODEL_ALPHA_PATH: str = "onnx/loan_model_alpha_august.onnx"
