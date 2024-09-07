import os
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.debugger import TensorBoardOutputConfig

ROLE = "arn:aws:iam::905418352696:role/SageMakerFullAccess"
BASE_JOB_NAME = "reinforce-hover-2"

boto_session = boto3.Session(
    profile_name="905418352696_AdministratorAccess", region_name="us-east-1"
)

sagemaker_session = sagemaker.Session(boto_session=boto_session)
default_path = os.path.join("s3://", sagemaker_session.default_bucket())

tensorboard_output_config = TensorBoardOutputConfig(
    s3_output_path=default_path,
    container_local_output_path="/opt/ml/output/tensorboard",
)

estimator = Estimator(
    sagemaker_session=sagemaker_session,
    image_uri="905418352696.dkr.ecr.us-east-1.amazonaws.com/ai-repo:reinforce_discrete.ae0ab33",
    role=ROLE,
    max_run=24 * 60 * 60,
    base_job_name=BASE_JOB_NAME,
    instance_count=1,
    checkpoint_s3_uri=os.path.join(default_path, 'checkpoints', BASE_JOB_NAME),
    container_arguments=[
        "train",
        "--episodes",
        "200",
        "--collect-episodes-per-iteration",
        "10",
        "--validation-episode",
        "5",
        "--buffer-size",
        "100",
        "--learning-rate",
        "0.001",
        "--entropy-rate",
        "0.1",
        "--prefix",
        "/opt/ml",
    ],
    tensorboard_output_config=tensorboard_output_config,
    instance_type="ml.g4dn.2xlarge",
)


estimator.fit(wait=False)

print(f"Job {BASE_JOB_NAME} submitted")
