import os
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.debugger import TensorBoardOutputConfig

ROLE = "arn:aws:iam::905418352696:role/SageMakerFullAccess"
BASE_JOB_NAME = "reinforce-500-1"

boto_session = boto3.Session(
    profile_name="905418352696_AdministratorAccess", region_name="us-east-1"
)

sagemaker_session = sagemaker.Session(boto_session=boto_session)

tensorboard_output_config = TensorBoardOutputConfig(
    s3_output_path=os.path.join("s3://", sagemaker_session.default_bucket()),
    container_local_output_path="/opt/ml/output/tensorboard",
)

estimator = Estimator(
    sagemaker_session=sagemaker_session,
    image_uri="905418352696.dkr.ecr.us-east-1.amazonaws.com/ai-repo:reinforce_discrete.cc28b05",
    role=ROLE,
    max_run=24 * 60 * 60,
    base_job_name=BASE_JOB_NAME,
    instance_count=1,
    container_arguments=[
        "train",
        "--episodes",
        "500",
        "--episode-trigger-step",
        "10",
        "--neurons",
        "100",
        "--batch-size",
        "128",
        "--buffer-size",
        "2000",
        "--prefix",
        "/opt/ml",
    ],
    tensorboard_output_config=tensorboard_output_config,
    instance_type="ml.g4dn.2xlarge",
)


estimator.fit(wait=False)

print(f"Job {BASE_JOB_NAME} submitted")
