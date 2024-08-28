import boto3
import sagemaker
from datetime import datetime
from sagemaker.estimator import Estimator
from sagemaker.debugger import TensorBoardOutputConfig
import os

role = "arn:aws:iam::905418352696:role/SageMakerFullAccess"
boto_session = boto3.Session(
    profile_name="905418352696_AdministratorAccess", region_name="us-east-1"
)
sagemaker_session = sagemaker.Session(boto_session=boto_session)

base_job_name = "hover-128-128-10000-5000"


date_str = datetime.now().strftime("%d-%m-%Y")
time_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
job_name = f"{base_job_name}-{time_str}"

s3_output_bucket = os.path.join("s3://", sagemaker_session.default_bucket(), job_name)

tensorboard_output_config = TensorBoardOutputConfig(
    s3_output_path=os.path.join(s3_output_bucket, "tensorboard"),
    container_local_output_path="/opt/ml/output/tensorboard",
)

estimator = Estimator(
    sagemaker_session=sagemaker_session,
    image_uri="905418352696.dkr.ecr.us-east-1.amazonaws.com/ai-repo:hover.592c16b",
    role=role,
    max_run=24 * 60 * 60,
    base_job_name=base_job_name,
    instance_count=1,
    container_arguments=[
        "train",
        "--episodes",
        "5001",
        "--episode-trigger-step",
        "100",
        "--neurons",
        "128",
        "--batch-size",
        "128",
        "--buffer-size",
        "10000",
        "--prefix",
        "/opt/ml",
    ],
    tensorboard_output_config=tensorboard_output_config,
    instance_type="ml.g4dn.2xlarge",
)


estimator.fit(wait=False)

print(f"Job {job_name} submitted")
