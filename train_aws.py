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

base_job_name = "tf-rl-train-job"

s3_output_bucket = os.path.join(
    "s3://", sagemaker_session.default_bucket(), base_job_name
)
date_str = datetime.now().strftime("%d-%m-%Y")
time_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
job_name = f"{base_job_name}-{time_str}"
output_path = os.path.join(s3_output_bucket, "sagemaker-output", date_str, job_name)

tensorboard_output_config = TensorBoardOutputConfig(
    s3_output_path=os.path.join(output_path, "tensorboard"),
    container_local_output_path="/opt/ml/output/tensorboard",
)

estimator = Estimator(
    sagemaker_session=sagemaker_session,
    image_uri="905418352696.dkr.ecr.us-east-1.amazonaws.com/ai-repo:py-bullet",
    role=role,
    base_job_name=base_job_name,
    instance_count=1,
    container_arguments=["train", "--episodes", "2", "--prefix", "/opt/ml"],
    tensorboard_output_config=tensorboard_output_config,
    instance_type="ml.g4dn.2xlarge",
)


estimator.fit(wait=False)

print(f'Job {job_name} submitted')
