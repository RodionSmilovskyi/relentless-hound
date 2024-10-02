import os
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.debugger import TensorBoardOutputConfig

ROLE = "arn:aws:iam::905418352696:role/SageMakerFullAccess"
BASE_JOB_NAME = "pytorch-dqn-cartpole"

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
    image_uri="905418352696.dkr.ecr.us-east-1.amazonaws.com/ai-repo:pytorch_dqn_cartpole.b78edef",
    role=ROLE,
    max_run=1 * 60 * 60,
    base_job_name=BASE_JOB_NAME,
    instance_count=1,
    checkpoint_s3_uri=os.path.join(default_path, "checkpoints", BASE_JOB_NAME),
    container_arguments=["train", "--prefix", "/opt/ml", "--use-virtual-display"],
    tensorboard_output_config=tensorboard_output_config,
    instance_type="ml.g4dn.2xlarge",
)


estimator.fit(wait=False)

print(f"Job {BASE_JOB_NAME} submitted")
