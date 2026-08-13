#!/usr/bin/env python3
"""Launch a TransFuser training job on SageMaker.

Uses the boto3 SageMaker client directly (create_training_job) rather than the
`sagemaker` SDK, mirroring the shape of this account's existing openpi training
jobs: a custom ECR image with a baked-in entrypoint, driven by
ContainerEntrypoint/ContainerArguments instead of the hyperparameters.json
contract. train.py itself is never modified - all of its flags are passed
through as container arguments.

Usage:
  AWS_CONFIG_FILE=docker/aws_config.ini AWS_PROFILE=sil-bag-upload \
    python3 docker/launch_sagemaker_training.py --tag transfuser-cab68c3-dirty --epochs 41

Data/model paths, role, and instance type all have defaults matching the
current setup (see the SageMaker plan this script was built from); override
via flags if any of that changes.
"""
import argparse
import time

import boto3


DEFAULT_REGION = "us-west-2"
DEFAULT_ACCOUNT_ID = "938145530947"
DEFAULT_ECR_REPO = "cobot-ai/transfuser-train"
DEFAULT_ROLE_ARN = f"arn:aws:iam::{DEFAULT_ACCOUNT_ID}:role/transfuser-sagemaker-execution-role"
DEFAULT_DATA_S3_URI = "s3://e2e-local-nav-processed-938145530947-us-west-2-an/"
DEFAULT_OUTPUT_S3_URI = "s3://e2e-local-nav-model-weights/sagemaker/transfuser"
DEFAULT_ENTRYPOINT = "/workspace/container/entrypoint_sagemaker_train.sh"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="Image tag in the transfuser-train ECR repo, e.g. transfuser-cab68c3-dirty")
    p.add_argument("--job-name", default=None, help="Defaults to transfuser-train-<timestamp>")
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--account-id", default=DEFAULT_ACCOUNT_ID)
    p.add_argument("--ecr-repo", default=DEFAULT_ECR_REPO)
    p.add_argument("--role-arn", default=DEFAULT_ROLE_ARN)
    p.add_argument("--data-s3-uri", default=DEFAULT_DATA_S3_URI, help="root_dir S3 prefix; mounted as the 'train' channel")
    p.add_argument("--output-s3-uri", default=DEFAULT_OUTPUT_S3_URI)
    p.add_argument("--instance-type", default="ml.g6e.12xlarge", help="4x L40S")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--volume-size-gb", type=int, default=500, help="EBS size; dataset is ~206GB, G6e has no local NVMe instance store")
    p.add_argument("--max-run-hours", type=int, default=24)
    p.add_argument("--dry-run", action="store_true", help="Print the create_training_job request without submitting it")

    # train.py flags, passed straight through as container arguments.
    p.add_argument("--id", default="transfuser")
    p.add_argument("--epochs", type=int, default=41)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=12, help="Per-GPU; effective batch size = this * num_gpus")
    p.add_argument("--setting", default="validate", choices=["all", "validate"])
    p.add_argument("--backbone", default="transFuser", choices=["transFuser", "late_fusion", "latentTF", "geometric_fusion"])
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--save-freq", type=int, default=20)
    p.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[], help="Any additional train.py flags, passed through verbatim")
    return p.parse_args()


def build_container_arguments(args):
    # --parallel_training/--root_dir are set by entrypoint_sagemaker_train.sh
    # itself (GPU count + $SM_CHANNEL_TRAIN), so they're deliberately omitted here.
    container_args = [
        "--id", args.id,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch_size", str(args.batch_size),
        "--setting", args.setting,
        "--backbone", args.backbone,
        "--val_every", str(args.val_every),
        "--save_freq", str(args.save_freq),
    ]
    container_args.extend(args.extra_args)
    return container_args


def main():
    args = parse_args()
    job_name = args.job_name or f"transfuser-train-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    image_uri = f"{args.account_id}.dkr.ecr.{args.region}.amazonaws.com/{args.ecr_repo}:{args.tag}"

    request = dict(
        TrainingJobName=job_name,
        RoleArn=args.role_arn,
        AlgorithmSpecification=dict(
            TrainingImage=image_uri,
            TrainingInputMode="File",
            ContainerEntrypoint=[DEFAULT_ENTRYPOINT],
            ContainerArguments=build_container_arguments(args),
        ),
        InputDataConfig=[
            dict(
                ChannelName="train",
                DataSource=dict(
                    S3DataSource=dict(
                        S3DataType="S3Prefix",
                        S3Uri=args.data_s3_uri,
                        S3DataDistributionType="FullyReplicated",
                    )
                ),
            )
        ],
        OutputDataConfig=dict(S3OutputPath=args.output_s3_uri),
        ResourceConfig=dict(
            InstanceType=args.instance_type,
            InstanceCount=args.instance_count,
            VolumeSizeInGB=args.volume_size_gb,
        ),
        StoppingCondition=dict(MaxRuntimeInSeconds=args.max_run_hours * 3600),
    )

    if args.dry_run:
        import json
        print(json.dumps(request, indent=2))
        return

    sm = boto3.client("sagemaker", region_name=args.region)
    resp = sm.create_training_job(**request)
    print(f"Launched training job: {job_name}")
    print(f"  Image:  {image_uri}")
    print(f"  ARN:    {resp['TrainingJobArn']}")
    print(f"  Monitor: aws sagemaker describe-training-job --training-job-name {job_name} --region {args.region}")


if __name__ == "__main__":
    main()
