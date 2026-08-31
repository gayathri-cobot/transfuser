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
import json
import time

import boto3


DEFAULT_REGION = "us-west-2"
DEFAULT_ACCOUNT_ID = "938145530947"
DEFAULT_ECR_REPO = "cobot-ai/transfuser-train"
DEFAULT_ROLE_ARN = f"arn:aws:iam::{DEFAULT_ACCOUNT_ID}:role/transfuser-sagemaker-execution-role"
# DEFAULT_DATA_S3_URI = "s3://e2e-local-nav-processed-938145530947-us-west-2-an/"
DEFAULT_DATA_S3_URI = "s3://e2e-local-nav-processed-distance/"
DEFAULT_OUTPUT_S3_URI = "s3://e2e-local-nav-model-weights/sagemaker/transfuser"
DEFAULT_ENTRYPOINT = "/workspace/container/entrypoint_sagemaker_train.sh"
DEFAULT_EXCLUDE_TOWNS = ["scenario_1/"]
MANIFEST_KEY = "_manifests/train_manifest.json"


def list_top_level_prefixes(s3, bucket):
    """List the bucket's top-level 'town' prefixes (e.g. 'scenario_2/'), handling pagination."""
    prefixes = []
    token = None
    while True:
        kwargs = dict(Bucket=bucket, Delimiter="/")
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        prefixes.extend(cp["Prefix"] for cp in resp.get("CommonPrefixes", []))
        if not resp.get("IsTruncated"):
            break
        token = resp["NextContinuationToken"]
    return prefixes


def list_all_keys(s3, bucket, exclude_prefixes):
    """List every object key in the bucket, skipping keys under exclude_prefixes.
    SageMaker's ManifestFile format only accepts literal object keys, not
    directory-style trailing-slash prefixes (verified empirically - a manifest
    of trailing-slash entries downloads zero files even though the job reports
    a successful 'Downloading' phase)."""
    keys = []
    token = None
    while True:
        kwargs = dict(Bucket=bucket)
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if not any(key.startswith(p) for p in exclude_prefixes):
                keys.append(key)
        if not resp.get("IsTruncated"):
            break
        token = resp["NextContinuationToken"]
    return keys


def build_train_manifest(s3, bucket, exclude_towns, upload=True):
    """List every object key in the bucket except those under exclude_towns and this
    script's own scratch prefix, and optionally upload the SageMaker manifest JSON.
    Returns (manifest_s3_uri, included_town_prefixes) - included_town_prefixes is just
    for the printed summary, not part of the manifest itself."""
    exclude = set(exclude_towns) | {MANIFEST_KEY.split("/")[0] + "/"}
    all_town_prefixes = list_top_level_prefixes(s3, bucket)
    included_towns = [p for p in all_town_prefixes if p not in exclude]
    keys = list_all_keys(s3, bucket, exclude)
    manifest = [{"prefix": f"s3://{bucket}/"}] + keys
    manifest_uri = f"s3://{bucket}/{MANIFEST_KEY}"
    if upload:
        s3.put_object(Bucket=bucket, Key=MANIFEST_KEY, Body=json.dumps(manifest).encode("utf-8"), ContentType="application/json")
    return manifest_uri, included_towns, len(keys)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="Image tag in the transfuser-train ECR repo, e.g. transfuser-cab68c3-dirty")
    p.add_argument("--job-name", default=None, help="Defaults to transfuser-train-<timestamp>")
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--account-id", default=DEFAULT_ACCOUNT_ID)
    p.add_argument("--ecr-repo", default=DEFAULT_ECR_REPO)
    p.add_argument("--role-arn", default=DEFAULT_ROLE_ARN)
    p.add_argument("--data-s3-uri", default=DEFAULT_DATA_S3_URI, help="root_dir S3 prefix; mounted as the 'train' channel")
    p.add_argument("--exclude-town", nargs="*", default=DEFAULT_EXCLUDE_TOWNS,
                    help="Top-level town prefixes to skip when pulling from S3 (only applies to the default "
                         "--data-s3-uri bucket; ignored for a custom --data-s3-uri). Default: scenario_1/ only "
                         "- scenario_1_2026-07-29/, scenario_1_gray/, scenario_1_lit/ etc. are NOT excluded.")
    p.add_argument("--output-s3-uri", default=DEFAULT_OUTPUT_S3_URI)
    p.add_argument("--instance-type", default="ml.g6e.12xlarge", help="4x L40S")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--volume-size-gb", type=int, default=500, help="EBS size; dataset is ~206GB, G6e has no local NVMe instance store")
    p.add_argument("--max-run-hours", type=int, default=24)
    p.add_argument("--dry-run", action="store_true", help="Print the create_training_job request without submitting it")

    # train.py flags, passed straight through as container arguments.
    p.add_argument("--id", default="transfuser")
    p.add_argument("--epochs", type=int, default=71)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=12, help="Per-GPU; effective batch size = this * num_gpus")
    p.add_argument("--setting", default="validate", choices=["all", "validate"])
    p.add_argument("--backbone", default="transFuser", choices=["transFuser", "late_fusion", "latentTF", "geometric_fusion"])
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--save-freq", type=int, default=20)
    p.add_argument('--image_architecture', type=str, default='resnet34', choices=['efficientnet_b0', 'resnet34', 'regnety_032'])
    p.add_argument('--lidar_architecture', type=str, default='resnet34', choices=['efficientnet_b0', 'resnet34', 'regnety_032'])
    p.add_argument("--logdir", default="log", help="train.py --logdir; checkpoints land under model_ckpt/<logdir>/<id>/")
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
        "--logdir", args.logdir,
        "--image_architecture", args.image_architecture,
        "--lidar_architecture", args.lidar_architecture,
    ]
    container_args.extend(args.extra_args)
    return container_args


def build_data_source(args):
    """S3Prefix for a custom --data-s3-uri (e.g. a smoke-test subset); a generated
    ManifestFile for the default bucket so --exclude-town can skip specific towns
    (S3Prefix mode has no exclusion mechanism)."""
    if args.data_s3_uri != DEFAULT_DATA_S3_URI or not args.exclude_town:
        return dict(S3DataType="S3Prefix", S3Uri=args.data_s3_uri, S3DataDistributionType="FullyReplicated")

    bucket = DEFAULT_DATA_S3_URI[len("s3://"):].rstrip("/")
    exclude_towns = [t if t.endswith("/") else t + "/" for t in args.exclude_town]
    s3 = boto3.client("s3", region_name=args.region)
    manifest_uri, included_towns, num_keys = build_train_manifest(s3, bucket, exclude_towns, upload=not args.dry_run)
    print(f"Excluding towns: {exclude_towns}")
    print(f"Manifest includes {len(included_towns)} towns, {num_keys} objects: {included_towns}")
    return dict(S3DataType="ManifestFile", S3Uri=manifest_uri, S3DataDistributionType="FullyReplicated")


def main():
    args = parse_args()
    job_name = args.job_name or f"transfuser-train-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    image_uri = f"{args.account_id}.dkr.ecr.{args.region}.amazonaws.com/{args.ecr_repo}:{args.tag}"
    checkpoint_s3_uri = f"{args.output_s3_uri}/{job_name}/checkpoints"

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
                DataSource=dict(S3DataSource=build_data_source(args)),
            )
        ],
        OutputDataConfig=dict(S3OutputPath=args.output_s3_uri),
        # entrypoint_sagemaker_train.sh points model_ckpt at CHECKPOINT_DIR
        # (/opt/ml/checkpoints, matched by not passing LocalPath here so the
        # SageMaker default applies), which SageMaker syncs to this S3 URI
        # continuously during training - not just at job end like the final
        # model.tar.gz artifact. This is what makes live TensorBoard possible.
        CheckpointConfig=dict(S3Uri=checkpoint_s3_uri),
        ResourceConfig=dict(
            InstanceType=args.instance_type,
            InstanceCount=args.instance_count,
            VolumeSizeInGB=args.volume_size_gb,
        ),
        StoppingCondition=dict(MaxRuntimeInSeconds=args.max_run_hours * 3600),
    )

    if args.dry_run:
        print(json.dumps(request, indent=2))
        return

    sm = boto3.client("sagemaker", region_name=args.region)
    resp = sm.create_training_job(**request)
    print(f"Launched training job: {job_name}")
    print(f"  Image:      {image_uri}")
    print(f"  ARN:        {resp['TrainingJobArn']}")
    print(f"  Checkpoints (live, synced every ~minute during training): {checkpoint_s3_uri}")
    print(f"  Monitor: aws sagemaker describe-training-job --training-job-name {job_name} --region {args.region}")


if __name__ == "__main__":
    main()
