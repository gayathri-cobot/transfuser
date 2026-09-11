# Model Training

The network used is based on TransFuser.

The network adopts an imitation learning approach where LiDAR and camera data are fused into a shared representation and mapped to waypoints. The network learns to imitate the behavior of the expert.

## Network Architecture

An overview of the method used for waypoint generation is shown in the network above.

There are three auxiliary heads:

- Depth decoder
- Segmentation decoder
- Costmap decoder

The transformers are a series of GPT transformers. The waypoint generation head is a GRU that auto-regressively generates the next waypoint.

## Data Preprocessing

The data preprocessing was done in two ways:

- Time-based sampling
- Distance-based sampling

Each ROSBag was run through a data-preprocessing pipeline that converted the `.mcap` file into a set of RGB images, depth images, LiDAR point clouds and trajectories.

### Camera Throttle Node

Initially the ROSBags recorded contained all three images left, right and front cameras and the Proxie being run in Isaac Sim used all 5 cameras — front, left, right, rear and blindspot. As a result, the RTF of the camera topics in Isaac Sim was very less (~0.06) resulting in very high camera topic frequencies.

The camera frequency is throttled using the `camera_throttle` node with an output frequency of 40 Hz. This node then publishes image topics at the rate of 0.06 × 40 = 24 Hz.

For later ROSBags, only the front camera was used. This resulted in a massive speed up in Isaac Sim that improved the RTF as well. As a result, the `camera_throttle` node was deprecated in later ROSBags.

Some ROSBags contain the recorded topics as:

```python
"/front_camera/color/image_view",
"/front_camera/aligned_depth_to_color/image_rect_raw"
```

while some other bags have:

```python
"/front_camera/color/image_view_throttled",
"/front_camera/aligned_depth_to_color/image_rect_raw_throttled"
```

The preprocessing script accounts for both these variations in recorded topics.

### Time-Based Sampling

Here, the `/front_camera/color/image_view` is chosen as the reference topic. We cap the maximum number of frames per run to be 200. This results in the average sampling frequency to be between 2 Hz and 3 Hz.

> **Note:** A more accurate way of sampling would have been to pick frames with specific time interval between them.

For each reference topic, we look for values of all other topics within a time interval of 10/6 seconds.

Enter the `proxie-transfuser` container via:

```bash
./docker/run_transfuser.sh
```

To run the time-based preprocessing script use:

```bash
python3 scripts/data_preprocess_time_synced_frequency.py /workspace/bag_data/scenario_16/config_1_route1_015422/
```

The pre-processed dataset can be found [here](https://us-west-2.console.aws.amazon.com/s3/buckets/e2e-local-nav-processed-938145530947-us-west-2-an?region=us-west-2&tab=objects) (`s3://e2e-local-nav-processed-938145530947-us-west-2-an`).

### Distance-Based Sampling

Here, the `/tf` is chosen as the reference topic. Using the `tf` tree, the position of the robot in the map frame is computed. The sampling is done such that the minimum distance between the first frame and the second frame is 0.05 meters.

For each reference frame chosen, we check if there is a valid message in every other topic within a time duration of 0.1 seconds. Since the `/costmap` topic is recorded at a lower frequency, the last available message is used.

To run the data preprocessing script based on distance use:

```bash
python3 scripts/data_preprocess_distance_synced_frequency.py /workspace/bag_data/scenario_16/config_1_route1_015422/
```

The pre-processed dataset can be found [here](https://us-west-2.console.aws.amazon.com/s3/buckets/e2e-local-nav-processed-distance?region=us-west-2&tab=objects) (`s3://e2e-local-nav-processed-distance`).

### Pre-processing over recorded ROSBags

To pull the recorded ROSBags from an AWS bucket, run the preprocessing script over them and push the bags back to another bucket use:

> **Note:** The preprocessing only takes place over one scenario at a time.

Log in to AWS:

```bash
export AWS_CONFIG_FILE=docker/aws_config.ini
export AWS_PROFILE=sil-bag-upload
aws sso login 
```
## Training

The hyperparameters used for training are mentioned here.

### Distance-Based Sampling

**Hyperparameters**

| Hyperparameter | Value |
| --- | --- |
| Batch size | 48 (12 × num_gpu) |
| Number of GPUs | 4 |
| Epochs | 71 |
| Image Architecture | regnety_032 |
| LiDAR Architecture | regnety_032 |
| Use velocity | False |
| Learning Rate | 0.0001 |
| Optimizer | AdamW |
| Scheduler | ReduceLROnPlateau |

**Model weights**

`s3://e2e-local-nav-model-weights/sagemaker/transfuser/transfuser-train-2026-09-01-09-38-26/checkpoints/log/transfuser/` → use `model_71.pth`

### Time-Based Sampling

**Hyperparameters**

| Hyperparameter | Value |
| --- | --- |
| Batch size | 48 (12 × num_gpu) |
| Number of GPUs | 4 |
| Epochs | 71 |
| Image Architecture | regnety_032 |
| LiDAR Architecture | regnety_032 |
| Use velocity | False |
| Learning Rate | 0.0001 |
| Optimizer | AdamW |
| Scheduler | ReduceLROnPlateau |

**Model weights**

`s3://e2e-local-nav-model-weights/sagemaker/transfuser/transfuser-train-2026-08-20-11-52-09/checkpoints/log/transfuser/` → use `model_101.pth`

### Running Training

To run training enter the container and run:


```bash
./docker/run_transfuser.sh
python3 scripts/transfuser/train.py --logdir test --epochs 100
```

Note, the current training is done with a batch size of 12 across 4 GPUs. The training leverages AWS SageMaker.

### Running Training on AWS SageMaker

**1. Build and push the Docker image** with all necessary scripts:

```bash
cd /home/gayathrirajesh/repos/transfuser
export AWS_CONFIG_FILE=docker/aws_config.ini AWS_PROFILE=sil-bag-upload
REGION=us-west-2; ACCT=938145530947; REPO=cobot-ai/transfuser-train
TAG=transfuser-$(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)

# login: base image lives in 458214780330, push target is 938145530947
aws ecr get-login-password --region $REGION | docker login -u AWS --password-stdin 458214780330.dkr.ecr.$REGION.amazonaws.com
aws ecr get-login-password --region $REGION | docker login -u AWS --password-stdin $ACCT.dkr.ecr.$REGION.amazonaws.com

# build context is container_files/transfuser (Dockerfile does COPY . /workspace)
docker build -f container_files/transfuser/container/Dockerfile.sagemaker \
  -t $ACCT.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG container_files/transfuser
docker push $ACCT.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG
```

**2. Start the training job** on SageMaker:

```bash
python3 docker/launch_sagemaker_training.py --tag $TAG \
  --epochs 41 --lr 1e-4 --batch-size 12 --setting validate \
  --backbone transFuser --val-every 5 --save-freq 20 --logdir sagemaker_all
```
