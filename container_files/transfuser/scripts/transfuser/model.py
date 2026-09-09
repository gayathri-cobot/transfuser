from collections import deque
import torch.nn.functional as F
import cv2

from utils import *
from transfuser import TransfuserBackbone, SegDecoder, DepthDecoder
from copy import deepcopy
from point_pillar import PointPillarNet


from PIL import Image, ImageFont, ImageDraw
from torchvision import models

# Copyright (c) OpenMMLab. All rights reserved.
import math
from functools import partial

import torch
import torch.nn as nn
import numpy as np
from torchvision.ops import batched_nms

# eval metrics
import metrics

class LidarCenterNet(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        in_channels: input channels
    """

    def __init__(self, config, device, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=True):
        super().__init__()
        self.device = device
        print("Using device:", self.device)
        self.config = config
        self.pred_len = config.pred_len
        self.use_target_point_image = config.use_target_point_image
        self.gru_concat_target_point = config.gru_concat_target_point
        self.use_point_pillars = config.use_point_pillars

        if(self.use_point_pillars == True):
            self.point_pillar_net = PointPillarNet(config.num_input, config.num_features,
                                                   min_x = config.min_x, max_x = config.max_x,
                                                   min_y = config.min_y, max_y = config.max_y,
                                                   pixels_per_meter = int(config.pixels_per_meter),
                                                  )



        self._model = TransfuserBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
       
        print("Using backbone TransFuser")

        if config.multitask:
            self.seg_decoder   = SegDecoder(self.config,   self.config.perception_output_features).to(self.device)
            self.depth_decoder = DepthDecoder(self.config, self.config.perception_output_features).to(self.device)

        channel = config.channel

        self.pred_bev = nn.Sequential(
                            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(channel, 2, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        ).to(self.device)

        # prediction heads
        self.i = 0

        # waypoints prediction
        self.join = nn.Sequential(
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True),
                        ).to(self.device)

        self.decoder = nn.GRUCell(input_size=4 if self.gru_concat_target_point else 2, # 2 represents x,y coordinate
                                  hidden_size=self.config.gru_hidden_size).to(self.device)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(self.config.gru_hidden_size, 3).to(self.device)

    def forward_gru(self, z, target_point):
        z = self.join(z)
    
        output_wp = list()
        
        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(z.device)

        
        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            if self.gru_concat_target_point:
                x_in = torch.cat([x, target_point], dim=1)
            else:
                x_in = x
            
            z = self.decoder(x_in, z)
            dx = self.output(z)
            
            x = dx[:,:2] + x
            
            output_wp.append(x[:,:2])
            
        pred_wp = torch.stack(output_wp, dim=1)


        pred_brake = None
        steer = None
        throttle = None
        brake = None

        return pred_wp, pred_brake, steer, throttle, brake

    
    def forward_ego(self, rgb, lidar_bev, target_point, target_point_image, ego_vel, save_path="/workspace/viz/", expert_waypoints=None,
         bev_gt=None, depth=None, semantic=None  ,num_points=None, debug=True, eval=False):
        
        if(self.use_point_pillars == True):
            lidar_bev = self.point_pillar_net(lidar_bev, num_points)
            lidar_bev = torch.rot90(lidar_bev, -1, dims=(2, 3)) #For consitency this is also done in voxelization

        if self.use_target_point_image:
            lidar_bev = torch.cat((lidar_bev, target_point_image), dim=1)

        
        features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        
        pred_wp, _, _, _, _ = self.forward_gru(fused_features, target_point)

        self.i += 1
  
        if debug or eval:
            pred_bev = self.pred_bev(features[0])
            pred_bev = F.interpolate(pred_bev, (self.config.bev_resolution_height, self.config.bev_resolution_width), mode='bilinear', align_corners=True)
            pred_semantic = self.seg_decoder(image_features_grid)
            pred_depth = self.depth_decoder(image_features_grid)

            if debug and self.i % 2 != 0 and not (save_path is None):
                self.visualize_model_io(save_path, self.i, self.config, rgb, lidar_bev, target_point,
                                pred_wp, pred_bev, pred_semantic, pred_depth, self.device,
                                expert_waypoints=expert_waypoints)
            if eval:
                bev_iou_mean, bev_iou_std = metrics.compute_IoU(pred_bev.argmax(dim=1).cpu().numpy(), bev_gt.cpu().numpy())
                seg_iou_mean, seg_iou_std = metrics.compute_IoU(pred_semantic.argmax(dim=1).cpu().numpy(), semantic.cpu().numpy())
                depth_rmse_mean, depth_rmse_std = metrics.compute_RMSE(pred_depth.cpu().numpy(), depth.cpu().numpy())
                # ade_path = metrics.compute_ADE(pred_wp.cpu().numpy(), expert_waypoints.cpu().numpy())
                similarity_mean, similarity_std = metrics.compute_fretchet(pred_wp.cpu().numpy(), expert_waypoints.cpu().numpy())
                return pred_wp, {'bev_iou_mean': bev_iou_mean, 'bev_iou_std': bev_iou_std, 'seg_iou_mean': seg_iou_mean, 'seg_iou_std': seg_iou_std, 'depth_rmse_mean': depth_rmse_mean, 'depth_rmse_std': depth_rmse_std, 'similarity_mean': similarity_mean, 'similarity_std': similarity_std}

        return pred_wp, None

    def forward(self, rgb, lidar_bev, ego_waypoint, target_point, target_point_image, ego_vel, bev, depth, semantic, num_points=None, save_path=None, ):
        loss = {}

        if(self.use_point_pillars == True):
            lidar_bev = self.point_pillar_net(lidar_bev, num_points)
            lidar_bev = torch.rot90(lidar_bev, -1, dims=(2, 3)) #For consitency this is also done in voxelization


        if self.use_target_point_image:
            lidar_bev = torch.cat((lidar_bev, target_point_image), dim=1)

        
        features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        

        pred_wp, _, _, _, _ = self.forward_gru(fused_features, target_point)

        # pred topdown view
        pred_bev = self.pred_bev(features[0])
        pred_bev = F.interpolate(pred_bev, (self.config.bev_resolution_height, self.config.bev_resolution_width), mode='bilinear', align_corners=True)

        # free:occupied pixel ratio measured across all 17 routes in data/ is ~4.16:1, not 11:1
        weight = torch.from_numpy(np.array([1., 3.])).to(dtype=torch.float32, device=pred_bev.device)
        #4.16:1

        loss_bev = F.cross_entropy(pred_bev, bev, weight=weight).mean()

        loss_wp = torch.mean(torch.abs(pred_wp - ego_waypoint))
        loss.update({
            "loss_wp": loss_wp,
            "loss_bev": loss_bev
        })

        if self.config.multitask:
            pred_semantic = self.seg_decoder(image_features_grid)
            pred_depth = self.depth_decoder(image_features_grid)
            # non-floor:floor pixel ratio measured across all 16 routes in data/ is ~5.4:1 - floor (class 1) is
            # the minority class here (front camera mostly shows walls/ceiling), opposite of the BEV imbalance
            # semantic_weight = torch.from_numpy(np.array([1., 5.41])).to(dtype=torch.float32, device=pred_semantic.device)
            # loss_semantic = self.config.ls_seg * F.cross_entropy(pred_semantic, semantic).mean() # unweighted, ignored the class imbalance
            loss_semantic = self.config.ls_seg * F.cross_entropy(pred_semantic, semantic).mean()

            loss_depth = self.config.ls_depth * F.l1_loss(pred_depth, depth).mean()
            loss.update({
                "loss_depth": loss_depth,
                "loss_semantic": loss_semantic
            })
        else:
            loss.update({
                "loss_depth": torch.zeros_like(loss_wp),
                "loss_semantic": torch.zeros_like(loss_wp)
            })

        self.i += 1
        if ((self.config.debug == True) and (self.i % self.config.train_debug_save_freq == 0) and (save_path != None)):
            with torch.no_grad():
                self.visualize_model_io(save_path, self.i, self.config, rgb, lidar_bev, target_point,
                                   pred_wp, pred_bev, pred_semantic, pred_depth, self.device, expert_waypoints=ego_waypoint,)

        return loss


    def draw_waypoints(self, waypoints, image, color = (255, 255, 255)):
        waypoints = waypoints.detach().cpu().numpy()

        for points in  waypoints:

            # convert to image space
            # need to negate y componet as we do for lidar points
            # we directly construct points in the image coordiante
            # for lidar, forward +x, top +y
            #            y
            #            +
            #            |
            #            |
            #            |---------+x
            #
            # for image, ---------> x
            #            |
            #            |
            #            +
            #            y

            points[:, 0] += self.config.lidar_pos[0]
            points[:, 1] *= -1
            points = points * self.config.pixels_per_meter
            points[:, 1] += int(self.config.lidar_resolution_height / 2.0)

            points_to_draw = []
            for point in points[:, :2]:
                points_to_draw.append(point.copy())
                point = point.astype(np.int32)
                cv2.circle(image, tuple(point), radius=3, color=color, thickness=3)
        return image


    def draw_target_point(self, target_point, image, color = (255, 255, 255)):
        target_point = target_point.copy()

        target_point[0] += self.config.lidar_pos[0]
        point = target_point * self.config.pixels_per_meter
        point[1] *= -1
        point[1] += int(self.config.lidar_resolution_height / 2.0)
        point = point.astype(np.int32)
        point = np.clip(point, 0, 512)
        cv2.circle(image, tuple(point), radius=5, color=color, thickness=3)
        return image

    def visualize_model_io(self, save_path, step, config, rgb, lidar_bev, target_point,
                        pred_wp, pred_bev, pred_semantic, pred_depth, device, expert_waypoints=None):
        font = ImageFont.load_default()
        i = 0 # We only visualize the first image if there is a batch of them.
        frame_id = step // 2

        # print(target_point, expert_waypoints)

        if config.multitask:
            classes_list = config.classes_list
            converter = np.array(classes_list)

            dataset_size = (config.img_resolution[1], config.img_resolution[0]) # cv2 wants (width, height)

            depth_image = pred_depth[i].detach().cpu().numpy()
            depth_image = np.stack((depth_image, depth_image, depth_image), axis=2)
            depth_image = (depth_image * 255).astype(np.uint8)
            depth_image = cv2.resize(depth_image, dataset_size)
            cv2.imwrite(str(save_path + ("/%d_depth_pred.png" % frame_id)), depth_image)

            indices = np.argmax(pred_semantic.detach().cpu().numpy(), axis=1)
            semantic_image = converter[indices[i, ...], ...].astype('uint8')
            semantic_image = cv2.resize(semantic_image, dataset_size, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(save_path + ("/%d_semantic_pred.png" % frame_id)), semantic_image)

        lidar_panel = np.concatenate(list(lidar_bev.detach().cpu().numpy()[i][:2]), axis=1)
        lidar_panel = (lidar_panel * 255).astype(np.uint8)
        lidar_panel = np.stack([lidar_panel, lidar_panel, lidar_panel], axis=-1)
        lidar_panel = np.concatenate([lidar_panel, np.zeros_like(lidar_panel[:50])], axis=0)


        lidar_panel = Image.fromarray(lidar_panel)
        draw = ImageDraw.Draw(lidar_panel)
        lidar_panel = np.array(lidar_panel)
        cv2.imwrite(str(save_path + ("/%d_lidar_bev_detections_waypoints.png" % frame_id)), lidar_panel)

        bev = pred_bev[i].detach().cpu().numpy().argmax(axis=0).astype(np.float32)
        bev = np.stack([bev, bev, bev], axis=2) * 255.
        bev_image = bev.astype(np.uint8)

        if not expert_waypoints is None:
            bev_image = self.draw_waypoints(expert_waypoints[i:i+1, -1:], bev_image, color=(0, 0, 255))

        bev_image = self.draw_waypoints(deepcopy(pred_wp[i:i + 1, 2:]), bev_image, color=(255, 255, 255))
        bev_image = self.draw_waypoints(deepcopy(pred_wp[i:i + 1, :2]), bev_image, color=(255, 0, 0))

        bev_image = self.draw_target_point(target_point[i].detach().cpu().numpy(), bev_image)

        if (not (expert_waypoints is None)):
            aim = expert_waypoints[i:i + 1, :2].detach().cpu().numpy()[0].mean(axis=0)
            expert_angle = np.degrees(np.arctan2(aim[1], aim[0] + self.config.lidar_pos[0]))

            aim = pred_wp[i:i + 1, :2].detach().cpu().numpy()[0].mean(axis=0)
            ego_angle = np.degrees(np.arctan2(aim[1], aim[0] + self.config.lidar_pos[0]))
            angle_error = normalize_angle_degree(expert_angle - ego_angle)

            bev_image = Image.fromarray(bev_image)
            draw = ImageDraw.Draw(bev_image)
            draw.text((0, 0), "Angle error:        %.2f°" % (angle_error), font=font)
            bev_image = np.array(bev_image)

        cv2.imwrite(str(save_path + ("/%d_predicted_bev_semantic.png" % frame_id)), bev_image)

        rgb_image = rgb[i].permute(1, 2, 0).detach().cpu().numpy()[:, :, [2, 1, 0]]
        cv2.imwrite(str(save_path + ("/%d_rgb_input.png" % frame_id)), rgb_image)