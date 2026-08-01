"""
Credit: Tianwei Yin
Copied from LAV repo
"""

# from torch_scatter import scatter_mean, scatter_max
from torch import nn
import torch

#helper functions

def scatter_mean(src, index, dim=0, dim_size=None):
    """Drop-in replacement for torch_scatter.scatter_mean"""
    if index.dim() == 1:
        shape = [1] * src.dim()
        shape[dim] = -1
        index = index.view(*shape).expand_as(src)
    size = list(src.shape)
    size[dim] = dim_size if dim_size is not None else int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_reduce(dim, index, src, reduce="mean", include_self=False)

def scatter_max(src, index, dim=0, dim_size=None):
    """Drop-in replacement for torch_scatter.scatter_max — returns (values, argmax) like the original"""
    if index.dim() == 1:
        shape = [1] * src.dim()
        shape[dim] = -1
        index_expanded = index.view(*shape).expand_as(src)
    else:
        index_expanded = index

    size = list(src.shape)
    size[dim] = dim_size if dim_size is not None else int(index.max()) + 1

    out = torch.full(size, float("-inf"), dtype=src.dtype, device=src.device)
    out = out.scatter_reduce(dim, index_expanded, src, reduce="amax", include_self=False)

    # torch_scatter fills empty groups with 0, not -inf — match that convention
    out = torch.where(torch.isinf(out), torch.zeros_like(out), out)

    # compute argmax indices to match the original tuple return
    # for each src row, check whether it equals the (already-computed) max for its group
    matches = (src == out[index_expanded]) if index.dim() == 1 else (src == out.gather(dim, index_expanded))
    argmax = torch.full(size, -1, dtype=torch.long, device=src.device)
    src_positions = torch.arange(src.shape[dim], device=src.device)
    if index.dim() == 1:
        pos_expanded = src_positions.view(*([1]*dim), -1, *([1]*(src.dim()-dim-1))).expand_as(src)
    else:
        pos_expanded = src_positions  # adjust if you need multi-dim index support here

    argmax = argmax.scatter_reduce(dim, index_expanded, torch.where(matches, pos_expanded, torch.full_like(pos_expanded, -1)), reduce="amax", include_self=True)

    return out, argmax

class DynamicPointNet(nn.Module):
    def __init__(self, num_input=9, num_features=[32,32]):
        super().__init__()

        L = []
        for num_feature in num_features:
            L += [
                nn.Linear(num_input, num_feature),
                nn.BatchNorm1d(num_feature),
                nn.ReLU(inplace=True),
            ]

            num_input = num_feature

        self.net = nn.Sequential(*L)
        
    def forward(self, points, inverse_indices):
        """
        TODO: multiple layers
        """
        feat = self.net(points)
        feat_max = scatter_max(feat, inverse_indices, dim=0)[0]
        # feat_max = scatter_max(points, inverse_indices, dim=0)[0]
        return feat_max


class PointPillarNet(nn.Module):
    def __init__(self, num_input=9, num_features=[32,32], 
        min_x=-10, max_x=70,
        min_y=-40, max_y=40,
        pixels_per_meter=4):

        super().__init__()
        self.point_net = DynamicPointNet(num_input, num_features)

        self.nx = (max_x-min_x) * pixels_per_meter
        self.ny = (max_y-min_y) * pixels_per_meter
        self.min_x = min_x 
        self.min_y = min_y 
        self.max_x = max_x 
        self.max_y = max_y 
        self.pixels_per_meter = pixels_per_meter

    def decorate(self, points, unique_coords, inverse_indices):
        dtype = points.dtype
        x_centers = unique_coords[inverse_indices][:, 2:3].to(dtype) / self.pixels_per_meter + self.min_x 
        y_centers = unique_coords[inverse_indices][:, 1:2].to(dtype) / self.pixels_per_meter + self.min_y 

        xyz = points[:, :3]

        points_cluster = xyz - scatter_mean(xyz, inverse_indices, dim=0)[inverse_indices]

        points_xp = xyz[:, :1] - x_centers 
        points_yp = xyz[:, 1:2] - y_centers

        features = torch.cat([points, points_cluster, points_xp, points_yp], dim=-1)
        return features 

    def grid_locations(self, points):
        keep = (points[:, 0] >= self.min_x) & (points[:, 0] < self.max_x) & \
            (points[:, 1] >= self.min_y) & (points[:, 1] < self.max_y)
        points = points[keep, :]

        #Visualize
        #import open3d as o3d
        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(points[...,:3].detach().cpu().numpy())
        #o3d.visualization.draw_geometries([pcd])


        coords = (points[:, [0, 1]] - torch.tensor([self.min_x, self.min_y], 
            device=points.device)) * self.pixels_per_meter
        coords = coords.long()

        return points, coords 

    def pillar_generation(self, points, coords):
        unique_coords, inverse_indices = coords.unique(return_inverse=True, dim=0)
        decorated_points = self.decorate(points, unique_coords, inverse_indices)

        return decorated_points, unique_coords, inverse_indices

    def scatter_points(self, features, coords, batch_size ):
        canvas = torch.zeros(batch_size, features.shape[1], self.ny, self.nx, dtype=features.dtype, device=features.device)
        canvas[coords[:, 0], :, torch.clamp(self.ny-1-coords[:, 1],0,self.ny-1), torch.clamp(coords[:, 2],0,self.nx-1)] = features
        return canvas 

    def forward(self, lidar_list, num_points):
        batch_size = len(lidar_list)
        with torch.no_grad():
            coords = [] 
            filtered_points = [] 
            for batch_id, points in enumerate(lidar_list):
                points = points[:num_points[batch_id]]
                points, grid_yx= self.grid_locations(points)

                # batch indices 
                grid_byx = torch.nn.functional.pad(grid_yx, 
                    (1, 0), mode='constant', value=batch_id)

                coords.append(grid_byx)
                filtered_points.append(points)

            # batch_size, grid_y, grid_x 
            coords = torch.cat(coords, dim=0)
            filtered_points = torch.cat(filtered_points, dim=0)

            decorated_points, unique_coords, inverse_indices = self.pillar_generation(filtered_points, coords)

        features = self.point_net(decorated_points, inverse_indices)

        return self.scatter_points(features, unique_coords, batch_size)
