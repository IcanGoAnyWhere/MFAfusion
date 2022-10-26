import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import matplotlib
from torch.nn.functional import grid_sample
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import open3d as o3d
import torch.nn.functional as F

# from nuscenes.utils.geometry_utils import view_points
from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ....utils import common_utils, calibration_kitti

class Fusion_Conv_att(nn.Module):
    def __init__(self, IMG_chl, L_chl):

        super(Fusion_Conv_att, self).__init__()
        #multihead
        self.IMG_head = nn.ModuleList()
        self.headnum = 4
        for i in range(self.headnum):
            self.IMG_head.append(nn.Conv1d(IMG_chl, L_chl, 1))
        self.conv_L = nn.Conv1d(L_chl, L_chl, 1)
        self.ln = nn.LayerNorm(4096)

        self.conv_fusion = torch.nn.Conv1d(L_chl, L_chl, 1)
        # self.bn1 = torch.nn.BatchNorm1d(L_chl)


    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)
        IMG_key = []
        L_query = self.conv_L(point_features)
        score = []
        dk = L_query.size(1)
        pointnum = L_query.size(2)
        for i in range(self.headnum):

            IMG_key_cr = self.IMG_head[i](img_features)
            IMG_key.append(IMG_key_cr)
            score.append(torch.sum(torch.mul(IMG_key_cr, L_query),dim=1)/math.sqrt(dk))
        score = torch.cat(score)
        weightmap = torch.softmax(score, dim=0).unsqueeze(0)

        z = torch.tensor(np.zeros([1,dk,pointnum])).to(torch.float32).cuda()
        for i in range(self.headnum):
            z_cr = torch.mul(weightmap[:,i,:], IMG_key[i])
            z += z_cr
        fusion_features = self.conv_fusion(self.ln(z+L_query))
        # fusion_features = z


        return fusion_features,weightmap

class Fusion_Conv(nn.Module):
    def __init__(self, IMG_chl, L_chl):

        super(Fusion_Conv, self).__init__()
        self.conv1 = torch.nn.Conv1d(L_chl+IMG_chl, L_chl, 1)
        self.bn1 = torch.nn.BatchNorm1d(L_chl)


    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)

        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(
        torch.t(Id) * wd)
    return ans


def sample_points_with_roi(rois, points, sample_radius_with_roi, num_max_points_of_part=200000):
    """
    Args:
        rois: (M, 7 + C)
        points: (N, 3)
        sample_radius_with_roi:
        num_max_points_of_part:

    Returns:
        sampled_points: (N_out, 3)
    """
    if points.shape[0] < num_max_points_of_part:
        distance = (points[:, None, :] - rois[None, :, 0:3]).norm(dim=-1)
        min_dis, min_dis_roi_idx = distance.min(dim=-1)
        roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
        point_mask = min_dis < roi_max_dim + sample_radius_with_roi
    else:
        start_idx = 0
        point_mask_list = []
        while start_idx < points.shape[0]:
            distance = (points[start_idx:start_idx + num_max_points_of_part, None, :] - rois[None, :, 0:3]).norm(dim=-1)
            min_dis, min_dis_roi_idx = distance.min(dim=-1)
            roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
            cur_point_mask = min_dis < roi_max_dim + sample_radius_with_roi
            point_mask_list.append(cur_point_mask)
            start_idx += num_max_points_of_part
        point_mask = torch.cat(point_mask_list, dim=0)

    sampled_points = points[:1] if point_mask.sum() == 0 else points[point_mask, :]

    return sampled_points, point_mask


def sector_fps(points, num_sampled_points, num_sectors):
    """
    Args:
        points: (N, 3)
        num_sampled_points: int
        num_sectors: int

    Returns:
        sampled_points: (N_out, 3)
    """
    sector_size = np.pi * 2 / num_sectors
    point_angles = torch.atan2(points[:, 1], points[:, 0]) + np.pi
    sector_idx = (point_angles / sector_size).floor().clamp(min=0, max=num_sectors)
    xyz_points_list = []
    xyz_batch_cnt = []
    num_sampled_points_list = []
    for k in range(num_sectors):
        mask = (sector_idx == k)
        cur_num_points = mask.sum().item()
        if cur_num_points > 0:
            xyz_points_list.append(points[mask])
            xyz_batch_cnt.append(cur_num_points)
            ratio = cur_num_points / points.shape[0]
            num_sampled_points_list.append(
                min(cur_num_points, math.ceil(ratio * num_sampled_points))
            )

    if len(xyz_batch_cnt) == 0:
        xyz_points_list.append(points)
        xyz_batch_cnt.append(len(points))
        num_sampled_points_list.append(num_sampled_points)
        print(f'Warning: empty sector points detected in SectorFPS: points.shape={points.shape}')

    xyz = torch.cat(xyz_points_list, dim=0)
    xyz_batch_cnt = torch.tensor(xyz_batch_cnt, device=points.device).int()
    sampled_points_batch_cnt = torch.tensor(num_sampled_points_list, device=points.device).int()

    sampled_pt_idxs = pointnet2_stack_utils.stack_farthest_point_sample(
        xyz.contiguous(), xyz_batch_cnt, sampled_points_batch_cnt
    ).long()
    sampled_pt_idxs = sampled_pt_idxs[0:num_sampled_points].unsqueeze(0)

    sampled_points = xyz[sampled_pt_idxs]

    return sampled_points,sampled_pt_idxs


class VPSAwithAtt(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        SA_cfg = self.model_cfg.SA_LAYER

        self.SA_layers = nn.ModuleList()
        self.Fusion_Conv = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR

            if SA_cfg[src_name].get('INPUT_CHANNELS', None) is None:
                input_channels = SA_cfg[src_name].MLPS[0][0] \
                    if isinstance(SA_cfg[src_name].MLPS[0], list) else SA_cfg[src_name].MLPS[0]
            else:
                input_channels = SA_cfg[src_name]['INPUT_CHANNELS']

            cur_layer, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
                input_channels=input_channels, config=SA_cfg[src_name]
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)

            c_in += cur_num_c_out

        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            self.SA_rawpoints, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
                input_channels=num_rawpoint_features - 3, config=SA_cfg['raw_points']
            )

            c_in += cur_num_c_out

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in

        for i in range(len(model_cfg.IMG_CHANNELS) - 1):
            self.Fusion_Conv.append(Fusion_Conv_att(model_cfg.IMG_CHANNELS[i + 1],
                                                model_cfg.POINT_CHANNELS[i]))

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        """
        Args:
            keypoints: (N1 + N2 + ..., 4)
            bev_features: (B, C, H, W)
            batch_size:
            bev_stride:

        Returns:
            point_bev_features: (N1 + N2 + ..., C)
        """
        x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]

        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            bs_mask = (keypoints[:, 0] == k)

            cur_x_idxs = x_idxs[bs_mask]
            cur_y_idxs = y_idxs[bs_mask]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features)

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (N1 + N2 + ..., C)
        return point_bev_features

    def sectorized_proposal_centric_sampling(self, roi_boxes, points):
        """
        Args:
            roi_boxes: (M, 7 + C)
            points: (N, 3)

        Returns:
            sampled_points: (N_out, 3)
        """

        sampled_points, _ = sample_points_with_roi(
            rois=roi_boxes, points=points,
            sample_radius_with_roi=self.model_cfg.SPC_SAMPLING.SAMPLE_RADIUS_WITH_ROI,
            num_max_points_of_part=self.model_cfg.SPC_SAMPLING.get('NUM_POINTS_OF_EACH_SAMPLE_PART', 200000)
        )
        sampled_points, sampled_pt_idxs = sector_fps(
            points=sampled_points, num_sampled_points=self.model_cfg.NUM_KEYPOINTS,
            num_sectors=self.model_cfg.SPC_SAMPLING.NUM_SECTORS
        )
        return sampled_points, sampled_pt_idxs

    def vector_angle(self,x, y):
        lx = np.sqrt(x.dot(x))
        ly = (np.sum(y ** 2, axis=1)) ** 0.5
        cos_angle = np.sum(x * y, axis=1) / (lx * ly)
        angle = np.arccos(cos_angle)  # arccos计算出的角度是弧度制
        angle2 = np.degrees(angle)  # 阈值设置中用的是角度制，因此这里需要将弧度转换为角度
        return angle2

    def get_curvature_sampled_points(self, batch_dict):
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:4]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask]
            knn_num = 300  # 自定义参数值(邻域点数)
            angle_thre = 30  # 自定义参数值(角度制度)
            N = 5  # 自定义参数值(每N个点采样一次)
            C = 10  # 自定义参数值(采样均匀性>N)

            src_points = sampled_points.cpu().contiguous()
            point = np.asarray(src_points)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point)
            point_size = src_points.shape[0]
            tree = o3d.geometry.KDTreeFlann(pcd)  # 建立KD树索引
            o3d.geometry.PointCloud.estimate_normals(
                pcd, search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_num))  # 计算法向量
            normal = np.asarray(pcd.normals)
            normal_angle = np.zeros(point_size)

            for i in range(point_size):
                [_, idx, dis] = tree.search_knn_vector_3d(point[i], knn_num + 1)
                current_normal = normal[i]
                knn_normal = normal[idx[1:]]
                normal_angle[i] = np.mean(self.vector_angle(current_normal, knn_normal))

            point_high = point[np.where(normal_angle >= angle_thre)]  # 位于特征明显区域的点

            pcd_high = o3d.geometry.PointCloud()
            pcd_high.points = o3d.utility.Vector3dVector(point_high)

            finl_point_size = np.asarray(pcd_high.points).shape[0]
            print("原始点云点的个数为：", point_size)
            print("下采样后点的个数为：", finl_point_size)
            o3d.visualization.draw_geometries([pcd_high], window_name="曲率下采样",
                                              width=1024, height=768,
                                              left=50, top=50,
                                              mesh_show_back_face=False)
            keypoints = torch.tensor(point_high).cuda()
            keypoints_list.append(keypoints)

        keypoints = torch.cat(keypoints_list, dim=0).unsqueeze(0)  # (B, M, 3) or (N1 + N2 + ..., 4)
        if len(keypoints.shape) == 3:
            batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1,keypoints.shape[1]).view(-1, 1)
            keypoints = torch.cat((batch_idx.float(), keypoints.view(-1, 3)), dim=1)

        return keypoints

    def farthest_point_sample(self, xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        device = xyz.device
        batchsize, ndataset, dimension = xyz.shape
        # to方法Tensors和Modules可用于容易地将对象移动到不同的设备（代替以前的cpu()或cuda()方法）
        # 如果他们已经在目标设备上则不会执行复制操作
        centroids = torch.zeros(batchsize, npoint, dtype=torch.long).to(device)
        distance = torch.ones(batchsize, ndataset).to(device) * 1e10
        # randint(low, high, size, dtype)
        # torch.randint(3, 5, (3,))->tensor([4, 3, 4])
        farthest = torch.randint(0, ndataset, (batchsize,), dtype=torch.long).to(device)
        # batch_indices=[0,1,...,batchsize-1]
        batch_indices = torch.arange(batchsize, dtype=torch.long).to(device)
        for i in range(npoint):
            # 更新第i个最远点
            centroids[:, i] = farthest
            # 取出这个最远点的xyz坐标
            centroid = xyz[batch_indices, farthest, :].view(batchsize, 1, xyz.shape[2])
            # 计算点集中的所有点到这个最远点的欧式距离
            # 等价于torch.sum((xyz - centroid) ** 2, 2)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
            mask = dist < distance
            distance[mask] = dist[mask]
            # 从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
            # 取出每一行的最大值构成列向量，等价于torch.max(x,2)
            farthest = torch.max(distance, -1)[1]
        return centroids

    def get_sampled_points(self, batch_dict):
        """
        Args:
            batch_dict:

        Returns:
            keypoints: (N1 + N2 + ..., 4), where 4 indicates [bs_idx, x, y, z]
        """
        batch_size = batch_dict['batch_size']
        num_keypoints = self.model_cfg['NUM_KEYPOINTS']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:4]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        pts_lidar2img_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)

            if self.model_cfg.SAMPLE_METHOD == 'FPS':

                cur_pt_idxs = pointnet2_stack_utils.farthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                ).long()

                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    times = int(self.model_cfg.NUM_KEYPOINTS / sampled_points.shape[1]) + 1
                    non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
                    cur_pt_idxs[0] = non_empty.repeat(times)[:self.model_cfg.NUM_KEYPOINTS]

                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

                pts_lidar = sampled_points[0][cur_pt_idxs[0]].squeeze(0).cpu().numpy()
                # filter ground
                # pts_lidar_mask = np.where(pts_lidar[:,2]>-1.5)
                # pts_lidar =pts_lidar[pts_lidar_mask]

                pts_lidar = np.hstack((pts_lidar, np.ones((pts_lidar.shape[0], 1), dtype=np.float32)))
                P2 = batch_dict['trans_cam_to_img'].squeeze(0).cpu().numpy()  # 内参矩阵
                V2R = batch_dict['trans_lidar_to_cam'].squeeze(0).cpu().numpy()  # 外参矩阵
                trans_matrix = np.matmul(P2, V2R)  # 转换矩阵

                if batch_size != 1:
                    trans = trans_matrix[bs_idx, :, :]
                else:
                    trans = trans_matrix
                pts_lidar2img = np.matmul(trans, pts_lidar.T)
                pts_lidar2img = np.transpose(pts_lidar2img)
                pts_lidar2img = pts_lidar2img / pts_lidar2img[:, [2]]
                pts_lidar2img = torch.tensor(pts_lidar2img).cuda().unsqueeze(0)


            elif self.model_cfg.SAMPLE_METHOD == 'PFPS':
                if 'kitti' in self.model_cfg.get('DATASET', None):
                    pts_lidar = sampled_points.squeeze(0).cpu().numpy()
                    # filter ground
                    # pts_lidar_mask = np.where(pts_lidar[:,2]>-1.4)
                    # pts_lidar =pts_lidar[pts_lidar_mask]
                    # sampled_points = sampled_points[:,pts_lidar_mask,:].squeeze(0)

                    pts_lidar = np.hstack((pts_lidar, np.ones((pts_lidar.shape[0], 1), dtype=np.float32)))
                    P2 = batch_dict['trans_cam_to_img'].squeeze(0).cpu().numpy()  # 内参矩阵
                    V2R = batch_dict['trans_lidar_to_cam'].squeeze(0).cpu().numpy()  # 外参矩阵
                    trans_matrix = np.matmul(P2, V2R)  # 转换矩阵

                    if batch_size != 1:
                        trans = trans_matrix[bs_idx, :, :]
                    else:
                        trans = trans_matrix
                    pts_lidar2img = np.matmul(trans, pts_lidar.T)
                    pts_lidar2img = np.transpose(pts_lidar2img)
                    pts_lidar2img = pts_lidar2img / pts_lidar2img[:, [2]]
                    pts_lidar2img = torch.tensor(pts_lidar2img).cuda().unsqueeze(0)

                    pts_fusion = torch.dstack((pts_lidar2img, 250*sampled_points))


                alpha = 0.3
                cur_pt_idxs_PFPS = pointnet2_stack_utils.farthest_point_sample(
                    pts_fusion[:, :, 0:3].contiguous(), int(num_keypoints*alpha)
                ).long().cpu().numpy()
                cur_pt_idxs_FPS = pointnet2_stack_utils.farthest_point_sample(
                    pts_fusion[:, :, 3:6].contiguous(), int(num_keypoints*(1-alpha))
                ).long().cpu().numpy()
                sampled_num = np.size(cur_pt_idxs_PFPS,1)+np.size(cur_pt_idxs_FPS,1)

                cur_pt_idxs = np.zeros((1,num_keypoints))

                cur_pt_idxs[0][:sampled_num] = np.hstack([cur_pt_idxs_PFPS[0],cur_pt_idxs_FPS[0]])
                # cur_pt_idxs = cur_pt_idxs_FPS

                if np.size(cur_pt_idxs,1) < self.model_cfg.NUM_KEYPOINTS:
                    times = int(self.model_cfg.NUM_KEYPOINTS / np.size(cur_pt_idxs,1)) + 1
                    non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
                    cur_pt_idxs[0] = non_empty.repeat(times)[:self.model_cfg.NUM_KEYPOINTS]
                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
                pts_lidar2img = pts_lidar2img[0][cur_pt_idxs[0]].unsqueeze(dim=0)


                # visualize_point = sampled_points[0][cur_pt_idxs[0]].cpu().numpy()
                # # visualize_point = sampled_points.squeeze(0).cpu().numpy()
                # pcd_point = o3d.geometry.PointCloud()
                # pcd_point.points = o3d.utility.Vector3dVector(visualize_point)
                # pcd_point.paint_uniform_color([0, 0, 1])
                # o3d.visualization.draw_geometries([pcd_point], window_name="keypoints",
                #                                   width=1024, height=768,
                #                                   left=50, top=50,
                #                                   mesh_show_back_face=False)

            elif self.model_cfg.SAMPLE_METHOD == 'SPC':
                cur_keypoints, sampled_pt_idxs = self.sectorized_proposal_centric_sampling(
                    roi_boxes=batch_dict['rois'][bs_idx], points=sampled_points[0]
                )
                # cur_keypoints = cur_keypoints[0:self.model_cfg.NUM_KEYPOINTS,:]
                if cur_keypoints.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    times = int(self.model_cfg.NUM_KEYPOINTS / cur_keypoints.shape[1]) + 1
                    non_empty = sampled_pt_idxs[0, :cur_keypoints.shape[1]]
                    sampled_pt_mask = non_empty.repeat(times)[:self.model_cfg.NUM_KEYPOINTS]
                    keypoints =sampled_points[0][sampled_pt_mask].unsqueeze(0)
                else:
                    keypoints = cur_keypoints

                if 'kitti' in self.model_cfg.get('DATASET', None):
                    pts_lidar = keypoints.squeeze(0).cpu().numpy()
                    # filter ground
                    # pts_lidar_mask = np.where(pts_lidar[:,2]>-1.4)
                    # pts_lidar =pts_lidar[pts_lidar_mask]
                    # sampled_points = sampled_points[:,pts_lidar_mask,:].squeeze(0)

                    pts_lidar = np.hstack((pts_lidar, np.ones((pts_lidar.shape[0], 1), dtype=np.float32)))
                    P2 = batch_dict['trans_cam_to_img'].squeeze(0).cpu().numpy()  # 内参矩阵
                    V2R = batch_dict['trans_lidar_to_cam'].squeeze(0).cpu().numpy()  # 外参矩阵
                    trans_matrix = np.matmul(P2, V2R)  # 转换矩阵

                    if batch_size != 1:
                        trans = trans_matrix[bs_idx, :, :]
                    else:
                        trans = trans_matrix
                    pts_lidar2img = np.matmul(trans, pts_lidar.T)
                    pts_lidar2img = np.transpose(pts_lidar2img)
                    pts_lidar2img = pts_lidar2img / pts_lidar2img[:, [2]]
                    pts_lidar2img = torch.tensor(pts_lidar2img).cuda().unsqueeze(0)


            elif self.model_cfg.SAMPLE_METHOD == 'CBS':
                if 'kitti' in self.model_cfg.get('DATASET', None):

                    # size_range = [1242.0, 375.0]

                    pts_lidar = sampled_points.squeeze(0).cpu().numpy()
                    # filter ground
                    # pts_lidar_mask = np.where(pts_lidar[:,2]>-1.5)
                    # pts_lidar =pts_lidar[pts_lidar_mask]

                    pts_lidar = np.hstack((pts_lidar, np.ones((pts_lidar.shape[0], 1), dtype=np.float32)))
                    P2 = batch_dict['trans_cam_to_img'].squeeze(0).cpu().numpy()  # 内参矩阵
                    V2R = batch_dict['trans_lidar_to_cam'].squeeze(0).cpu().numpy()  # 外参矩阵
                    trans_matrix = np.matmul(P2, V2R)  # 转换矩阵

                    if batch_size != 1:
                        trans = trans_matrix[bs_idx, :, :]
                    else:
                        trans = trans_matrix
                    pts_lidar2img = np.matmul(trans, pts_lidar.T)
                    pts_lidar2img = np.transpose(pts_lidar2img)
                    pts_lidar2img = pts_lidar2img / pts_lidar2img[:, [2]]
                    pts_lidar2img = torch.tensor(pts_lidar2img).cuda().unsqueeze(0)
                bs_mask = (batch_indices == bs_idx)
                sampled_points = src_points[bs_mask]
                knn_num = 300  # 自定义参数值(邻域点数)
                angle_thre = 30  # 自定义参数值(角度制度)
                N = 5  # 自定义参数值(每N个点采样一次)
                C = 10  # 自定义参数值(采样均匀性>N)

                src_points = sampled_points.cpu().contiguous()
                point = np.asarray(src_points)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point)
                point_size = src_points.shape[0]
                tree = o3d.geometry.KDTreeFlann(pcd)  # 建立KD树索引
                o3d.geometry.PointCloud.estimate_normals(
                    pcd, search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_num))  # 计算法向量
                normal = np.asarray(pcd.normals)
                normal_angle = np.zeros(point_size)

                for i in range(point_size):
                    [_, idx, dis] = tree.search_knn_vector_3d(point[i], knn_num + 1)
                    current_normal = normal[i]
                    knn_normal = normal[idx[1:]]
                    normal_angle[i] = np.mean(self.vector_angle(current_normal, knn_normal))

                point_high = point[np.where(normal_angle >= angle_thre)]  # 位于特征明显区域的点

                # pcd_high = o3d.geometry.PointCloud()
                # pcd_high.points = o3d.utility.Vector3dVector(point_high)
                #
                # finl_point_size = np.asarray(pcd_high.points).shape[0]
                # print("原始点云点的个数为：", point_size)
                # print("下采样后点的个数为：", finl_point_size)
                # o3d.visualization.draw_geometries([pcd_high], window_name="曲率下采样",
                #                                   width=1024, height=768,
                #                                   left=50, top=50,
                #                                   mesh_show_back_face=False)

                sampled_points = torch.tensor(point_high).cuda().unsqueeze(dim=0)
                cur_pt_idxs = pointnet2_stack_utils.farthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                ).long()

                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    times = int(self.model_cfg.NUM_KEYPOINTS / sampled_points.shape[1]) + 1
                    non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
                    cur_pt_idxs[0] = non_empty.repeat(times)[:self.model_cfg.NUM_KEYPOINTS]
                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
                pts_lidar2img = pts_lidar2img[0][cur_pt_idxs[0]].unsqueeze(dim=0)

            else:
                raise NotImplementedError

            keypoints_list.append(keypoints)
            pts_lidar2img_list.append(pts_lidar2img)

        keypoints_lidar2img = torch.cat(pts_lidar2img_list, dim=0)
        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3) or (N1 + N2 + ..., 4)
        if len(keypoints.shape) == 3:
            batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1,keypoints.shape[1]).view(-1, 1)
            keypoints = torch.cat((batch_idx.float(), keypoints.view(-1, 3)), dim=1)
            keypoints_lidar2img = torch.cat((batch_idx.float(), keypoints_lidar2img.view(-1, 3)), dim=1)


        return keypoints,keypoints_lidar2img

    def get_sample_feature(self, uv, featuremap, size_range):

        img_uv = copy.deepcopy(uv)
        img_uv[:, :, 0] = uv[:, :, 0] / (size_range[3] - 1.0) * 2.0 - 1.0
        img_uv[:, :, 1] = uv[:, :, 1] / (size_range[2] - 1.0) * 2.0 - 1.0
        img_uv = img_uv.unsqueeze(1).cuda().type(torch.float32)
        img_sample_feature = grid_sample(featuremap, img_uv).squeeze(2)

        return img_sample_feature

    @staticmethod
    def aggregate_keypoint_features_from_one_source(
            batch_size, aggregate_func, xyz, xyz_features, xyz_bs_idxs, new_xyz, new_xyz_batch_cnt,
            filter_neighbors_with_roi=False, radius_of_neighbor=None, num_max_points_of_part=200000, rois=None
    ):
        """

        Args:
            aggregate_func:
            xyz: (N, 3)
            xyz_features: (N, C)
            xyz_bs_idxs: (N)
            new_xyz: (M, 3)
            new_xyz_batch_cnt: (batch_size), [N1, N2, ...]

            filter_neighbors_with_roi: True/False
            radius_of_neighbor: float
            num_max_points_of_part: int
            rois: (batch_size, num_rois, 7 + C)
        Returns:

        """
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        if filter_neighbors_with_roi:
            point_features = torch.cat((xyz, xyz_features), dim=-1) if xyz_features is not None else xyz
            point_features_list = []
            for bs_idx in range(batch_size):
                bs_mask = (xyz_bs_idxs == bs_idx)
                _, valid_mask = sample_points_with_roi(
                    rois=rois[bs_idx], points=xyz[bs_mask],
                    sample_radius_with_roi=radius_of_neighbor, num_max_points_of_part=num_max_points_of_part,
                )
                point_features_list.append(point_features[bs_mask][valid_mask])
                xyz_batch_cnt[bs_idx] = valid_mask.sum()

            valid_point_features = torch.cat(point_features_list, dim=0)
            xyz = valid_point_features[:, 0:3]
            xyz_features = valid_point_features[:, 3:] if xyz_features is not None else None
        else:
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (xyz_bs_idxs == bs_idx).sum()

        pooled_points, pooled_features = aggregate_func(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=xyz_features.contiguous(),
        )
        return pooled_features



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        batch_size = batch_dict['batch_size']

        keypoints, keypoints_lidar2img = self.get_sampled_points(batch_dict)
        # print(keypoints.shape)
        # keypoints = self.get_curvature_sampled_points(batch_dict)[0:4096,:]

        point_features_list = []
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            point_bev_features = self.interpolate_from_bev_features(
                keypoints, batch_dict['spatial_features'], batch_dict['batch_size'],
                bev_stride=batch_dict['spatial_features_stride']
            )
            point_features_list.append(point_bev_features)



        new_xyz = keypoints[:, 1:4].contiguous()
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        for k in range(batch_size):
            new_xyz_batch_cnt[k] = (keypoints[:, 0] == k).sum()

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            raw_points = batch_dict['points']

            pooled_features = self.aggregate_keypoint_features_from_one_source(
                batch_size=batch_size, aggregate_func=self.SA_rawpoints,
                xyz=raw_points[:, 1:4],
                xyz_features=raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None,
                xyz_bs_idxs=raw_points[:, 0],
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt,
                filter_neighbors_with_roi=self.model_cfg.SA_LAYER['raw_points'].get('FILTER_NEIGHBOR_WITH_ROI', False),
                radius_of_neighbor=self.model_cfg.SA_LAYER['raw_points'].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None),
                rois=batch_dict.get('rois', None)
            )
            point_features_list.append(pooled_features)

        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
            cur_features = batch_dict['multi_scale_3d_features'][src_name].features.contiguous()

            xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4], downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range
            )

            pooled_features = self.aggregate_keypoint_features_from_one_source(
                batch_size=batch_size, aggregate_func=self.SA_layers[k],
                xyz=xyz.contiguous(), xyz_features=cur_features, xyz_bs_idxs=cur_coords[:, 0],
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt,
                filter_neighbors_with_roi=self.model_cfg.SA_LAYER[src_name].get('FILTER_NEIGHBOR_WITH_ROI', False),
                radius_of_neighbor=self.model_cfg.SA_LAYER[src_name].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None),
                rois=batch_dict.get('rois', None)
            )

            point_features_list.append(pooled_features)

        num_keypoints = self.model_cfg['NUM_KEYPOINTS']


        # if 'kitti' in self.model_cfg.get('DATASET', None):
        #
        #     size_range = [1242.0, 375.0]
        #     pts_lidar = keypoints[:, 1:4].cpu().numpy()
        #     pts_lidar = np.hstack((pts_lidar, np.ones((pts_lidar.shape[0], 1), dtype=np.float32)))
        #     P2 = batch_dict['trans_cam_to_img'].squeeze(0).cpu().numpy()  # 内参矩阵
        #     V2R = batch_dict['trans_lidar_to_cam'].squeeze(0).cpu().numpy()  # 外参矩阵
        #     trans_matrix = np.matmul(P2, V2R)  # 转换矩阵
        #
        #     pts_lidar2img = np.matmul(trans_matrix, pts_lidar.T)
        #     pts_lidar2img = np.transpose(pts_lidar2img)
        #     keypoints_lidar2img = pts_lidar2img / pts_lidar2img[:, [2]]

        # elif 'nuscenes' in self.model_cfg.get('DATASET', None):
        #
        #     size_range = [1600.0, 900.0]
        #     pts_lidar = keypoints[:, 1:4].cpu().numpy().T
        #
        #     r_matrix = batch_dict['r_matrix'].squeeze(0).cpu().numpy()
        #     t_matrix = batch_dict['t_matrix'].squeeze(0).cpu().numpy()
        #     camera_intrinsic = batch_dict['camera_intrinsic'].squeeze(0).cpu().numpy()
        #
        #     pts_lidar = np.matmul(r_matrix[0], pts_lidar) + t_matrix[0]
        #     pts_lidar = np.matmul(r_matrix[1], pts_lidar) + t_matrix[1]
        #     pts_lidar = np.matmul(r_matrix[2], pts_lidar + t_matrix[2])
        #     pts_lidar = np.matmul(r_matrix[3], pts_lidar + t_matrix[3])
        #
        #     pts_lidar2img = view_points(pts_lidar, np.array(camera_intrinsic), normalize=True)
        #     pts_lidar2img = pts_lidar2img.T

        pts_lidar2img_list = keypoints_lidar2img[:, 1:4].cpu().numpy()
        xylist = []
        weightmap_list = []
        featuremap_list = []

        VP_features_cat_1 = []
        VP_features_cat_2 = []
        VP_features_cat_3 = []
        VP_features_cat_4 = []

        for bs in range(batch_size):
            multi_img_features = batch_dict['multi_img_features']
            VP_features_list = []
            for i, img_src_name in enumerate(multi_img_features):
                pts_lidar2img_bs = pts_lidar2img_list[num_keypoints*bs:num_keypoints*(bs+1),:]
                xy_1 = pts_lidar2img_bs[:, [0, 1]] / 2 ** (i + 2)
                xy = torch.tensor(xy_1).unsqueeze(0)
                featuremap = multi_img_features[img_src_name][bs,:,:,:].unsqueeze(0)
                size_range = featuremap.size()
                img_sample_feature = self.get_sample_feature(xy, featuremap, size_range)

                featuremap_list.append(featuremap)
                lidar_feature = point_features_list[i + 1].permute(1, 0).unsqueeze(0)
                lidar_feature = lidar_feature[:,:,num_keypoints*bs:num_keypoints*(bs+1)]
                VP_features, weightmap = self.Fusion_Conv[i](lidar_feature, img_sample_feature)
                # VP_features = self.Fusion_Conv[i](lidar_feature, img_sample_feature)
                VP_features_list.append(VP_features)

                if self.model_cfg.DEBUG:
                    xylist.append(copy.deepcopy(xy))
                    weightmap_list.append(weightmap)

            VP_features_cat_1.append(VP_features_list[0])
            VP_features_cat_2.append(VP_features_list[1])
            VP_features_cat_3.append(VP_features_list[2])
            VP_features_cat_4.append(VP_features_list[3])

        point_features_list[1] = torch.cat(VP_features_cat_1, dim=-1).squeeze(0).permute(1, 0)
        point_features_list[2] = torch.cat(VP_features_cat_2, dim=-1).squeeze(0).permute(1, 0)
        point_features_list[3] = torch.cat(VP_features_cat_3, dim=-1).squeeze(0).permute(1, 0)
        point_features_list[4] = torch.cat(VP_features_cat_4, dim=-1).squeeze(0).permute(1, 0)

        point_features = torch.cat(point_features_list, dim=-1)
        batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))

        batch_dict['point_features'] = point_features  # (BxN, C)
        batch_dict['point_coords'] = keypoints # (BxN, 4)
        # print(featuremap_list[0][0,0,0,0])

        if self.model_cfg.DEBUG:
            bs = 0
            idx = 2
            # 获取图片
            imgbatch = tv.utils.make_grid(batch_dict['images'][bs,:,:,:]).cpu().numpy()
            # img_cov1 = tv.utils.make_grid(featuremap_list[idx]).cpu()[0:3, :, :]
            # imgbatch = img_cov1.detach().numpy()

            # u = xylist[idx][:, :, 0]
            # v = xylist[idx][:, :, 1]
            u = pts_lidar2img_list[num_keypoints*bs:num_keypoints*(bs+1), 0]
            v = pts_lidar2img_list[num_keypoints*bs:num_keypoints*(bs+1), 1]

            plt.imshow(np.transpose(imgbatch, (1, 2, 0)))

            colormap = weightmap_list[idx].squeeze(0).cpu().detach().numpy()
            colormap_index = np.argmax(colormap, axis=0)
            plt.scatter(u, v, s=5, c=colormap_index, cmap='cool')
            # plt.scatter(u, v, s=5, c='white')
            plt.show()


            # pointshow = batch_dict['points'][:, 1:4].cpu().numpy()
            # point_cloud = open3d.geometry.PointCloud()
            # point_cloud.points = open3d.utility.Vector3dVector(pointshow)
            # open3d.visualization.draw_geometries([point_cloud])


        return batch_dict
