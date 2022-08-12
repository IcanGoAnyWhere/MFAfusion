import open3d as o3d
import numpy as np
olderr = np.seterr(all='ignore')


def vector_angle(x, y):
    lx = np.sqrt(x.dot(x))
    ly = (np.sum(y ** 2, axis=1)) ** 0.5
    cos_angle = np.sum(x * y, axis=1) / (lx * ly)
    angle = np.arccos(cos_angle)  # arccos计算出的角度是弧度制
    angle2 = np.degrees(angle)    # 阈值设置中用的是角度制，因此这里需要将弧度转换为角度
    return angle2


knn_num = 10     # 自定义参数值(邻域点数)
angle_thre = 30  # 自定义参数值(角度制度)
N = 5            # 自定义参数值(每N个点采样一次)
C = 10           # 自定义参数值(采样均匀性>N)

pcd = o3d.io.read_point_cloud("data//bunny.pcd")
point = np.asarray(pcd.points)
point_size = point.shape[0]
tree = o3d.geometry.KDTreeFlann(pcd)  # 建立KD树索引
o3d.geometry.PointCloud.estimate_normals(
    pcd, search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_num))  # 计算法向量
normal = np.asarray(pcd.normals)
normal_angle = np.zeros(point_size)
for i in range(point_size):
    [_, idx, dis] = tree.search_knn_vector_3d(point[i], knn_num + 1)
    current_normal = normal[i]
    knn_normal = normal[idx[1:]]
    normal_angle[i] = np.mean(vector_angle(current_normal, knn_normal))

point_high = point[np.where(normal_angle >= angle_thre)]  # 位于特征明显区域的点
point_low = point[np.where(normal_angle < angle_thre)]    # 位于特征不明显区域的点
pcd_high = o3d.geometry.PointCloud()
pcd_high.points = o3d.utility.Vector3dVector(point_high)
pcd_low = o3d.geometry.PointCloud()
pcd_low.points = o3d.utility.Vector3dVector(point_low)
pcd_high_down = o3d.geometry.PointCloud.uniform_down_sample(pcd_high, N)  # 均匀采样特征明显区域
pcd_low_down = o3d.geometry.PointCloud.uniform_down_sample(pcd_low, C)    # 均匀采样特征不明显区域
pcd_finl = o3d.geometry.PointCloud()
pcd_finl.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd_high_down.points),
                                                             np.asarray(pcd_low_down.points))))  # 拼接点云
finl_point_size = np.asarray(pcd_finl.points).shape[0]
print("原始点云点的个数为：", point_size)
print("下采样后点的个数为：", finl_point_size)
o3d.visualization.draw_geometries([pcd_finl], window_name="曲率下采样",
                                  width=1024, height=768,
                                  left=50, top=50,
                                  mesh_show_back_face=False)





