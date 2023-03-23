import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch

from pcdet.datasets.kitti.kitti_dataset import KittiDataset
import torchvision.models as model
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D


def entropy_cal(input):
    input_soft = torch.softmax(input, dim=1)
    predict_map = torch.log2(input_soft)
    entropy_cha = predict_map * input_soft
    entropy_cha = torch.sum(entropy_cha,dim=1)

    return entropy_cha

# xx = torch.tensor(np.random.normal(60, 5, 100))
# yy = torch.tensor(np.random.normal(20, 5, 100))
xx = torch.tensor([5., 5., 5., 5., 5., 10., 5., 5., 5., 5., 5., 5., 10., 5.]) * torch.tensor(10)
yy = torch.tensor([3., 3., 3., 3., 3., 8., 3., 3., 3., 3., 3., 3., 8., 3.]) * torch.tensor(10)

# # scale_xx = xx/torch.sum(xx)
# maenxx = torch.abs(xx / torch.mean(xx))
# input_softxx = torch.softmax(maenxx, dim=0)
# predict_mapxx = torch.log2(input_softxx)
# entropy_chaxx = -predict_mapxx * input_softxx
# entropy_chaxx = torch.sum(entropy_chaxx,dim=0) * torch.tensor(10)
# varxx = torch.var(xx)
#
# # scale_yy = yy/torch.sum(yy)
# maenyy = torch.abs(yy / torch.mean(yy))
# input_softyy = torch.softmax(maenyy, dim=0)
# predict_mapyy = torch.log2(input_softyy)
# entropy_chayy = -predict_mapyy * input_softyy
# entropy_chayy = torch.sum(entropy_chayy,dim=0) * torch.tensor(10)
# varyy = torch.var(yy)



model = model.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-2])
model.eval().cuda()

img1 = cv.imread('/media/xrd/data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/aachen/aachen_000021_000019_leftImg8bit.png')
img2 = cv.imread('/media/xrd/data/cityscapes/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy/train/aachen/aachen_000021_000019_leftImg8bit_foggy_beta_0.02.png')

noise = KittiDataset.get_noise(self=None, img=img1, value=1000)
rain = KittiDataset.rain_blur(self=None, noise=noise, length=50, angle=-30, w=3)
image_cv = KittiDataset.alpha_rain(self=None, rain=rain, img=img1, beta=0.8)  # 方法一，透明度赋值
image_cv = cv.GaussianBlur(image_cv, (33, 33), 3)
image_cv = cv.cvtColor(image_cv, cv.COLOR_BGR2RGB)
img3 = np.float32(image_cv)
img3 /= 255.0

#   feature entropy
img1_cv = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
# img1_cv = cv.resize(img1, [224,224])
img1_cv = np.float32(img1_cv)/255.0
input1 = torch.from_numpy(img1_cv).permute(2, 0, 1).unsqueeze(0).cuda()

img2_cv = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
# img1_cv = cv.resize(img1, [224,224])
img2_cv = np.float32(img2_cv)/255.0
input2 = torch.from_numpy(img2_cv).permute(2, 0, 1).unsqueeze(0).cuda()

with torch.no_grad():
    fea_img1 = model(input1)
    fea_img2 = model(input2)

entropy1 = entropy_cal(fea_img1)
heatmap1 = entropy1.squeeze(0).cpu().numpy()
heatmap1 -= heatmap1.min()
heatmap1 /= heatmap1.max()
heatmap1 *= heatmap1*255

entropy2 = entropy_cal(fea_img2)
heatmap2 = entropy2.squeeze(0).cpu().numpy()
heatmap2 -= heatmap2.min()
heatmap2 /= heatmap2.max()
heatmap2 *= heatmap2*255

hist1 = cv.calcHist(heatmap1,[0],None,[256],[0,256])
p1 = hist1/np.sum(hist1)
p1 = p1[p1 > 0]
fea_en1 = -np.sum(p1 * np.log2(p1))

hist2 = cv.calcHist(heatmap2,[0],None,[256],[0,256])
p2 = hist2/np.sum(hist2)
p2 = p2[p2 > 0]
fea_en2 = -np.sum(p2 * np.log2(p2))
print('feature_en1:',fea_en1, 'feature_en1:',fea_en2)

# 3D surface
x = np.arange(0, heatmap1.shape[1], 1)
y = np.arange(0, heatmap1.shape[0], 1)
fig = plt.figure(1)
X,Y = np.meshgrid(x, y)
sub = fig.add_subplot(211, projection='3d')
sub.plot_surface(X, Y, heatmap1,
                rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))

sub = fig.add_subplot(212, projection='3d')
sub.plot_surface(X, Y, heatmap2,
                rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))


# 3D bar
fig = plt.figure(2)
X, Y, bar1 = X.ravel(), Y.ravel(), heatmap1.ravel()
X, Y, bar2 = X.ravel(), Y.ravel(), heatmap2.ravel()
dx = np.ones_like(X)
dy = dx.copy()
bottom = np.zeros_like(X)
sub = fig.add_subplot(211, projection='3d')
sub.bar3d(X, Y, bottom, dx, dy, bar1, shade=True)
sub = fig.add_subplot(212, projection='3d')
sub.bar3d(X, Y, bottom, dx, dy, bar2, shade=True)




#   image entropy

r1, g1, b1 = cv.split(img1)
rhist1 = cv.calcHist(r1,[0],None,[256],[0,256])
ghist1 = cv.calcHist(g1,[0],None,[256],[0,256])
bhist1 = cv.calcHist(b1,[0],None,[256],[0,256])
hist1 = np.hstack([rhist1,ghist1,bhist1])

p1 = hist1/np.sum(hist1)
p1 = p1[p1 > 0]
img_en1 = -np.sum(p1 * np.log2(p1))

r2, g2, b2 = cv.split(img2)
rhist2 = cv.calcHist(r2,[0],None,[256],[0,256])
ghist2 = cv.calcHist(g2,[0],None,[256],[0,256])
bhist2 = cv.calcHist(b2,[0],None,[256],[0,256])
hist2 = np.hstack([rhist2,ghist2,bhist2])

p2 = hist2/np.sum(hist2)
p2 = p2[p2 > 0]
img_en2 = -np.sum(p2 * np.log2(p2))

print('img_en1:',img_en1, 'img_en1:',img_en2)

plt.figure(3)
plt.subplot(4,1,1)
plt.imshow(img1)
plt.subplot(4,1,2)
plt.imshow(heatmap1)
plt.subplot(4,1,3)
plt.imshow(img2)
plt.subplot(4,1,4)
plt.imshow(heatmap2)

plt.figure(4)
plt.imshow(img3)
plt.show()
plt.figure(5)
plt.plot(rhist1)
plt.plot(rhist2)
plt.show()



# # 熵
# def entropy(signal):
#         lensig=signal.size
#         symset=list(set(signal))
#         propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]#每个值的概率
#         ent=np.sum([p*np.log2(1.0/p) for p in propab])
#         return ent
# # 读图，也可以用Opencv啥的
# colorIm=np.array(img1)
# # 灰度
# greyIm=np.array(colorIm)
# N=3
# S=greyIm.shape
# E=np.array(greyIm)
#
# #以图像左上角为坐标0点
# for row in range(S[0]):
#     for col in range(S[1]):
#         Left_x=np.max([0,col-N])
#         Right_x=np.min([S[1],col+N])
#         up_y=np.max([0,row-N])
#         down_y=np.min([S[0],row+N])
#         region=greyIm[up_y:down_y,Left_x:Right_x].flatten()  # 返回一维数组
#         E[row,col]=entropy(region)
#
# plt.subplot(1,3,1)
# plt.imshow(colorIm)
#
# plt.subplot(1,3,2)
# plt.imshow(greyIm, cmap=plt.cm.gray)
#
# plt.subplot(1,3,3)
# plt.imshow(E, cmap=plt.cm.jet)
# plt.xlabel('6x6 邻域熵')
# plt.colorbar()
#
# plt.show()








