import cv2
import os
def img2avi(file_dir):
    # 要被合成的多张图片所在文件夹
    # 路径分隔符最好使用“/”,而不是“\”,“\”本身有转义的意思；或者“\\”也可以。
    # 因为是文件夹，所以最后还要有一个“/”

    list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            list.append(file)
            # VideoWriter是cv2库提供的视频保存方法，将合成的视频保存到该路径中
    # 'MJPG'意思是支持jpg格式图片
    # fps = 5代表视频的帧频为5，如果图片不多，帧频最好设置的小一点
    # (1280,720)是生成的视频像素1280*720，一般要与所使用的图片像素大小一致，否则生成的视频无法播放
    # 定义保存视频目录名称和压缩格式，像素为1280*720
    fps =20
    video = cv2.VideoWriter(file_dir + '/../test.avi',
                            cv2.VideoWriter_fourcc(*'MJPG'),fps,(800,600))

    for i in range(1,len(list)):
        #读取图片
        img = cv2.imread(file_dir+"%s"%(i-1) + ".jpg")
        # resize方法是cv2库提供的更改像素大小的方法
        # 将图片转换为1280*720像素大小
        img = cv2.resize(img, (800, 600))
        # 写入视频
        video.write(img)

    # 释放资源
    video.release()

if __name__ == '__main__':
    img2avi()