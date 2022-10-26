import io
import numpy as np

num = 376

'''create calib'''
path = '/home/xrd/PycharmProjects/VPfusion/data/collect_xinzhen_32/training/calib/'
calib_model = '000000.txt'

modelpath = path + calib_model
f = open(modelpath)
calib = f.read()
f.close()

for idx in range(num-1):
    wrt_path = path + str(idx+1).zfill(6) + '.txt'
    with open(wrt_path,'w') as wrt:
        wrt.write(calib)


'''create imageset'''
# wrt_path = '/home/xrd/PycharmProjects/VPfusion/data/collect_xinzhen_32/ImageSets/train.txt'
# indexnum = []
# for idx in range(num):
#     indexnum.append(str(idx).zfill(6) + '\n')
#
# # indexnum = np.array(indexnum)
# with open(wrt_path,'w') as wrt:
#     wrt.writelines(indexnum)