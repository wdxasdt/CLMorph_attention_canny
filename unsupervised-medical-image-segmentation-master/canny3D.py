import cv2
import nibabel as nib
from skimage.feature import canny
import numpy as np
# import matplotlib
# # matplotlib.use('TkAgg')
# from matplotlib import pylab as plt
def canny_edges_3d(grayImage):
    # canny算法参数
    #（1）如果当前边缘像素的梯度值大于或等于 maxVal，则将当前边缘像素标记为强边缘。
    #（2）如果当前边缘像素的梯度值介于 maxVal 与 minVal 之间，则将当前边缘像素标记为虚边缘（需要保留）。
    #（3）如果当前边缘像素的梯度值小于或等于 minVal，则抑制当前边缘像素。
    #在上述过程中，我们得到了虚边缘，需要对其做进一步处理。一般通过判断虚边缘与强边缘是否连接，来确定虚边缘到底属于哪种情况。通常情况下，如果一个虚边缘
    MIN_CANNY_THRESHOLD = 0.25
    MAX_CANINY_THRESHOLD = 0.4

    # 得到图像维度 初始化edges_x\y\z为与输入图像相同大小的bool类型0矩阵，用于存储边缘像素的位置
    dim = np.shape(grayImage)
    edges_x = np.zeros(grayImage.shape,dtype=bool)
    edges_y = np.zeros(grayImage.shape,dtype=bool)
    edges_z = np.zeros(grayImage.shape,dtype=bool)
    edges = np.zeros(grayImage.shape,dtype=bool)
    # print(np.shape(edges))

    # for循环对三个维度进行遍历，对每切片使用canny函数进行边缘检测，并将结果存入edges_x\y\z
    for i in range(dim[0]):
        edges_x[i,:,:] = canny(grayImage[i,:,:], low_threshold=MIN_CANNY_THRESHOLD,
                               high_threshold=MAX_CANINY_THRESHOLD, sigma = 0)
    for j in range(dim[1]):
        edges_y[:,j,:]= canny(grayImage[:,j,:], low_threshold=MIN_CANNY_THRESHOLD,
                              high_threshold=MAX_CANINY_THRESHOLD,  sigma = 0)
    for k in range(dim[2]):
        edges_z[:,:,k]= canny(grayImage[:,:,k], low_threshold=MIN_CANNY_THRESHOLD,
                              high_threshold=MAX_CANINY_THRESHOLD,  sigma = 0)
    # edges = canny(grayImage， low_threshold=IN_CANY_THRESHOLD,high_threshold=NAx_CAMY_THRESHOLD)
    # 使用三重for循环将三个维度的边缘像素位置合并到edges中
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                edges[i,j,k] = (edges_x[i,j,k] and edges_y[i,j,k]) or \
                               (edges_x[i,j,k] and edges_z[i,j,k]) or (edges_y[i,j,k] and edges_z[i,j,k])

    return edges

def save_nii(data ,save_name ,affine,header):
    new_img = nib.Nifti1Image(data.astype(np.int16),affine,header)
    nib.save(new_img,save_name)
# 读取nii
def load_nii(filename) :
    Image = nib.load(filename)
    img_arr = Image.get_fdata()  # img_arr是一个ndarray类型的三维矩阵
    name = filename.split('/')[-1]
    return img_arr.astype(np.float32), Image.affine, Image.header
    # 返回的img_arr是一个ndarray类型的三维矩阵，affine是放射变换矩阵 header是nii文件头信息


def canny3D(img):
    # img,shape,name,affine,header = load_nii(filename)
    # img, shape,name, affine, header = filename
    # print(img)
    edges = canny_edges_3d(img)
    edges = np.array(edges).astype(int)
    # edges = edges.get_fdata()
    # save_nii(edges,new_path,affine,header)
    # print(edges)
    return edges
# print(edges)

if __name__ == '__main__':
    import os
    import os

    source_path = 'datasets/LPBA40/LPBA40_rigidly_registered_pairs_histogram_standardization_small'
    target_path = 'datasets/LPBA40/canny'
    file_list = os.listdir(source_path)
    print(file_list)
    for file in file_list:
        source_file = os.path.join(source_path, file)
        target_file = os.path.join(target_path, file)
        load_source, affine, header = load_nii(source_file)
        edges = canny3D(load_source)
        save_nii(edges,target_file, affine, header)


    # filename = load_nii('LPBA40/fixed.nii.gz')
    # edges,affine, header = canny3D(filename)
    # print(edges)
    # save_nii(edges,new_path,affine,header)
    # img = nib.load(new_path)
    # width, heigh, queue = img.dataobj.shape
    # print(img.dataobj.shape)
    # num = 1
    # for i in range(0, queue, 10): # 从0到queue-1，每隔10个数返回一个，这里的queue是一个变量，表示数据在第三个维度上的大小
    #     img_arr = img.dataobj[:, :, i]
    #     plt.subplot(5, 4, num) # 第num行，第num%4列的位置
    #     plt.imshow(img_arr, cmap='gray', origin='lower')
    #     num += 1
    # plt.show()
