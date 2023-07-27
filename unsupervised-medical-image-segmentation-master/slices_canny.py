import torch
import cv2
from canny3D import canny3D
# 假设输入张量为 tensor，维度为（8，1，72，96，72）

def slices_canny(tensor):
    # 将第一维度拆分为单独的张量切片
    slices = torch.split(tensor, 1, dim=0)

    # 逐个处理每个切片
    processed_slices = []
    for slice in slices:
        # 将张量切片转换为 numpy 数组
        array = slice.squeeze().numpy()

        # 对每个切片进行 Canny 边缘检测处理
        edges = canny3D(array)

        # 将处理后的结果转换回张量，并添加到列表中
        processed_slices.append(torch.from_numpy(edges).unsqueeze(0))

    # 将处理后的张量切片重新组合为张量
    processed_tensor = torch.cat(processed_slices, dim=0)
    processed_tensor = processed_tensor.unsqueeze(1)
    return processed_tensor