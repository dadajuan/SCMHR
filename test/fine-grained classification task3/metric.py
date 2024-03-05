
import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
#输入的tensor已经是[batch_size, 1600]

from vibromaf.metrics.stsim import st_sim
##计算方式是求两个序列信号的每个数的差值，然后求平方和，然后除以序列信号的长度，然后再开根号
def  RMSE_loss(tensor1, tensor2):
    rowwise_diff = (tensor1 - tensor2) ** 2
    rowwise_mse = torch.mean(rowwise_diff, dim=1)
    print(rowwise_mse.shape)
    rowwise_rmse = torch.sqrt(rowwise_mse)

    rowwise_rmse_list = rowwise_rmse.tolist()

    return rowwise_rmse_list


def contrastive_loss(fv, fa, label_batch,lambda1 ):

    '''输入分别是视觉特征，触觉特征，一个批次的标签，温度系数'''

    '''计算视觉与触觉对应的相关性'''
    #计算每个视觉与每个触觉对应的相似性
    similarity_matrix_v2a = torch.matmul(fv, fa.t())  # 视觉是每行，然后听觉是每一列
    similarity_matrix_v2a_exp = torch.exp(similarity_matrix_v2a/lambda1)
    # 计算每行的和
    row_sums = similarity_matrix_v2a_exp.sum(dim=1, keepdim=True)
    N = len(label_batch)
    # 归一化矩阵
    normalized_similarity_row = similarity_matrix_v2a_exp / row_sums  # 按行归一化的矩阵
    '''获得混要矩阵，该矩阵对视觉和听觉都是同样有用滴'''
    # 获得这个对应的标签，即视觉和触觉是否属于同一类的一个元素为0/1的矩阵
    hunyaometric = (label_batch.unsqueeze(1) == label_batch.unsqueeze(0)).int()  # 这个对于A和B是一致的；

    # 把混要矩阵中等与1的替换为相似矩阵中的值, 把其中等于0的替换为0
    v2a_sim = normalized_similarity_row * (hunyaometric == 1).float()
    #print('v2a_sim:', v2a_sim )
    # 求交叉熵损失，然后对一个批次中的所有的求和
    mask = v2a_sim != 0
    v_a_cross_entropy = torch.log(v2a_sim[mask]).sum()
    #print('v_a_cross_entropy:', v_a_cross_entropy.item())


    '''计算触觉与视觉对应的相关性'''
    #计算每个触觉与视觉对应的相关性：
    similarity_matrix_a2v = torch.matmul(fa, fv.t())  # 听觉是每行，然后触觉是每一列
    similarity_matrix_a2v_exp = torch.exp(similarity_matrix_a2v/lambda1)
    #计算每一行的和
    row_sums_a2v = similarity_matrix_a2v_exp.sum(dim=1, keepdim=True)
    # 归一化矩阵
    normalized_similarity_row_a2v = similarity_matrix_a2v_exp / row_sums_a2v  # 按行归一化的矩阵

    # 把混要矩阵中等与1的替换为相似矩阵中的值, 把其中等于0的替换为0
    a2v_sim = normalized_similarity_row_a2v * (hunyaometric == 1).float()
    #print('a2v_sim',a2v_sim)
    # 求交叉熵损失，然后对一个批次中的所有的求和
    mask2 = a2v_sim != 0
    a_v_cross_entropy = torch.log(a2v_sim[mask2]).sum()

    #print('a_v_cross_entropy:', a_v_cross_entropy.item())
    # 对二者进行求和
    sum = -(1/N)*(1/2)*(v_a_cross_entropy.item()+a_v_cross_entropy.item())
    return sum



#这个也是把一个批次里面的都打印出来，然后我到最后再统一计算。
def SIM(predictions, targets):
    '''计算两个序列的SIM值'''
    row_chebyshev_distances = torch.max(torch.abs(predictions - targets), dim=1).values  # 求一个batch中每个样本的距离最大的差值

    exp_tensor = torch.exp(row_chebyshev_distances)
    result_one_batch = 1 / exp_tensor  # 得到每个样本的最大值
    print(result_one_batch.shape)
    result_one_batch_list = result_one_batch.tolist()

    return result_one_batch_list

def sumprescise(encoder_hr_z_logit, data_target):
    lena = len(data_target)
    precisenum = 0
    for i in range(len(data_target)):
        a1 = list(encoder_hr_z_logit[i])
        if a1.index(max(a1)) == data_target[i]:
            precisenum = precisenum+1
    return precisenum, lena

def normalize_images(batch_images):
    # 定义均值和标准差
    mean = [0.485, 0.456, 0.406]  # ImageNet 数据集的均值
    std = [0.229, 0.224, 0.225]  # ImageNet 数据集的标准差

    # 定义归一化转换
    normalize = transforms.Normalize(mean=mean, std=std)

    # 应用归一化转换
    normalized_images = torch.stack([normalize(img) for img in batch_images])

    return normalized_images

def sumprescise(encoder_hr_z_logit, data_target):
    lena = len(data_target)
    precisenum = 0
    for i in range(len(data_target)):
        a1 = list(encoder_hr_z_logit[i])
        if a1.index(max(a1)) == data_target[i]:
            precisenum = precisenum+1
    return precisenum, lena

def normalize_images(batch_images):
    # 定义均值和标准差
    mean = [0.485, 0.456, 0.406]  # ImageNet 数据集的均值
    std = [0.229, 0.224, 0.225]  # ImageNet 数据集的标准差

    # 定义归一化转换
    normalize = transforms.Normalize(mean=mean, std=std)

    # 应用归一化转换
    normalized_images = torch.stack([normalize(img) for img in batch_images])

    return normalized_images

def train_save_images(haptic_gen, haptci_real, batch_done, folder_name_save_images):
# 设置图像尺寸
    fig, ax = plt.subplots(figsize=(15, 5))
    A = haptic_gen
    B = haptci_real
    # 将数据按横轴切分成16张图并保存到指定文件夹
    for i in range(A.size(0)):
        # 获取当前切片的数据
        slice_A = A[i, :].cpu().detach().numpy()  # 获取第i行数据
        slice_B = B[i, :].cpu().detach().numpy()  # 获取第i行数据

        # 绘制A和B的图像
        ax.plot(slice_A, label='haptci_gen')
        ax.plot(slice_B, label='haptci_real')

        # 添加标题、标签等
        ax.set_title(f'Plot {i+1}')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Value')
        ax.legend()

        # 保存图像到指定文件夹，命名为plot_1.png至plot_16.png
        file_path = os.path.join(folder_name_save_images, f'plot_{batch_done}_{i+1}.png')
        plt.savefig(file_path, bbox_inches='tight')

        # 清空图像以便绘制下一张图
        ax.clear()

    # 关闭图表
    plt.close()

def ST_SIM(haptic_gen, haptic_real):
    a = [ ]
    for i in range(haptic_gen.size(0)):
        gen = haptic_gen[i, : ].unsqueeze(0)
        gen = gen.squeeze()  # 去掉维度为1的情况
        gen = gen.cpu().detach().numpy()
        # print(type(gen))
        # print(gen.shape)

        ref = haptic_real[i, : ].unsqueeze(0)
        ref = ref.squeeze()
        ref = ref.cpu().detach().numpy()
        #print(type(ref))

        st_sim_score = st_sim(gen, ref)
        a.append(st_sim_score)
        a_array = np.array(a)
        a_mean = np.mean(a_array)
    return a_mean