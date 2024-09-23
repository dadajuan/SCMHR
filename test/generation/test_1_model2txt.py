import torch
from torch import nn

'''
当模型训练完之后运行本代码；
保存训练的每一个样本生成的触觉信号;
之后去运行text2代码去画图,新的


'''
import torch
from torchvision import transforms
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(torch.__version__)

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import cv2
from torch.autograd import Variable
from datset_V_A_H import MyDataSet_V_A_H
from utils_test_model2txt import read_split_data, read_split_data_audio, read_split_data_haptic

from new_haptic_encoder_gen import haptic_Generator, TransformerLayer_fusion, Encoder_Image_Resnet, Encoder_Audio_Resnet

from metric import RMSE_loss, SIM, sumprescise, contrastive_loss, ST_SIM

k_mse = 10   #
batch_size = 16
#按照0.2的比例选测试集和训练接

output_folder = "output_haptic_files_1.2_gan"  # 保存生成的触觉信号，保存的是txt文件

os.makedirs(output_folder, exist_ok=True)

train_images_path = 'D:\Desktop\ACM10.7\coding\source_data_train_test\Visual'

train_audio_path = 'D:\Desktop\ACM10.7\coding\source_data_train_test\Audio'

train_tactile_path = 'D:\Desktop\ACM10.7\coding\source_data_train_test\Haptic'


train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(train_images_path)


train_audio_path, train_audio_label, val_audio_path, val_audio_label = read_split_data_audio(train_audio_path)


train_tactile_path, train_tactile_label, val_tactile_path, val_tactile_label = read_split_data_haptic(train_tactile_path)


''''数据集是如何做到对齐的，即虽然read_Split_aduio这些在运行，但其他后面根本是没用到的，所有的都是根据img的path更改的，即MyDataSet_V_A_H这个模型修改的'''


train_data_set = MyDataSet_V_A_H(images_path=train_images_path,
                                 audio_path=train_audio_path,
                                 haptic_path=train_tactile_path,
                                 images_class=train_images_label,
                                 transform=None)


val_data_set = MyDataSet_V_A_H(images_path=val_images_path,
                                      audio_path=val_audio_path,
                                      haptic_path=val_tactile_path,
                                      images_class=val_images_label,
                                      transform=None)


train_loader = torch.utils.data.DataLoader(train_data_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           drop_last=True,
                                           collate_fn=train_data_set.collate_fn)

val_loader = torch.utils.data.DataLoader(val_data_set,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=0,
                                         drop_last=True,  # 把最后的几个不够一个batch的去掉
                                         collate_fn=val_data_set.collate_fn)

print('device:', device)


Encoder_Image = Encoder_Image_Resnet().to(device)
Encoder_Audio = Encoder_Audio_Resnet().to(device)
gen_haptic = haptic_Generator(512).to(device)
Fusion_I_A = TransformerLayer_fusion(512,8).to(device)  # 这个是说明融合网络的最终输出的维度


file_path1 = 'Encoder_Image_gan.pth'
file_path2 = 'Encoder_Audio_gan.pth'
file_path3 = 'Fusion_I_A_gan.pth '
file_path4 = 'gen_haptic_gan.pth'


Encoder_Image.load_state_dict(torch.load(file_path1))

Encoder_Audio.load_state_dict(torch.load(file_path2))

Fusion_I_A.load_state_dict(torch.load(file_path3))

gen_haptic.load_state_dict(torch.load(file_path4))


# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失函数,GAN网络的损失
mse_loss = torch.nn.MSELoss()  # 生成器的损失,也是自编码器的损失
criterion_fenlei = nn.CrossEntropyLoss()   # 编码网路最后两层的损失，即分类损失,交叉熵损失


def evaluate_model(enconder_I,  encoder_A,  Fusion_I_A, gen_haptic, dataloader):
    encoder_A.eval()
    enconder_I.eval()
    Fusion_I_A.eval()

    gen_haptic.eval()

    haptic_loss_mse_total = []
    haptic_loss_rmse_total = []
    haptic_loss_sim_total = []
    haptic_loss_st_sim_total = []

    with torch.no_grad():

        for j, (image_batch, aduio_batch, haptic_batch, label_batch, filepath) in enumerate(val_loader):
            # train()
            label_batch_target = Variable(label_batch, requires_grad=False).to(device)  # 标签.to(device)

            # print('label_batch_target:', label_batch_target)

            images = Variable(image_batch, requires_grad=False).float().to(device)/255.0

            audios = Variable(aduio_batch, requires_grad=False).float().to(device)/255.0

            haptics = Variable(haptic_batch, requires_grad=False).float().to(device)


            '''需要开展一个归一化的函数,把触觉信号归一化到0-1之间'''
            min_val, _ = haptics.min(dim=1, keepdim=True)  # 沿维度1找到每行的最小值
            max_val, _ = haptics.max(dim=1, keepdim=True)  # 沿维度1找到每行的最大值
            normalized_haptics = (haptics - min_val) / (max_val - min_val)
            haptics = normalized_haptics
            haptic_real = haptics


            I_encoder = Encoder_Image(images)
            A_encoder = Encoder_Audio(audios)
            fusion_I_A = Fusion_I_A(I_encoder, A_encoder)

            haptic_gen = gen_haptic(fusion_I_A)
            haptic_loss_mse = mse_loss(haptic_gen, haptic_real)  # 这个里面已经求平均了
            haptic_loss_mse_total.append(haptic_loss_mse) # 这个是求的一个一个批次平均的；

            haptics_loss_rmse = RMSE_loss(haptic_gen, haptic_real)  # 这个是求这个批次的每个的
           # print('haptics_loss_rmse:', haptics_loss_rmse)
            haptic_loss_rmse_total.extend(haptics_loss_rmse)

            haptic_loss_sim = SIM(haptic_gen, haptic_real)  # 求得也是每个批次的；
            haptic_loss_sim_total.extend(haptic_loss_sim)
            #print('haptics_loss_sim:', haptic_loss_sim)

            haptic_loss_st_sim = ST_SIM(haptic_gen, haptic_real)
            haptic_loss_st_sim_total.append(haptic_loss_st_sim)


            # print('一共计算了{}个MSE,{}个RMSE,{}个SIM,前者*16应等于后者'.format(len(haptic_loss_mse_total), len(haptic_loss_rmse_total), len(haptic_loss_sim_total)))
            #
            # print('-------------MSE-----------------------')
            # print('到目前的所有批次的平均mse_loss：{},一共{}个batch，损失的最小值是{},最大值是{}'.format(
            #     sum(haptic_loss_mse_total) / len(haptic_loss_mse_total), len(haptic_loss_mse_total),
            #     min(haptic_loss_mse_total), max(haptic_loss_mse_total)))
            #
            # print('-------------RMSE-----------------------')
            # print('到目前的所有批次的平均rmse_loss：{},一共{}个元素，损失的最小值是{},最大值是{}'.format(
            #     sum(haptic_loss_rmse_total) / len(haptic_loss_rmse_total), len(haptic_loss_rmse_total), min(haptic_loss_rmse_total), max(haptic_loss_rmse_total)))
            #
            # print('----------------SIM-------------------')
            # print('到目前的所有批次的平均SIM_loss：{},一共{}个元素，SIM的最小值是{},SIM最大值是{}'.format(
            #     sum(haptic_loss_sim_total) / len(haptic_loss_sim_total), len(haptic_loss_sim_total),
            #     min(haptic_loss_sim_total), max(haptic_loss_sim_total)))
            #
            # print('----------------ST-SIM-------------------')
            # print('到目前的所有批次的平均SIM_loss：{},一共{}个元素，SIM的最小值是{},SIM最大值是{}'.format(
            #     sum(haptic_loss_st_sim_total) / len(haptic_loss_st_sim_total), len(haptic_loss_st_sim_total),
            #     min(haptic_loss_st_sim_total), max(haptic_loss_st_sim_total)))

            '''保存生成的触觉信号为列表,并把他们保存到txt文件中'''
                # fake_haptic_list = fake_haptic.tolist()
            file_name = filepath  # 原始的文件名字
            modified_filename = []  # 把生成的测试集合的数据都重新命名然后保存
                # 遍历元组中的字符串

            '''保存所有的需要保留的样本的名称 '''
            for item in file_name:
                # 在这个示例中，我们将每个字符串转为大写并添加 " - Modified" 后缀
                if item.split('_')[-1][1] == 'r':
                    modified_string = "DFT321" + item.split("Image")[
                        0] + f"Movement_train{item.split('Image_')[1].split('_')[0]}_gen_fake.txt"

                else:
                        item.split('_')[-1][1] == 'e'
                        modified_string = "DFT321" + item.split("Image")[
                            0] + f"Movement_test{item.split('Image_')[1].split('_')[0]}_gen_fake.txt"
                modified_filename.append(modified_string)

            '''保存所有生成的触觉样本 '''
            for i in range(batch_size):
                row = haptic_gen[i].tolist()  # 将一行转换为Python列表
                row_str = "\n".join(map(str, row))  # 将列表元素转换为字符串，并用空格隔开
                file_name_gen = os.path.join(output_folder, modified_filename[i])
                with open(file_name_gen, "w") as file:
                    file.write(row_str)
            print('--保存跨模态生成的触觉信号第{}批次,一共保存了{}样本'.format(j, j * 16))


            '''保存原始的触觉信号'''
            # 这两个都是按照
            modified_filentest_1_model2txt.pyame_real_haptic = []

            for item in file_name:
                if item.split('/')[-1].split('_')[-1][1] == 'r':
                    modified_string_real = "DFT321" + item.split('/')[-1].split("Image")[
                        0] + f"Movement_train{item.split('Image_')[1].split('_')[0]}.txt"
                else:
                    item.split('/')[-1].split('_')[-1][1] == 'e'
                    modified_string_real = "DFT321" + item.split('/')[-1].split("Image")[
                        0] + f"Movement_test{item.split('Image_')[1].split('_')[0]}.txt"

                modified_filename_real_haptic.append(modified_string_real)


            for i in range(batch_size):
                row_real = haptic_real[i].tolist()  # 将一行转换为Python列表
                row_real_str = "\n".join(map(str, row_real))  # 将列表元素转换为字符串，并用空格隔开

                file_name_real = os.path.join(output_folder,
                                              modified_filename_real_haptic[i])  # 使用原始的文件名列表中的元素作为文件名

                with open(file_name_real, "w") as file:
                    file.write(row_real_str)
            print('--保存原始的触觉信号第{}批次,一共保存了{}样本'.format(j, j * 16))

        mse_average = sum(haptic_loss_mse_total)/len(haptic_loss_mse_total)
        rmse_average = sum(haptic_loss_rmse_total)/len(haptic_loss_rmse_total)
        sim_average = sum(haptic_loss_sim_total) / len(haptic_loss_sim_total)


    return  mse_average, rmse_average, sim_average

val_haptic_mse, val_haptic_rmse, val_haptic_sim = evaluate_model(Encoder_Image, Encoder_Audio, Fusion_I_A,
                                                                     gen_haptic, val_loader)
