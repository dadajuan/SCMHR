import torch
from torch import nn

'''
模型训练完毕之后，检验模型的性能指标
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

from new_haptic_encoder_gen import haptic_Generator, TransformerLayer_fusion, Encoder_Image_Resnet, Encoder_Audio_Resnet, haptic_encoder_fenlei

from metric import RMSE_loss, SIM, sumprescise, contrastive_loss, ST_SIM

k_mse = 10   #
batch_size = 16
#按照0.2的比例选测试集和训练接

# output_folder = "output_haptic_files_1.2_gan"  # 保存生成的触觉信号，保存的是txt文件
#
# os.makedirs(output_folder, exist_ok=True)

# train_images_path = 'D:\Desktop\ACM10.7\coding\source_data_train_test\Visual'
#
# train_audio_path = 'D:\Desktop\ACM10.7\coding\source_data_train_test\Audio'
#
# train_tactile_path = 'D:\Desktop\ACM10.7\coding\source_data_train_test\Haptic'


train_images_path = '/data1/liaojunqi/ljq/coding/source_data_train_test/Visual'

train_audio_path = '/data1/liaojunqi/ljq/coding/source_data_train_test/Audio'

train_tactile_path = '/data1/liaojunqi/ljq/coding/source_data_train_test/Haptic'


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
#gen_haptic = haptic_Generator(512).to(device)
Fusion_I_A = TransformerLayer_fusion(512,8).to(device)  # 这个是说明融合网络的最终输出的维度
encoder_fenlei = haptic_encoder_fenlei(512).to(device)


file_path1 = 'Encoder_Image_fenlei_fine.pth'
file_path2 = 'Encoder_Audio_fenlei_fine.pth'
file_path3 = 'Fusion_I_A_fenlei_fine.pth'
file_path4 = 'encoder_haptic_fenlei_fine.pth'


Encoder_Image.load_state_dict(torch.load(file_path1))

Encoder_Audio.load_state_dict(torch.load(file_path2))

Fusion_I_A.load_state_dict(torch.load(file_path3))

encoder_fenlei.load_state_dict(torch.load(file_path4))



# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失函数,GAN网络的损失
mse_loss = torch.nn.MSELoss()  # 生成器的损失,也是自编码器的损失
criterion_fenlei = nn.CrossEntropyLoss()   # 编码网路最后两层的损失，即分类损失,交叉熵损失
k_constra = 0

def evaluate_model(Encoder_Image,  Encoder_Audio,  Fusion_I_A, encoder_fenlei, dataloader):

    total_fenlei = []
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

            images = images / 255.0
            audios = audios / 255.0

            '''需要开展一个归一化的函数,把触觉信号归一化到0-1之间'''

            '''鉴别器网络的优化'''

            I_encoder = Encoder_Image(images)
            A_encoder = Encoder_Audio(audios)

            fusion_I_A = Fusion_I_A(I_encoder, A_encoder)
            outputs = encoder_fenlei(fusion_I_A)

            # 转换预测结果为类别标签
            predicted_labels = torch.argmax(outputs, dim=1)
            correct = (predicted_labels == label_batch_target).sum().item()
            total_fenlei.append(correct)
            accuracy_batch = correct / len(label_batch_target)

            print('correct:', correct, '分母:', len(label_batch_target))

            fenlei_loss = criterion_fenlei(outputs, label_batch_target)
            print('分类损失：', fenlei_loss)

            constra_loss = contrastive_loss(I_encoder, A_encoder, label_batch_target,10)
            print('当前批次的constra_loss:', constra_loss)

            #loss_total = fenlei_loss + k_constra * constra_loss

            print('bathchid[{0}/{1}],contra_Loss:{2},fenlei_loss{3},acc{4}'.format(j,
                                                                                   len(val_loader),
                                                                                                  constra_loss,
                                                                                                  fenlei_loss,
                                                                                                  accuracy_batch))
        print('分子：', sum(total_fenlei), '分母：', (len(label_batch_target) * len(val_loader)))
        accuracy_total = sum(total_fenlei) / (len(label_batch_target) * len(val_loader))
        print('总的分类损失：', accuracy_total)

    return accuracy_total

accuracy_total = evaluate_model(Encoder_Image, Encoder_Audio, Fusion_I_A, encoder_fenlei, val_loader)