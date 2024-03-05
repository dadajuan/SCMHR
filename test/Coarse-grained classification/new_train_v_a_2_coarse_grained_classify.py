
import torch
from torch import nn
import argparse
import torch
from torchvision import transforms

import numpy as np

from torch.utils.data import Dataset
import cv2
from torch.autograd import Variable
from datset_V_A_H import MyDataSet_V_A_H
from utils import read_split_data, read_split_data_audio, plot_data_loader_image, read_split_data_tac_densenet, read_split_data_haptic

from new_haptic_encoder_gen import haptic_encoder, haptic_encoder_fenlei, haptic_dis_gan, VAE_GAN, Encoder_Image_VGG, Encoder_Audio_VGG, cross_modal_fusion, haptic_Generator, TransformerLayer_fusion, Encoder_Image_Resnet, Encoder_Audio_Resnet

from metric import RMSE_loss, SIM, sumprescise, contrastive_loss, train_save_images, ST_SIM
import wandb
import datetime
import os

'''
搞一个一个分类网络,完成跨模态的识别，这里是一个跨模态的分类实验
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(torch.__version__)

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



run_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '粗分类'+'实验测试以提高指标性能'


k_constra = 0

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--temp_contra_loss", type=float, default=10, help="对比损失比例")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

wandb.login(key = 'f8b62de4cb0df7e76a915afe08278b71b96393a8')
wandb.init(
      # Set the project where this run will be logged
      project="v+a_haptic",
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"experiment_{run_name}",
      # Track hyperparameters and run metadata
      config={
      "encoder_v_architecture": "resnet",
      "epochs": {opt.n_epochs},
      "batch_size": {opt.batch_size},
    '优化方式': 'one_loss_optim_all',
    '分类中对比损失函数的比例':{k_constra},
    "图像的归一化方式":'0-1',
    "触觉信号的归一化方式":'0-1',
    "融合方式：": "transformer"
      })



output_folder = "output_haptic_files_gan"  # 保存生成的触觉信号，保存的是txt文件

os.makedirs(output_folder, exist_ok=True)

folder_name_save_images = 'saved_images_train_gan' # 保存训练过程中生成的触觉信号并画图；
os.makedirs(folder_name_save_images, exist_ok=True)


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

#print(train_data_set)


train_loader = torch.utils.data.DataLoader(train_data_set,
                                           batch_size=opt.batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           drop_last=True,
                                           collate_fn=train_data_set.collate_fn)

val_loader = torch.utils.data.DataLoader(val_data_set,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=0,
                                         drop_last=True,  # 把最后的几个不够一个batch的去掉
                                         collate_fn=val_data_set.collate_fn)


'''定义不同的网络'''

print('device:', device)

# Encoder_Image = Encoder_Image_VGG().to(device)
# Encoder_Audio = Encoder_Audio_VGG().to(device)

Encoder_Image = Encoder_Image_Resnet().to(device)
Encoder_Audio = Encoder_Audio_Resnet().to(device)

encoder_fenlei = haptic_encoder_fenlei(512).to(device)  # 这是两层全连接的参数表示,网络输出的第二个用来计算全连接的损失

#gen_haptic = haptic_Generator(512).to(device)

#dis_haptic = haptic_dis_gan(1600).to(device)

#新的基于transformer的融合方式
Fusion_I_A = TransformerLayer_fusion(512, 8).to(device)

# 定义损失函数和优化器

criterion_fenlei = nn.CrossEntropyLoss()

optimizer_encoder_fenlei = torch.optim.Adam(list(Fusion_I_A.parameters())+list(Encoder_Image.parameters())+list(Encoder_Audio.parameters())+list(encoder_fenlei.parameters()), lr=0.0002, betas=(0.5, 0.999))  # 优化后两层，融合网络之后的分类器

best_val_haptic_mse = 1
batches_done = 0

''' 开始训练所有网络'''
for i in range(opt.n_epochs):
    total_fenlei = []
    for j, (image_batch, aduio_batch, haptic_batch, label_batch, filepath) in enumerate(train_loader):
        # train()

        label_batch_target = Variable(label_batch, requires_grad=False).to(device) # 标签.to(device)

        #print('label_batch_target:', label_batch_target)

        images = Variable(image_batch, requires_grad=False).float().to(device)  # [16,3,128,128]

        audios = Variable(aduio_batch, requires_grad=False).float().to(device) # [16,3, 217 , 334]

        haptics = Variable(haptic_batch, requires_grad=False).float().to(device)

        #对信号进行归一化
        images = images/255.0
        audios = audios/255.0

        '''需要开展一个归一化的函数,把触觉信号归一化到0-1之间'''

        min_val, _ = haptics.min(dim=1, keepdim=True)  # 沿维度1找到每行的最小值
        max_val, _ = haptics.max(dim=1, keepdim=True)  # 沿维度1找到每行的最大值
        normalized_haptics = (haptics - min_val) / (max_val - min_val)
        haptics = normalized_haptics


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


        constra_loss = contrastive_loss(I_encoder, A_encoder, label_batch_target, opt.temp_contra_loss)
        print('当前批次的constra_loss:', constra_loss )

        loss_total = fenlei_loss + k_constra * constra_loss

        optimizer_encoder_fenlei.zero_grad()

        loss_total.backward()
        optimizer_encoder_fenlei.step()

        print('len(label_batch_target),16', len(label_batch_target))
        print('len(train_loader),108', len(train_loader))


        print('epoch[{0}/{1}],bathchid[{2}/{3}],contra_Loss:{4},fenlei_loss{5},acc{6}'.format (i, opt.n_epochs, j, len(train_loader), constra_loss, fenlei_loss, accuracy_batch))
    accuracy_total = sum(total_fenlei) / (len(label_batch_target)*len(train_loader))

    print('当前第{}个epoch整个的accuracy{}'.format(i, accuracy_total))
    wandb.log({'train_constr_loss': constra_loss, 'train_fenlei_loss': fenlei_loss, 'acc':accuracy_batch, 'acc_epoch': accuracy_total})


    if i%10==0:
        best_Encoder_Image_state = Encoder_Image.state_dict()
        best_Encoder_Audio_state = Encoder_Audio.state_dict()
        best_Fusion_I_A_state = Fusion_I_A.state_dict()
        best_encoder_fenlei_state = encoder_fenlei.state_dict()

        torch.save(best_Encoder_Image_state, f"Encoder_Image_fenlei_full.pth")
        torch.save(best_Encoder_Audio_state, f"Encoder_Audio_fenlei_full.pth")
        torch.save(best_Fusion_I_A_state, "Fusion_I_A_fenlei_full.pth")
        torch.save(best_encoder_fenlei_state, "encoder_haptic_fenlei_full.pth")

wandb.finish()
