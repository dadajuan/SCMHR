
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
训练方式
单模态的编码网络不用分类损失去优化了，而是直接全部都采用最后的损失去优化；
前面这个融合网络的框架用这两个去优化；
这里把验证集的测试加上了.方便随时检验
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(torch.__version__)

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



run_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + 'gan-对比损失-resnet参与训练'


k_mse = 100
k_duibisunshi  = 0.1

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--temp_contra_loss", type=float, default=10, help="adam: learning rate")
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
      "GAN的类型": "wgan",
      "learning_rate_encoderV+I+F+G": 0.0005,
      "learning_rate_D": 0.0005,
      "encoder_v_architecture": "resnet",
      "encoder_v_train": "True",
      "epochs": {opt.n_epochs},
      "batch_size": {opt.batch_size},
    '优化方式': 'one_loss_optim_all',
    '生成器中损失函数的比例':{k_mse},
    "图像的归一化方式":'0-1',
    "触觉信号的归一化方式":'0-1',
    "融合方式：": "transformer"
      })




output_folder = "output_haptic_files_gan"  # 保存生成的触觉信号，保存的是txt文件

os.makedirs(output_folder, exist_ok=True)

folder_name_save_images = 'saved_images_train_gan' # 保存训练过程中生成的触觉信号并画图；
os.makedirs(folder_name_save_images, exist_ok=True)


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

def evaluate_model(enconder_I,  encoder_A,   Fusion_I_A, gen_haptic, dataloader):
    encoder_A.eval()
    enconder_I.eval()
    Fusion_I_A.eval()

    gen_haptic.eval()
    correct = 0
    total = 0
    haptic_loss_mse_total = []

    haptic_loss_rmse_total = []
    haptic_loss_sim_total = []
    best_mse =1
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

            haptic_loss_mse = mse_loss(haptic_gen, haptic_real)
            haptic_loss_mse_total.append(haptic_loss_mse)

            #print('haptic_loss_mse:', haptic_loss_mse)
            #print('haptic_loss_mse_total:', haptic_loss_mse_total)


            '''求损失'''
            haptics_loss_rmse = RMSE_loss(haptic_gen, haptic_real)
            #print('haptics_loss_rmse:', haptics_loss_rmse)
            haptic_loss_sim = SIM(haptic_gen, haptic_real)

            #print('haptic_loss_sim:', haptic_loss_sim)

            haptic_loss_rmse_total.extend(haptics_loss_rmse)
            haptic_loss_sim_total.extend(haptic_loss_sim)


                #print('一共计算了{}个MSE,{}个RMSE,{}个SIM'.format(len(haptic_loss_mse_total), len(haptic_loss_rmse_total), len(haptic_loss_sim_total)))
                # print('-------------RMSE-----------------------')
                # print('到目前的所有批次的平均rmse_loss：{},一共{}个batch，损失的最小值是{},最大值是{}'.format(
                #     sum(haptic_loss_rmse_total) / len(haptic_loss_rmse_total), len(haptic_loss_rmse_total)/batch_size, min(haptic_loss_rmse_total), max(haptic_loss_rmse_total)))
                # print('----------------SIM-------------------'
                # )
                # print('到目前的所有批次的平均sim_loss：{},一共{}个batch，损失的最小值是{},最大值是{}'.format(
                #     sum(haptic_loss_sim_total) / len(haptic_loss_sim_total), len(haptic_loss_sim_total) / batch_size,
                #     min(haptic_loss_sim_total), max(haptic_loss_sim_total)))

            mse_average = sum(haptic_loss_mse_total) / len(haptic_loss_mse_total)

            if mse_average < best_mse:
                best_mse = mse_average

                '''保存生成的触觉信号为列表,并把他们保存到txt文件中'''
                # fake_haptic_list = fake_haptic.tolist()
                file_name = filepath  # 原始的文件名字
                modified_filename = []  # 把生成的测试集合的数据都重新命名然后保存
                # 遍历元组中的字符串

                for item in file_name:
                    # 在这个示例中，我们将每个字符串转为大写并添加 " - Modified" 后缀
                    if item.split('/')[-1].split('_')[-1][1] == 'r':
                        modified_string = "DFT321" + item.split('/')[-1].split("Image")[
                            0] + f"Movement_train{item.split('Image_')[1].split('_')[0]}_gen_fake.txt"

                    else:
                        item.split('/')[-1].split('_')[-1][1] == 'e'
                        modified_string = "DFT321" + item.split('/')[-1].split("Image")[
                            0] + f"Movement_test{item.split('Image_')[1].split('_')[0]}_gen_fake.txt"

                    modified_filename.append(modified_string)

                for i in range(opt.batch_size):
                    row = haptic_gen[i].tolist()  # 将一行转换为Python列表
                    row_str = "\n".join(map(str, row))  # 将列表元素转换为字符串，并用空格隔开
                    file_name_gen = os.path.join(output_folder, modified_filename[i])
                    with open(file_name_gen, "w") as file:
                        file.write(row_str)

                '''保存原始的触觉信号'''
                # 这两个都是按照
                modified_filename_real_haptic = []

                for item in file_name:
                    if item.split('/')[-1].split('_')[-1][1] == 'r':
                        modified_string_real = "DFT321" + item.split('/')[-1].split("Image")[
                            0] + f"Movement_train{item.split('Image_')[1].split('_')[0]}.txt"
                    else:
                        item.split('/')[-1].split('_')[-1][1] == 'e'
                        modified_string_real = "DFT321" + item.split('/')[-1].split("Image")[
                            0] + f"Movement_test{item.split('Image_')[1].split('_')[0]}.txt"

                    modified_filename_real_haptic.append(modified_string_real)

                for i in range(opt.batch_size):
                    row_real = haptic_real[i].tolist()  # 将一行转换为Python列表
                    row_real_str = "\n".join(map(str, row_real))  # 将列表元素转换为字符串，并用空格隔开

                    file_name_real = os.path.join(output_folder,
                                                  modified_filename_real_haptic[i])  # 使用原始的文件名列表中的元素作为文件名

                    with open(file_name_real, "w") as file:
                        file.write(row_real_str)



                '''如果当前是最优的,则保存为最优的值'''

                best_Encoder_Image_state = Encoder_Image.state_dict()
                best_Encoder_Audio_state = Encoder_Audio.state_dict()
                best_Fusion_I_A_state = Fusion_I_A.state_dict()
                best_gen_haptic_state = gen_haptic.state_dict()

                torch.save(best_Encoder_Image_state, "Encoder_Image.pth")
                torch.save(best_Encoder_Audio_state, "Encoder_Audio.pth")
                torch.save(best_Fusion_I_A_state, "Fusion_I_A.pth")
                torch.save(best_gen_haptic_state, "gen_haptic.pth")


        mse_average = sum(haptic_loss_mse_total)/len(haptic_loss_mse_total)

        rmse_average = sum(haptic_loss_rmse_total)/len(haptic_loss_rmse_total)
        sim_average = sum(haptic_loss_sim_total) / len(haptic_loss_sim_total)



        '''重新设置为训练模式'''
        encoder_A.train()
        enconder_I.train()
        Fusion_I_A.train()
        gen_haptic.train()


    return  mse_average, rmse_average, sim_average





'''定义不同的网络'''

print('device:', device)

# Encoder_Image = Encoder_Image_VGG().to(device)
# Encoder_Audio = Encoder_Audio_VGG().to(device)

Encoder_Image = Encoder_Image_Resnet().to(device)
Encoder_Audio = Encoder_Audio_Resnet().to(device)



#encoder_fenlei = haptic_encoder_fenlei(512, 128).to(device)  # 这是两层全连接的参数表示,网络输出的第二个用来计算全连接的损失

gen_haptic = haptic_Generator(512).to(device)

dis_haptic = haptic_dis_gan(1600).to(device)

#新的基于transformer的融合方式
Fusion_I_A = TransformerLayer_fusion(512, 8).to(device)



# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失函数,GAN网络的损失

mse_loss = torch.nn.MSELoss()  # 生成器的损失,也是自编码器的损失

criterion_fenlei = nn.CrossEntropyLoss()   # 编码网路最后两层的损失，即分类损失,交叉熵损失

#  前面的编码网络先用分类损失去优化;
# optimizer_encoder_I = torch.optim.Adam(Encoder_Image.parameters(), lr=0.01)
#
# optimizer_encoder_A = torch.optim.Adam(Encoder_Audio.parameters(), lr=0.01)

#optimizer_encoder_fenlei = torch.optim.Adam(encoder_fenlei.parameters(), lr=0.0002)  # 优化后两层，融合网络之后的分类器

#optimizer_fusion = torch.optim.Adam(Fusion_I_A.parameters(), lr=0.0002)

'''生成器和鉴别器的损失'''
optimizer_gen_fusion_haptic = torch.optim.Adam(list(gen_haptic.parameters())+list(Fusion_I_A.parameters())+list(Encoder_Image.parameters())+list(Encoder_Audio.parameters()), lr=0.0002, betas=(0.5, 0.999))  # 优化融合网络和生成网络

optimizer_D = torch.optim.Adam(dis_haptic.parameters(), lr=0.0002, betas=(0.5, 0.999))

# optimizer_gen_fusion_haptic = torch.optim.RMSprop(list(gen_haptic.parameters())+list(Fusion_I_A.parameters())+list(Encoder_Image.parameters())+list(Encoder_Audio.parameters()), lr=0.0005)  # 优化融合网络和生成网络
#
# optimizer_D = torch.optim.RMSprop(dis_haptic.parameters(), lr=0.0005)


best_val_haptic_mse = 1


batches_done = 0
''' 开始训练所有网络'''
for i in range(opt.n_epochs):

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
        optimizer_D.zero_grad()  # 清除网络中所有参数的梯度,在反向传播之前.先将梯度清0.
        haptic_real = haptics
        I_encoder = Encoder_Image(images)
        A_encoder = Encoder_Audio(audios)

        constra_loss = contrastive_loss(I_encoder, A_encoder, label_batch_target, opt.temp_contra_loss)
        print('当前批次的constra_loss:', constra_loss )

        #print('当前批次的对比损失：', constra_loss)

        fusion_I_A = Fusion_I_A(I_encoder, A_encoder)
        haptic_gen = gen_haptic(fusion_I_A)
        # 计算鉴别器的损失
        real_labels = torch.ones(opt.batch_size, 1).to(device)
        fake_labels = torch.zeros(opt.batch_size, 1).to(device)

        real_outputs = dis_haptic(haptic_real)
        fake_outputs = dis_haptic(haptic_gen)
        # 计算损失
        d_loss_real = criterion(real_outputs, real_labels)   #这地方有的竟然是real label这是改成了是真是onehot编码的标签；
        d_loss_fake = criterion(fake_outputs, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        #d_loss = -torch.mean(real_outputs) + torch.mean(fake_outputs)

        # 反向传播和优化

        d_loss.backward(retain_graph=True)
        optimizer_D.step()

        # '''clip鉴别器的权重'''
        # for p in dis_haptic.parameters():
        #     p.data.clamp_(-0.01, 0.01)

        '''优化编码网络、融合网络、生成网络'''
        if j % opt.n_critic == 0:

            optimizer_gen_fusion_haptic.zero_grad()

            I_encoder = Encoder_Image(images)
            A_encoder = Encoder_Audio(audios)

            fusion_I_A = Fusion_I_A(I_encoder, A_encoder)
            haptic_gen = gen_haptic(fusion_I_A)

            fake_outputs = dis_haptic(haptic_gen)  # 因为已经更新了，所以需要用这个更新之后的网络的。

            #g_loss_1 = -torch.mean(fake_outputs)  # wgan的损失
            g_loss_1 = criterion(fake_outputs, real_labels) # 普通gan的损失

            g_loss_2 = mse_loss(haptic_gen, haptic_real)
            constra_loss = contrastive_loss(I_encoder, A_encoder, label_batch_target, opt.temp_contra_loss)
            print('g_loss_1:', g_loss_1, 'g_loss_2_mse:', g_loss_2, 'duibisunshi', constra_loss)

            g_loss = g_loss_1 + k_mse * g_loss_2
            totoal_loss = g_loss_1 + k_mse * g_loss_2 + k_duibisunshi * constra_loss

            totoal_loss.backward(retain_graph=True)
            optimizer_gen_fusion_haptic.step()

            rmse_list = RMSE_loss(haptic_gen, haptic_real)
            rmse_list_tensor = torch.tensor(rmse_list)
            rmse_mean = torch.mean(rmse_list_tensor)

            sim_list = SIM(haptic_gen, haptic_real)
            sim_list_tensor = torch.tensor(sim_list)
            sim_mean = torch.mean(sim_list_tensor)

            ST_SIM_LOSS = ST_SIM(haptic_gen, haptic_real)

            print('epoch[{0}/{1}],bathchid[{2}/{3}],D Loss:{4},G Loss:{5}, mse_loss{6}%'.format (i, opt.n_epochs, j, len(train_loader), d_loss.item(), g_loss.item(), g_loss_2.item()))

            print('epoch[{0}/{1}],bathchid[{2}/{3}],metric1 RMSE:{4},metric2 SIM:{5}, metric3 ST_SIM:{6}'.format(i, opt.n_epochs, j,len(train_loader), rmse_mean.item(), sim_mean.item(),ST_SIM_LOSS))

            if batches_done % opt.sample_interval == 0:
                train_save_images(haptic_gen, haptic_real, batches_done,folder_name_save_images)
                print('save images')
            batches_done += opt.n_critic


        #记录相应的指标,12.26暂时先停止记录这个验证集的跑
    #val_haptic_mse, val_haptic_rmse, val_haptic_sim = evaluate_model(Encoder_Image, Encoder_Audio, Fusion_I_A,
    #                                                                 gen_haptic, val_loader)

   # print('epoch:[{3}/{4}],  val_haptic_mse:{0}, rmse{1}, sim{2}'.format(val_haptic_mse, val_haptic_rmse, val_haptic_sim, i, opt.n_epochs))


    wandb.log({ 'train_constr_loss': constra_loss, 'train_gen_mse': g_loss_2, 'train_g_loss_1_yuanshi': g_loss_1, 'train_d_loss': d_loss,
                'train_rmse': rmse_mean.item(),  'train_sim': sim_mean.item()})


    if i%10==0:
        best_Encoder_Image_state = Encoder_Image.state_dict()
        best_Encoder_Audio_state = Encoder_Audio.state_dict()
        best_Fusion_I_A_state = Fusion_I_A.state_dict()
        best_gen_haptic_state = gen_haptic.state_dict()

        torch.save(best_Encoder_Image_state, f"Encoder_Image_gan_resnetlearning.pth")
        torch.save(best_Encoder_Audio_state, f"Encoder_Audio_gan_resnetlearning.pth")
        torch.save(best_Fusion_I_A_state, "Fusion_I_A_gan_resnetlearning.pth")
        torch.save(best_gen_haptic_state, "gen_haptic_gan_resnetlearning.pth")

wandb.finish()
