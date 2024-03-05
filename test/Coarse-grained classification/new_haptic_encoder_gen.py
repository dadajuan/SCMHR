import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision
from torch.nn import init

import torchvision.models as models


'''初始化网络参数'''

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

''''编码网络'''


class Encoder_Image_VGG(nn.Module):
    def __init__(self, channel_in=3, z_size=512):
        super(Encoder_Image_VGG, self).__init__()
        self.size = channel_in
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        self.l_out = nn.Linear(in_features=1000, out_features=z_size)
        self.l_var = nn.Linear(in_features=z_size, out_features=9)

    def forward(self, ten):
        ten = self.vgg16(ten)
        #print('sss', ten.shape)
        ten = ten.view(len(ten), -1)
        out = self.l_out(ten)
        #print('sss', out.shape)
        out_logit = self.l_var(out)
        #out_logit = nn.Softmax(out_logit)
        return out, out_logit


class Encoder_Audio_VGG(nn.Module):
    def __init__(self, channel_in=3, z_size=512):
        super(Encoder_Audio_VGG, self).__init__()
        self.size = channel_in
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        self.l_out = nn.Linear(in_features=1000, out_features=z_size)
        self.l_var = nn.Linear(in_features=z_size, out_features=9)

    def forward(self, ten):
        ten = self.vgg16(ten)
        out = self.l_out(ten)
        out_logit = self.l_var(out)
        #out_logit = nn.Softmax(out_logit)
        return out, out_logit



class Encoder_Image_Resnet(torch.nn.Module):
    def __init__(self, output_dim=512):
        super(Encoder_Image_Resnet, self).__init__()
        # 加载预训练的 ResNet-18 模型
        resnet = models.resnet18(pretrained=True)

        # 冻结 ResNet-18 的参数
        for param in resnet.parameters():
            param.requires_grad = True  # 此处为了提高分类的性能,把这个resnet改成了可以训练的

        # 提取 ResNet-18 的特征提取部分（卷积层）
        self.features = torch.nn.Sequential(*list(resnet.children())[:-1])

        # 添加新的全连接层
        self.fc = torch.nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        # 通过 ResNet-18 提取特征
        x = self.features(x)
        x = torch.flatten(x, 1)

        # 使用新的全连接层
        x = self.fc(x)
        return x

class Encoder_Audio_Resnet(torch.nn.Module):
    def __init__(self, output_dim=512):
        super(Encoder_Audio_Resnet, self).__init__()
        # 加载预训练的 ResNet-18 模型
        resnet = models.resnet18(pretrained=True)

        # 冻结 ResNet-18 的参数
        for param in resnet.parameters():
            param.requires_grad = True # 此处为了提高分类的性能,把这个resnet改成了可以训练的

        # 提取 ResNet-18 的特征提取部分（卷积层）
        self.features = torch.nn.Sequential(*list(resnet.children())[:-1])

        # 添加新的全连接层
        self.fc = torch.nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        # 通过 ResNet-18 提取特征
        x = self.features(x)
        x = torch.flatten(x, 1)
        # 使用新的全连接层
        x = self.fc(x)
        return x


class haptic_encoder(nn.Module):

    def __init__(self, input_size, output_size1):
        super(haptic_encoder, self).__init__()
        self.densenet1 = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024, momentum=0.9),
            nn.ReLU())

        self.densenet2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512, momentum=0.9),
            nn.ReLU())

        self.densenet3 = nn.Sequential(
            nn.Linear(512, output_size1),
            nn.BatchNorm1d(output_size1, momentum=0.9),
            nn.ReLU())
    def forward(self, x):
        x1 = self.densenet1(x)
        x2 = self.densenet2(x1)
        x3 = self.densenet3(x2)  # 用于重建所需要的参数量

        return x3




class haptic_encoder_fenlei(nn.Module):

    def __init__(self, input_size1):
        super(haptic_encoder_fenlei, self).__init__()

        self.densenet4 = nn.Linear(input_size1, 128)
        self.relu1 = nn.ReLU()
        self.densenet5 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.densenet4(x)
        x = self.relu1(x)
        x1 = self.densenet5(x)  # 分类网络这里不需要使用softmax函数,

        return x1



class VAE_GAN(nn.Module):
    def __init__(self, encoder, generator):
        super(VAE_GAN, self).__init__()
        self.encoder = encoder
        self.generator = generator

    def forward(self, x):
        z = self.encoder(x)
        reconstructed_x = self.generator(z)

        return reconstructed_x, z

'''触觉信号的鉴别器'''

class haptic_dis_wgan(nn.Module):
    def __init__(self, input_size):
        super(haptic_dis_wgan, self).__init__()

        self.densenet1 = nn.Sequential(
            nn.Linear(input_size, 1024),
            #nn.BatchNorm1d(1024, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True))

        self.densenet2 = nn.Sequential(
            nn.Linear(1024, 512),
            #nn.BatchNorm1d(512, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True))

        self.densenet4 = nn.Sequential(
            nn.Linear(512, 1))
            #nn.BatchNorm1d(1, momentum=0.9),
            #nn.Sigmoid())
        print('---------初始化鉴别器-------')
        init_weights(self.densenet1, init_type='normal')
        init_weights(self.densenet2, init_type='normal')
        init_weights(self.densenet4, init_type='normal')

    def forward(self, x):
        x1 = self.densenet1(x)
        x2 = self.densenet2(x1)
        x4 = self.densenet4(x2)
        return x4

class haptic_dis_gan(nn.Module):
    def __init__(self, input_size):
        super(haptic_dis_gan, self).__init__()

        self.densenet1 = nn.Sequential(
            nn.Linear(input_size, 1024),
            #nn.BatchNorm1d(1024, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True))

        self.densenet2 = nn.Sequential(
            nn.Linear(1024, 512),
            #nn.BatchNorm1d(512, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True))

        self.densenet4 = nn.Sequential(
            nn.Linear(512, 1),
            #nn.BatchNorm1d(1, momentum=0.9),
            nn.Sigmoid())
        print('---------初始化鉴别器-------')
        init_weights(self.densenet1, init_type='normal')
        init_weights(self.densenet2, init_type='normal')
        init_weights(self.densenet4, init_type='normal')

    def forward(self, x):
        x1 = self.densenet1(x)
        x2 = self.densenet2(x1)
        x4 = self.densenet4(x2)
        return x4

'''触觉信号的生成器'''

class haptic_Generator(nn.Module):
    def __init__(self, input_size):
        super(haptic_Generator, self).__init__()

        self.densenet1 = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024, momentum=0.9),
            nn.ReLU())

        self.densenet2 = nn.Sequential(
            nn.Linear(1024, 1600),
            nn.BatchNorm1d(1600, momentum=0.9),
            nn.Sigmoid())
        print('--------初始化生成器-------')
        init_weights(self.densenet1, init_type='normal')
        init_weights(self.densenet2, init_type='normal')
    def forward(self, x):

        x1 = self.densenet1(x)
        x2 = self.densenet2(x1)
        # 输出维度: [10, 1600]
        return x2

class cross_modal_fusion(nn.Module):

    def __init__(self, dimension_I):
        super(cross_modal_fusion, self).__init__()

        self.densenet1 = nn.Linear(dimension_I*2, dimension_I)
        self.relu1 = nn.ReLU()

    def forward(self, image_encoder, audio_encoder):
        fusion_z = torch.cat([image_encoder, audio_encoder], dim=1)
        fusion_z = self.densenet1(fusion_z)
        fusion_z = self.relu1(fusion_z)

        return fusion_z

class TransformerLayer_fusion(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(TransformerLayer_fusion, self).__init__()

        self.attention_V = nn.MultiheadAttention(hidden_size, num_heads)
        self.attention_A = nn.MultiheadAttention(hidden_size, num_heads)

        self.fc_V = nn.Linear(hidden_size, hidden_size)
        self.fc_A = nn.Linear(hidden_size, hidden_size)

        self.norm_v_1 = nn.LayerNorm(hidden_size)  # 是对每单个batch的归一化，
        self.norm_v_2 = nn.LayerNorm(hidden_size)

        self.norm_a_1 = nn.LayerNorm(hidden_size)
        self.norm_a_2 = nn.LayerNorm(hidden_size)

        self.MLP_V =nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.MLP_A = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        #self.conv1x1 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1)

    def forward(self, fv, fa):

        residual_v_1 = fv
        fv_normal = self.norm_v_1(fv)
        fa_normal = self.norm_a_1(fa)
        fv_iter, _ = self.attention_V(fa_normal, fv_normal, fv_normal)

        residual_v_2 = fv_iter + residual_v_1

        fv_output = self.MLP_V(self.norm_v_2(residual_v_2)) + residual_v_2


        residual_a_1 = fa
        fa_iter, _ = self.attention_A(fv_normal, fa_normal, fa_normal)

        residual_a_2 = fa_iter + residual_a_1

        fa_output = self.MLP_A(self.norm_a_2(residual_a_2)) + residual_v_2
        mm = fa_output + fv_output
        #mm = mm.unsqueeze(1)
        #f_v_a = self.conv1x1(mm.permute(0, 2, 1))
        return mm






# 创建生成网络实例









# # 创建自定义编码器实例
# encoder = Encoder_Audio_Resnet(output_dim=4096)
#
#
# # 打印自定义编码器结构
# print(encoder)
# tmp = torch.randn(64, 3, 334, 217)
#
# mm = encoder(tmp)
# print(mm.shape)
#exit()

def main():
    tmp = torch.randn(64, 512)
    dis = haptic_dis(512)

    ACC = dis(tmp)
    print(tmp)
    print(ACC.shape)

    exit()


    tmp = torch.randn(10, 3, 256, 224)

    encoder_I = Encoder_Image_VGG()

    encoder_A = Encoder_Audio_VGG()

    fusion = cross_modal_fusion(512)

    I_encoder, _ = encoder_I(tmp)

    print(len(I_encoder))

    A_encoder, _ = encoder_A(tmp)

    # I_encoder = torch.from_numpy(I_encoder)
    # A_encoder = torch.from_numpy(A_encoder)

    fusion_z = fusion(I_encoder, A_encoder)

    print(fusion_z)

# if __name__ == "__main__":
#     main()


