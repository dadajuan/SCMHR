from PIL import Image
import torch

import torch
print(torch.__version__)

from torch.utils.data import Dataset
import cv2
from torch.autograd import Variable

from utils import read_split_data, read_split_data_audio, plot_data_loader_image, read_split_data_tac_densenet, read_split_data_haptic


class MyDataSet_V_A_H(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, audio_path: list, haptic_path: list,  images_class: list, transform=None):
        self.images_path = images_path
        self.audio_path = audio_path
        self.haptic_path = haptic_path
        self.images_class = images_class
        self.transform = transform

    #该数据集下所有样本的个数
    def __len__(self):
        return len(self.images_path) #即为类别数

    def __getitem__(self, item):
        #查看高分、触觉、低分是否匹配，读取之后的数据集的前面的名字和后面的名字是不是一致的都；
        # if self.images_path[item].split("\\")[-1] != self.audio_path[item].split("\\")[-1]:
        #     print("!!!!存在不匹配!!!!")
        #
        # if self.images_path[item].split("\\")[-1].split("_")[0] != self.haptic_path[item].split("\\")[-1].split("_")[0] or \
        #         self.images_path[item].split("\\")[-1].split("Image_")[1].split("_")[0] != self.haptic_path[item].split("\\")[-1].split("Z_")[1].split("n")[1].split(".")[0]:
        #     print("!!!!存在不匹配!!!!")
        """完成与图像对应的听觉和触觉信号的查找"""

        #print('sss')
        #print('item:', item)
        #print('aaa:', self.images_path[item])

        imgfile = self.images_path[item].split("/")[-1]  #这个是那个图像的名字,如果是服务器，需要\\换成//

        #print(self.images_path[item]) # 该图片的地址
        #print('sss:', self.images_path[item].split('/')[-1])
        g = self.images_path[item].split('/')[-2]

        #print('g:', g)  #  属于哪一类
        #print(self.images_path[item].split('\\')[-1].split('_')[-1][1])
        if self.images_path[item].split('/')[-1].split('_')[-1][1] == 'r':

           # print('train')
            aduio_file = self.images_path[item].split("Visual")[0] + "Audio" + f"/{g}/" + imgfile.split("Image")[0] + f"Sound_Movement_train{imgfile.split('Image_')[1].split('_')[0]}.jpg"
            # print('train_audio_file', aduio_file)

            haptic_file = self.images_path[item].split("Visual")[0] + "Haptic" + f"/{g}/" + "DFT321" + imgfile.split("Image")[0] + f"Movement_train{imgfile.split('Image_')[1].split('_')[0]}.txt"

            # print('train_haptic_file', haptic_file)

        if self.images_path[item].split('//')[-1].split('_')[-1][1] == 'e':
           # print('test')
            aduio_file = self.images_path[item].split("Visual")[0] + "Audio" + f"/{g}/" + imgfile.split("Image")[0] + f"Sound_Movement_test{imgfile.split('Image_')[1].split('_')[0]}.jpg"

            haptic_file = self.images_path[item].split("Visual")[0] + "Haptic" + f"/{g}/" + "DFT321" +  imgfile.split("Image")[0] + f"Movement_test{imgfile.split('Image_')[1].split('_')[0]}.txt"

            # print('test_haptic_file', haptic_file)
            # print('test_audio_file', aduio_file)
        #print(type(self.images_path[item].split('\\')[-1].split('_')[-1]))

        #lrfile = self.images_path[item].split("Training")[0] + "Training_LR" + self.images_path[item].split("Training")[1]




        if item/12==0: #偶尔检验一下，是不是对齐了
            print('image_file:', imgfile)
            print('aduio_file:', aduio_file)
            print('haptic_file:', haptic_file)


        img = cv2.imread(self.images_path[item])
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        audio = cv2.imread(aduio_file)
        audio = audio.transpose(2, 0, 1)
        audio = torch.from_numpy(audio)

        # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))

        label = self.images_class[item]

        '''当触觉信号使用的stft变换之后的图像时候'''

        # img_haptic = cv2.imread(self.haptic_path[item])
        # img_haptic = img_haptic.transpose(2, 0, 1)
        # haptic = torch.from_numpy(img_haptic)


        '''当触觉信号是使用512/1600维的序列的时候的数据'''

        haptic = open(haptic_file).readlines()
        haptic = list(map(float, haptic)) #把list里面的每个str类型的元素转化成float，然后再拼接成list
        haptic = torch.Tensor(haptic)

        if self.transform is not None:
            img = self.transform(img)
        if self.transform is not None:
            audio = self.transform(audio)
        if self.transform is not None:
            haptic = self.transform(haptic)
        file = self.images_path[item].split("\\")[-1]
        #print(file)

        return img, audio, haptic,  label, file

    @staticmethod
    #静态方法
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        #print((batch))
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, audio_, haptics, labels, filepath = tuple(zip(*batch))

        images = torch.stack(images, dim=0)  #image变成（8,3,224,224）
        audio = torch.stack(audio_, dim=0)
        haptics = torch.stack(haptics, dim=0)
        labels = torch.as_tensor(labels)     #labels转化为tensor
        return images, audio, haptics, labels, filepath


# train_images_path = 'D:\Desktop\ACM10.7\coding\source_data_train_test\Visual'
#
# train_audio_path= 'D:\Desktop\ACM10.7\coding\source_data_train_test\Audio'
#
# train_tactile_path = 'D:\Desktop\ACM10.7\coding\source_data_train_test\Haptic'

#
# train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(train_images_path)
# print(train_images_path)
# print(len(train_images_path))
#
#
# train_images_lr_path, train_images_lr_label, val_images_lr_path, val_images_lr_label = read_split_data_audio(train_audio_path)
# print(len(train_images_lr_path))
#
# train_tactile_path, train_tactile_label, val_tactile_path, val_tactile_label = read_split_data_haptic(train_tactile_path)
#
# #
# train_data_set = MyDataSet_V_A_H(images_path=train_images_path,
#                                audio_path=train_audio_path,
#                                haptic_path=train_tactile_path,
#                                images_class=train_images_label,
#                                transform=None)
#
# train_loader = torch.utils.data.DataLoader(train_data_set,
#                                            batch_size=10,
#                                            shuffle=True,
#                                            num_workers=0,
#                                            collate_fn=train_data_set.collate_fn)
# n_epochs = 10
#
# for i in range(n_epochs):
#     for j, (image_batch, aduio_batch, haptic_batch, label_batch, filepath) in enumerate(train_loader):
#         # train()
#
#         label_batch_target = Variable(label_batch, requires_grad=False).float() # 标签
#
#         images = Variable(image_batch, requires_grad=False).float()
#
#         audios = Variable(aduio_batch, requires_grad=False).float()
#
#         haptics = Variable(haptic_batch, requires_grad=False).float()
#
#         print(label_batch_target.shape)
#         print(images.shape)
#         print(audios.shape)
#         print(haptics.shape)

