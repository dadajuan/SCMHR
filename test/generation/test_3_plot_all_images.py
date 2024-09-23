import torch
from torch import nn
import shutil
import os
import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import random
import os
import shutil
from metric import  SIM, RMSE_loss
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(torch.__version__)

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
from metric import RMSE_loss, SIM


#input_folder = 'D:\Desktop\ACM10.7\coding\code_process\output_haptic_files'  # 你保存的所有的触觉的文件的地址，包括生成的和原始的

'''保存并画出所有的触觉图片，新的'''


# 文件夹1--经过test_1之后生成的所有的触觉信号txt文件都保存在这个里面
input_folder =  'D:\Desktop\ACM10.7\coding\code_process\Atest\output_haptic_files_1.2_gan'

# 文件夹2--把生成的images都保存到这个里面
output_folder = 'D:\Desktop\ACM10.7\coding\code_process\Atest\output_haptic_images_files_1.2_gan_timesnewman'


# 文件夹路径
folder_path = input_folder

# 文件夹3--把所有的都存到对应的类别里面
folder_path_fenlei = 'D:\Desktop\ACM10.7\coding\code_process\Atest\output_haptic_images_G1_G9_files_1.2_gan'


# 该函数负责把读取到的两个txt文件，即一个生成的，一个真实的都保存成图片，然后存起来。
def plot_two_txt_files(file1, file2, output_image1, output_image2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = [float(line.strip()) for line in f1.readlines()]
        data2 = [float(line.strip()) for line in f2.readlines()]
        #print('data1', data1)

    #fig = plt.figure(figsize=(20, 5))
    # 绘制数据，可以在plot函数中指定颜色
    plt.plot(data1, label='gen haptic', color='blue')
    plt.xlabel('Time step', fontname = 'Times New Roman')
    plt.ylabel('Amplitude', fontname = 'Times New Roman')
    plt.xlim(0, 1600)
    plt.ylim(0, 1)
    plt.savefig(output_image1, bbox_inches='tight', pad_inches=0.1, dpi=600)
    plt.close()

    plt.plot(data2, label='real haptic', color='red')
    # 添加图例
    plt.xlabel('Time step', fontsize=12, fontname = 'Times New Roman')
    plt.ylabel('Amplitude',fontsize=12,  fontname = 'Times New Roman')
    plt.xlim(0, 1600)
    plt.ylim(0, 1)
    #plt.savefig("generated_signals/train/{}real tac.jpg".format(i), bbox_inches='tight', pad_inches=0.1, dpi=600)
    # 保存图像
    plt.savefig(output_image2, bbox_inches='tight', pad_inches=0.1, dpi=600)
    plt.close()

#  该函数负责把所有的文件都读起来然后存成images
def plot_batch(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    txt_files = [file for file in os.listdir(input_folder) if file.endswith('_gen_fake.txt')]  # 生成的假的图像

    for txt_file in txt_files:
        file1 = os.path.join(input_folder, txt_file)  # 保存所有的生成的触觉信号txt
        file2 = os.path.join(input_folder, txt_file.replace('_gen_fake.txt', '.txt'))  # 保存所有的真实的触觉信号的txt

        output_image_gen_haptic = os.path.join(output_folder, txt_file.replace('.txt', '.png'))   # 生成的图片的保存地址
        output_image_real_haptic = os.path.join(output_folder, txt_file.replace('_gen_fake.txt', '.png'))  # 真实的图片的保存地址

        plot_two_txt_files(file1, file2, output_image_gen_haptic, output_image_real_haptic)





'''保存并画出所有的图'''
plot_batch(input_folder, output_folder)

exit()
'''把生成的所有数据,按照G1-G9保存到对应的问价夹里面去'''

if not os.path.exists(folder_path_fenlei):
    os.makedirs(folder_path_fenlei)


# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        # 获取图片名字中 'DFT321' 后的两个字符
        start_index = filename.find('DFT321') + 6
        end_index = start_index + 2
        category = filename[start_index:end_index]

        # 创建相应的文件夹
        category_folder = os.path.join(folder_path_fenlei, category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)

        # 复制图片到相应的文件夹
        src_path = os.path.join(folder_path, filename)
        dst_path = os.path.join(category_folder, filename)
        shutil.copy(src_path, dst_path)

print("Images copied successfully!")
