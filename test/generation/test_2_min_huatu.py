
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
from metric import  SIM, RMSE_loss
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(torch.__version__)

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
from metric import RMSE_loss, SIM


def mse(tensor1, tensor2):
    # 按行计算MSE
    mse_list = []
    for i in range(tensor1.size(0)):  # 遍历每一行
        mse = torch.mean((tensor1[i] - tensor2[i]) ** 2).item()
        mse_list.append(mse)
    return mse_list
'''
第一步，找到损失最小的值的前几个,并把他们的名字保存下来;
第二步，这些图片保存到制定的文件夹里面；
第三步,把真实的和生成的去画图说明问题；

老实说：效果并不怎么样，感觉mse最小的是也并不是很好样子看起来，12.23
'''


def copya2b_ifinllist(destination_folder, source_folder, file_list):
    '''把在filelist列表里的名称的文件，从原文件夹移到目标文件夹'''
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for file_name in file_list:
        # 检查文件是否存在于源文件夹中
        source_path = os.path.join(source_folder, file_name)
        if os.path.exists(source_path):
            # 复制文件到目标文件夹
            destination_path = os.path.join(destination_folder, file_name)
            shutil.copyfile(source_path, destination_path)
            #print(f"文件 '{file_name}' 已成功复制到文件夹 '{destination_folder}' 中。")
        else:
            print(f"文件 '{file_name}' 不存在于文件夹 '{source_folder}' 中。")



input_folder = 'D:\Desktop\ACM10.7\coding\code_process\output_haptic_files'  # 你保存的所有的触觉的文件的地址，包括生成的和原始的
input_folder =  'D:\Desktop\ACM10.7\coding\code_process\1test\output_haptic_files_1.2_gan'
#output_folder = 'D:\Desktop\ACM10.7\coding\code_process\output_haptic_files_images'  # 输出文件夹的路径

# 找到其中SIM最大的一些图像；
max_sim_folder = 'D:\Desktop\ACM10.7\coding\code_process\max_sim_haptic_files'
max_sim_floder_images = 'D:\Desktop\ACM10.7\coding\code_process\max_sim_haptic_images'

#找到其中RMSE最小的一些图像；
min_rmse_folder = 'D:\Desktop\ACM10.7\coding\code_process\min_rmse_haptic_files'
min_rmse_folder_images = 'D:\Desktop\ACM10.7\coding\code_process\min_rmse_haptic_files_images'


#找到其中MSE最小的一些图像；
min_mse_folder = 'D:\Desktop\ACM10.7\coding\code_process\min_mse_haptic_files'
min_mse_folder_images = 'D:\Desktop\ACM10.7\coding\code_process\min_mse_haptic_files_images'



gen_haptic = [ ]
raw_haptic = [ ]
# 寻找匹配的txt文件
txt_files = [file for file in os.listdir(input_folder) if file.endswith('_gen_fake.txt')]

for txt_file in txt_files:
    file1 = os.path.join(input_folder, txt_file)
    file2 = os.path.join(input_folder, txt_file.replace('_gen_fake.txt', '.txt'))
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = [float(line.strip()) for line in f1.readlines()]  # 读取生成的触觉信号的值
        data2 = [float(line.strip()) for line in f2.readlines()]  # 读取原始的触觉信号的值
        gen_haptic.append(data1)
        raw_haptic.append(data2)


'''把他们转化为（batch, 1600)维的张量'''
gen_haptic_tensor = torch.tensor(gen_haptic)
raw_haptic_tensor = torch.tensor(raw_haptic)

'''分别计算上述三个损失，并找到对应的最大值和最小值，以及其相应的索引'''
sim_loss = SIM(gen_haptic_tensor, raw_haptic_tensor)
rmse_loss = RMSE_loss(gen_haptic_tensor, raw_haptic_tensor)

mse_loss = mse(gen_haptic_tensor, raw_haptic_tensor)



max_indexes_sim = sorted(range(len(sim_loss)), key=lambda i: sim_loss[i], reverse=True)[:10]

min_indexes_rmse = sorted(range(len(rmse_loss)), key=lambda i: rmse_loss[i])[:10]


min_indexes_mse = sorted(range(len(mse_loss)), key=lambda i: mse_loss[i])[:10]


max_sim = sorted(sim_loss, reverse=True)[:10]
min_rsme = sorted(rmse_loss)[:10]
min_mse = sorted(mse_loss)[:10]
print('最大的SIM的：', max_sim)
print('最小的rmse的：', min_rsme)
print('最小的mse的：', min_mse)

'''
第二步,都保存到一个问价夹里面
'''

'''把最大的sim的触觉信号都整到一个新的文件夹里面去'''
txt_files_max_sim = [txt_files[i] for i in max_indexes_sim]
modified_a = [elem.replace('_gen_fake.txt', '.txt') for elem in txt_files_max_sim ]
txt_files_max_sim_final = txt_files_max_sim  + modified_a



'''把最小的rmse的触觉信号都整到一个新的文件夹里面去'''
txt_files_min_rmse = [txt_files[i] for i in min_indexes_rmse]
modified_b = [elem.replace('_gen_fake.txt', '.txt') for elem in txt_files_min_rmse]
txt_files_min_rmse_final = txt_files_min_rmse + modified_b


'''把最小的mse的触觉信号都整到一个新的文件夹里面去'''
txt_files_min_mse = [txt_files[i] for i in min_indexes_mse]
modified_c = [elem.replace('_gen_fake.txt', '.txt') for elem in txt_files_min_mse]
txt_files_min_mse_final = txt_files_min_mse + modified_c


copya2b_ifinllist(max_sim_folder, input_folder, txt_files_max_sim_final)  # 把他们都保存到了一个新的文件夹了里面已经
copya2b_ifinllist(min_rmse_folder, input_folder, txt_files_min_rmse_final)  #把他们都保存到了一个新的文件夹了
copya2b_ifinllist(min_mse_folder, input_folder, txt_files_min_mse_final)

'''
第三步， 画图
'''

# 把两个序列信号画到一个图形里面去;
def plot_two_txt_files(file1, file2, output_image):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = [float(line.strip()) for line in f1.readlines()]
        data2 = [float(line.strip()) for line in f2.readlines()]
        #print('data1', data1)
    fig = plt.figure(figsize=(20, 5))
    # 绘制数据，可以在plot函数中指定颜色
    plt.plot(data1, label='File 1_real haptic', color='blue')
    plt.plot(data2, label='File 2_fake haptic', color='red')

    # 添加图例
    plt.legend()

    # 保存图像
    plt.savefig(output_image)
    plt.close()


# 指定包含所有txt文件的子文件夹


def plot_batch(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    txt_files = [file for file in os.listdir(input_folder) if file.endswith('_gen_fake.txt')]

    for txt_file in txt_files:
        file1 = os.path.join(input_folder, txt_file)
        file2 = os.path.join(input_folder, txt_file.replace('_gen_fake.txt', '.txt'))

        output_image = os.path.join(output_folder, txt_file.replace('.txt', '.png'))
        plot_two_txt_files(file1, file2, output_image)

plot_batch(max_sim_folder, max_sim_floder_images)
plot_batch(min_rmse_folder, min_rmse_folder_images)

plot_batch(min_mse_folder, min_mse_folder_images)




