import scipy
import scipy.io
import os
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score, accuracy_score
from torch.utils.data import Dataset
import h5py
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
from torchvision.transforms.functional import rotate
import random
import pandas as pd

#转为BT亮温
def func_CAL_BT(radiance, wavelength):
    const1 = (1.19107*10**8) / (wavelength**5.0)
    const2 = (1.43883*10**4) / wavelength
    BT = const2 / np.log(1 + const1 / radiance)
    return BT


#亮温的组合——Day and Night
#COT训练集读取
def file_reading(filename):
    try:
        with h5py.File(filename, 'r') as f:
            if 'emis_29_1d' in f.keys() and 'emis_31_1d' in f.keys() and 'emis_32_1d' in f.keys():
                data1 = f['emis_29_1d'][()]
                data2 = f['emis_31_1d'][()]
                data3 = f['emis_32_1d'][()]
                cot = f['COT_retrieved'][()]


                data1 = func_CAL_BT(data1, 8.7)
                data2 = func_CAL_BT(data2, 10.8)
                data3 = func_CAL_BT(data3, 12.0)

                data = np.zeros((3, 128, 128))
                data[0, :, :] = data3-data2  #Red
                data[1, :, :] = data2-data1  #Green
                data[2, :, :] = data2        #Blue

                gama_R = 1
                gama_G = 2
                gama_B = 1
                R = data[0] #Red
                G = data[1] #Green
                B = data[2] #Blue

                #归一化 y = (x - xmin) / (xmax - xmin)
                L_bond = np.min(R)
                H_bond = np.max(R)
                R = np.clip(R, L_bond, H_bond)
                R = (R - L_bond) / (H_bond - L_bond)
                R = np.power(R, 1 / gama_R)
                

                L_bond = np.min(G)
                H_bond = np.max(G)
                G = np.clip(G, L_bond, H_bond)
                G = (G - L_bond) / (H_bond - L_bond)
                G = np.power(G, 1 / gama_G)

                L_bond = np.min(B)
                H_bond = np.max(B)
                B = np.clip(B, L_bond, H_bond)
                B = (B - L_bond) / (H_bond - L_bond)
                B = np.power(B, 1 / gama_B)

                #归一化cot  y = (x - xmin) / (xmax - xmin)
                L_bond = np.min(cot)
                H_bond = np.max(cot)
                cot = (cot - L_bond) / (H_bond - L_bond)

                weirgb = np.zeros((4, 128, 128))

                weirgb[0, :, :] = R  #Red
                weirgb[1, :, :] = G  #Green
                weirgb[2, :, :] = B  #Blue
                weirgb[3, :, :] = cot

                return weirgb
    except OSError as e:
        print(f"{filename}文件无法打开。")
    return None



def guiyihua(R,gama_R):
    L_bond = np.min(R)
    H_bond = np.max(R)
    R = np.clip(R, L_bond, H_bond)
    R = (R - L_bond) / (H_bond - L_bond)
    R = np.power(R, 1 / gama_R)
    return R




# 测试准确率
def accuracy(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return accuracy_score(target, y_pred)


# 计算f1
def calculate_f1_macro(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return f1_score(target, y_pred, average='macro')


# 计算recall
def calculate_recall_macro(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()
    # tp fn fp
    return recall_score(target, y_pred, average="macro", zero_division=0)




#修改混淆矩阵
class CloudDataset_new(Dataset):
    def __init__(self, data_path, transform=None):
        label_map = {'cat0_Closed_MCC': 1, 'cat1_Clustered_Cu': 4, 'cat2_Disorganized MCC': 3, 'cat3_Open_MCC': 2,
                     'cat4_Solid_stratus': 0, 'cat5_Suppressed_Cu': 5}
        self.transform = transform
        self.data = []
        for folder in os.listdir(data_path):
            folder_path = os.path.join(data_path, folder)
            if not os.path.isdir(folder_path):
                continue
            for file_name in os.listdir(folder_path):
                if not file_name.endswith('.mat'):
                    continue
                file_path = os.path.join(folder_path, file_name)
                mat_data = file_reading(file_path)
                if mat_data is not None:
                    data = torch.from_numpy(mat_data) #将一个NumPy数组（mat_data）转换为PyTorch张量（Tensor）
                    label = label_map[folder]
                    label = torch.tensor(label)
                    self.data.append((data, label))


    def __len__(self):
        return len(self.data)
    
    #当你使用 for 循环或其他迭代方式遍历 DataLoader 对象时，它会逐批次地加载数据，每次加载都会调用 train_dataset 中的 __getitem__ 方法来获取一个数据样本
    def __getitem__(self, idx):  
        data, label = self.data[idx]

        if self.transform:
            random_prob = random.random() #生成0-1之间的浮点数
            if random_prob < 0.5:
                angle = random.uniform(0, 360)  # 生成0到360之间的随机旋转角度
                data = rotate(data, angle)

        return data, label






# 读取夜间128×128小块的数据
def file_reading_128(filename):
    try:
        with h5py.File(filename, 'r') as f:
            if 'emis_29_CB_last' in f.keys() and 'emis_31_CB_last' in f.keys() and 'emis_32_CB_last' in f.keys():
                lon = f['lon_128_last'][()]
                lat = f['lat_128_last'][()]

                lon_center = f['lon_center_last'][()]
                lat_center = f['lat_center_last'][()]

                data1 = f['emis_29_CB_last'][()]
                data2 = f['emis_31_CB_last'][()]
                data3 = f['emis_32_CB_last'][()]

                data1 = func_CAL_BT(data1, 8.7)
                data2 = func_CAL_BT(data2, 10.8)
                data3 = func_CAL_BT(data3, 12.0)

                data = np.zeros((3, 128, 128))
                data[0, :, :] = data3-data2  #Red
                data[1, :, :] = data2-data1  #Green
                data[2, :, :] = data2        #Blue

                gama_R = 1
                gama_G = 1.2
                gama_B = 1
                R = data[0] #Red
                G = data[1] #Green
                B = data[2] #Blue

                #归一化 y = (x - xmin) / (xmax - xmin)
                L_bond = np.min(R)
                H_bond = np.max(R)
                R = np.clip(R, L_bond, H_bond)
                R = (R - L_bond) / (H_bond - L_bond)
                R = np.power(R, 1 / gama_R)
                

                L_bond = np.min(G)
                H_bond = np.max(G)
                G = np.clip(G, L_bond, H_bond)
                G = (G - L_bond) / (H_bond - L_bond)
                G = np.power(G, 1 / gama_G)

                L_bond = np.min(B)
                H_bond = np.max(B)
                B = np.clip(B, L_bond, H_bond)
                B = (B - L_bond) / (H_bond - L_bond)
                B = np.power(B, 1 / gama_B)

                weirgb = np.zeros((3, 128, 128))

                weirgb[0, :, :] = R  #Red
                weirgb[1, :, :] = G  #Green
                weirgb[2, :, :] = B  #Blue

                return weirgb,lon,lat,lon_center,lat_center
    except OSError as e:
        print(f"{filename}文件无法打开。")
    return None



# 读取待判断的数据
class JudgeDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = []
        for file_name in os.listdir(data_path):
            if not file_name.endswith('.mat'):
                print("some files cant be judged")
                continue
            file_path = os.path.join(data_path, file_name)
            mat_data,lon,lat,lon_center,lat_center = file_reading_128(file_path)
            if mat_data is not None:
                data = torch.from_numpy(mat_data) #将一个NumPy数组（mat_data）转换为PyTorch张量（Tensor）
                self.data.append((data, file_name, lon,lat,lon_center,lat_center))
            else:
                print("some files cant be judged")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, file_name, lon,lat,lon_center,lat_center = self.data[idx]
        return data, file_name, lon,lat,lon_center,lat_center



#写文章画散点图用
def file_reading_wenzhang(filename):
    try:
        with h5py.File(filename, 'r') as f:
                data1 = f['emis_29_CB'][()]
                data2 = f['emis_31_CB'][()]
                data3 = f['emis_32_CB'][()]
                cot = f['COT_CB'][()]
                Cloud_Fraction = f['Cloud_Fraction'][()]
                High_Fraction = f['High_Fraction'][()]
                Ice_Fraction = f['Ice_Fraction'][()]
                SenZ_mean = f['SenZ_mean'][()]
                clear_sky = f['clear_sky'][()]
                lat_center = f['lat_center'][()]
                lon_center = f['lon_center'][()]
                lat = f['lat_128'][()]
                lon = f['lon_128'][()]

                num_layers = data1.shape[0]
                all_weirgb = []

                for i in range(num_layers):

                    data11 = func_CAL_BT(data1[i], 8.7)
                    data22 = func_CAL_BT(data2[i], 10.8)
                    data33 = func_CAL_BT(data3[i], 12.0)

                    gama_R = 1
                    gama_G = 2
                    gama_B = 1
                    R = data33-data22 #Red
                    G = data22-data11 #Green
                    B = data22 #Blue

                    #归一化 y = (x - xmin) / (xmax - xmin)
                    L_bond = np.min(R)
                    H_bond = np.max(R)
                    R = np.clip(R, L_bond, H_bond)
                    R = (R - L_bond) / (H_bond - L_bond)
                    R = np.power(R, 1 / gama_R)
                    

                    L_bond = np.min(G)
                    H_bond = np.max(G)
                    G = np.clip(G, L_bond, H_bond)
                    G = (G - L_bond) / (H_bond - L_bond)
                    G = np.power(G, 1 / gama_G)

                    L_bond = np.min(B)
                    H_bond = np.max(B)
                    B = np.clip(B, L_bond, H_bond)
                    B = (B - L_bond) / (H_bond - L_bond)
                    B = np.power(B, 1 / gama_B)

                    #归一化cot  y = (x - xmin) / (xmax - xmin)
                    cot1 = cot[i]
                    L_bond = np.min(cot1)
                    H_bond = np.max(cot1)
                    # 检查是否存在除以零的情况
                    if L_bond == H_bond:
                        cot2 = cot1.copy()  # 如果范围为零，直接复制数据
                    else:
                        cot2 = (cot1 - L_bond) / (H_bond - L_bond)
                    
                    weirgb = np.zeros((4, 128, 128))

                    weirgb[0, :, :] = R  #Red
                    weirgb[1, :, :] = G  #Green
                    weirgb[2, :, :] = B  #Blue
                    weirgb[3, :, :] = cot2

                    all_weirgb.append(weirgb)
                all_weirgb = np.array(all_weirgb)
                return all_weirgb,lat_center.squeeze(),lon_center.squeeze(),Cloud_Fraction.squeeze(),High_Fraction.squeeze(),Ice_Fraction.squeeze(),SenZ_mean.squeeze(),clear_sky.squeeze(),lat.squeeze(),lon.squeeze()
    except OSError as e:
        print(f"{filename}文件无法打开。")
    return None



# 写文章时的散点图例子
class JudgeDataset_COT(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = []
        self.lat_center_list = [] 
        self.lon_center_list = []
        for file_name in os.listdir(data_path):
            if not file_name.endswith('.mat'):
                print("some files cant be judged")
                continue
            file_path = os.path.join(data_path, file_name)
            mat_data,lat_center,lon_center,Cloud_Fraction,High_Fraction,Ice_Fraction,SenZ_mean,clear_sky,lat,lon = file_reading_wenzhang(file_path)
            print(mat_data.shape,lat.shape,lat_center.shape)
            #一层层筛选
            for i in range(Cloud_Fraction.shape[0]):  # 假设第一个维度是层数
                if clear_sky[i] == 1 or Cloud_Fraction[i] <= 0.01 or Ice_Fraction[i] >= 0.1 or High_Fraction[i] > 0.1 or SenZ_mean[i] > 45:
                    continue
                data = torch.from_numpy(mat_data[i]) #将一个NumPy数组（mat_data）转换为PyTorch张量（Tensor）
                lat128 = lat[i]
                lon128 = lon[i]
                self.data.append((data,lat128,lon128))
                self.lat_center_list.append(lat_center[i])
                self.lon_center_list.append(lon_center[i])
            save_path = '/work11/wuyy/yun_IR_zuhe/article_case_result_365/lat_lon_centers.mat'
            scipy.io.savemat(save_path, {'lat_center': np.array(self.lat_center_list), 'lon_center': np.array(self.lon_center_list)})


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data,lat128,lon128 = self.data[idx]
        return data,lat128,lon128



#全年judge
def file_reading_2014(filename):
    try:
        with h5py.File(filename, 'r') as f:
                data1 = f['emis_29_CB'][()]
                data2 = f['emis_31_CB'][()]
                data3 = f['emis_32_CB'][()]
                cot = f['COT_CB'][()]

                num_layers = data1.shape[0]
                all_weirgb = []

                for i in range(num_layers):

                    data11 = func_CAL_BT(data1[i], 8.7)
                    data22 = func_CAL_BT(data2[i], 10.8)
                    data33 = func_CAL_BT(data3[i], 12.0)

                    gama_R = 1
                    gama_G = 2
                    gama_B = 1
                    R = data33-data22 #Red
                    G = data22-data11 #Green
                    B = data22 #Blue

                    #归一化 y = (x - xmin) / (xmax - xmin)
                    L_bond = np.min(R)
                    H_bond = np.max(R)
                    R = np.clip(R, L_bond, H_bond)
                    R = (R - L_bond) / (H_bond - L_bond)
                    R = np.power(R, 1 / gama_R)
                    

                    L_bond = np.min(G)
                    H_bond = np.max(G)
                    G = np.clip(G, L_bond, H_bond)
                    G = (G - L_bond) / (H_bond - L_bond)
                    G = np.power(G, 1 / gama_G)

                    L_bond = np.min(B)
                    H_bond = np.max(B)
                    B = np.clip(B, L_bond, H_bond)
                    B = (B - L_bond) / (H_bond - L_bond)
                    B = np.power(B, 1 / gama_B)

                    #归一化cot  y = (x - xmin) / (xmax - xmin)
                    cot1 = cot[i]
                    L_bond = np.min(cot1)
                    H_bond = np.max(cot1)
                    # 检查是否存在除以零的情况
                    if L_bond == H_bond:
                        cot2 = cot1.copy()  # 如果范围为零，直接复制数据
                    else:
                        cot2 = (cot1 - L_bond) / (H_bond - L_bond)
                    
                    weirgb = np.zeros((4, 128, 128))

                    weirgb[0, :, :] = R  #Red
                    weirgb[1, :, :] = G  #Green
                    weirgb[2, :, :] = B  #Blue
                    weirgb[3, :, :] = cot2

                    all_weirgb.append(weirgb)
                all_weirgb = np.array(all_weirgb)
                return all_weirgb,num_layers
    except OSError as e:
        print(f"{filename}文件无法打开。")
    return None


#夜间全年不筛选的识别
class JudgeDataset_2014(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = []
        for file_name in os.listdir(data_path):
            if not file_name.endswith('.mat'):
                print("some files cant be judged")
                continue
            file_path = os.path.join(data_path, file_name)
            mat_data,num_layers = file_reading_2014(file_path)
            print(mat_data.shape,num_layers)
            if mat_data is not None:
                data = torch.from_numpy(mat_data) #将一个NumPy数组（mat_data）转换为PyTorch张量（Tensor）
                self.data.extend((data))
            else:
                print("some files cant be judged")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data
    




class MetricManager:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
