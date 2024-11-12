import scipy.io
import os
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


#转为BT亮温
def func_CAL_BT(radiance, wavelength):
    const1 = (1.19107*10**8) / (wavelength**5.0)
    const2 = (1.43883*10**4) / wavelength
    BT = const2 / np.log(1 + const1 / radiance)
    return BT


#全年judge
def file_reading(filename):
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
                        cot2 = cot1.copy()  
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
class JudgeDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = []
        for file_name in os.listdir(data_path):
            if not file_name.endswith('.mat'):
                print("some files cant be judged")
                continue
            file_path = os.path.join(data_path, file_name)
            mat_data,num_layers = file_reading(file_path)
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
    

