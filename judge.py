import timm
from torch import nn
import mytorchutils
import torch
from torch.utils.data import DataLoader
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import cartopy.crs as ccrs

#用TIR模型分类白天的图像

params = {
    # 将需要进行分类的文件放置在该文件夹下，文件夹名称可修改
    'model_path': "/work11/wuyy/yun_IR_zuhe/model_COT_hunxiaojuzheng_1_2_1___title/64epochs_accuracy0.91456.pth",
    'batch_size': 512,
    'device_num': '2'
}

def save_batch_to_npz(all_cats, all_certs, subdirectory):
    npz_filename = f'judged_night_{subdirectory}.npz'
    npz_path = os.path.join('/work13/wuyy/twenty_years_nofilter_judged_CB_new/day_TIR_model/2018',subdirectory, npz_filename)
    if not os.path.exists(os.path.dirname(npz_path)):
        os.makedirs(os.path.dirname(npz_path))
    np.savez(npz_path, cats=all_cats, certs=all_certs)


#判断夜间图像，没有label, 只预测
def judge(judge_loader, model, class_names, device, subdirectory):
    model.eval()
    all_cats = []
    all_certs = []

    with torch.no_grad():
        for i, (input) in enumerate(judge_loader, start=1):
            input = input.to(torch.float32).to(device)
            output = model(input)
            y_pred = torch.softmax(output, dim=1).cpu()
            cert = torch.max(y_pred, dim=1).values.cpu() / torch.sum(y_pred, dim=1).cpu()  #预测的置信度
            cat = torch.argmax(y_pred, dim=1).cpu().numpy()  #预测的类别
            print(cat,cert)
            all_cats.extend(cat)
            all_certs.extend(cert)  #因为 append 将整个 cats 数组作为一个元素添加到列表中，而 extend 将 cats 数组中的每个元素作为单独的元素添加到列表中
    
    all_cats = np.array(all_cats)
    all_certs = np.array(all_certs)
    print(all_cats.shape)
    save_batch_to_npz(all_cats, all_certs, subdirectory)


if __name__ == '__main__':
    # 初始化类名、设备、数据集、模型
    class_names = ['Solid stratus','Closed MCC', 'Open MCC','Disorganized MCC','Clustered Cu', 'Suppressed Cu']
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device('cuda:' + params['device_num'] if torch.cuda.is_available() else 'cpu')

    folder_path = "/work13/wuyy/twenty_years_night_1d_production/day_TIR_model_production/2018"

    # 获取指定目录下的所有子目录
    subdirectories = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    size = len(subdirectories)
    print(subdirectories,size)
    for subdirectory in subdirectories:
        print("子目录：",subdirectory)

        subdirectory_path = os.path.join(folder_path, subdirectory)

        test_dataset = mytorchutils.JudgeDataset_2014(subdirectory_path)
        test_len = mytorchutils.CloudDataset_new.__len__(test_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
        model = timm.create_model('resnet50d', pretrained=False)
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(2048, 6)
        model = model.to(device)
        weights = torch.load(params['model_path'])
        model.load_state_dict(weights)
        model = model.to(device)
        # 开始判断
        judge(test_dataloader, model, class_names, device, subdirectory)
        # 清空test_dataset和test_dataloader
        del test_dataset
        del test_dataloader