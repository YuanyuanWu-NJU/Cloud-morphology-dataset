import timm
from torch import nn
import mytorchutils
import torch
from torch.utils.data import DataLoader
import os
import numpy as np

# Only need input:
params = {
    'folder_path': 'The folder containing the processed 1-degree grid data.',
    'model_path': './64epochs_accuracy0.91456.pth',
    'batch_size': 512,
    'device_num': '1',
    'save_path':'The path to save the classification results'
}


def save_batch_to_npz(all_cats, all_certs, subdirectory):
    npz_filename = f'judged_night_{subdirectory}.npz'
    npz_path = os.path.join(params['save_path'],subdirectory, npz_filename)
    if not os.path.exists(os.path.dirname(npz_path)):
        os.makedirs(os.path.dirname(npz_path))
    np.savez(npz_path, cats=all_cats, certs=all_certs)


def judge(judge_loader, model, class_names, device, subdirectory):
    model.eval()
    all_cats = []
    all_certs = []

    with torch.no_grad():
        for i, (input) in enumerate(judge_loader, start=1):
            input = input.to(torch.float32).to(device)
            output = model(input)
            y_pred = torch.softmax(output, dim=1).cpu()
            cert = torch.max(y_pred, dim=1).values.cpu() / torch.sum(y_pred, dim=1).cpu() 
            cat = torch.argmax(y_pred, dim=1).cpu().numpy()  
            print(cat,cert)
            all_cats.extend(cat)
            all_certs.extend(cert)
    
    all_cats = np.array(all_cats)
    all_certs = np.array(all_certs)
    print(all_cats.shape)
    save_batch_to_npz(all_cats, all_certs, subdirectory)


if __name__ == '__main__':
    class_names = ['Solid stratus','Closed MCC', 'Open MCC','Disorganized MCC','Clustered Cu', 'Suppressed Cu'] #[0,1,2,3,4,5]
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device('cuda:' + params['device_num'] if torch.cuda.is_available() else 'cpu')

    subdirectories = [f for f in os.listdir(params['folder_path']) if os.path.isdir(os.path.join(params['folder_path'], f))]
    size = len(subdirectories)
    print(subdirectories,size)
    for subdirectory in subdirectories:
        print(subdirectory)
        subdirectory_path = os.path.join(params['folder_path'], subdirectory)
        test_dataset = mytorchutils.JudgeDataset(subdirectory_path)
        test_len = mytorchutils.JudgeDataset.__len__(test_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
        model = timm.create_model('resnet50d', pretrained=False)
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(2048, 6)
        model = model.to(device)
        weights = torch.load(params['model_path'])
        model.load_state_dict(weights)
        model = model.to(device)

        judge(test_dataloader, model, class_names, device, subdirectory)

        del test_dataset
        del test_dataloader