import argparse
import os

from train import *

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CustomDataset
from utils import format_item, generate_item, load_config
from model import AnoSRe

parser = argparse.ArgumentParser(description='Load your dataset and parameters')
parser.add_argument('--config', type=str, help='Path to the config file')
parser.add_argument('--save_dir', type=str, help='Path to your output position')
parser.add_argument('--VE', type=str, help='Train your vision encoder')
parser.add_argument('--TRE', type=str, help='Train your vision encoder')
parser.add_argument('--FD', type=str, help='Train your structured data encoder')
parser.add_argument('--VCE', type=str, help='Train your vision encoder clip')
parser.add_argument('--TCE', type=str, help='Train your vision encoder clip')
args = parser.parse_args()

config = load_config(args.config)
output_dir = args.save_dir
category = config['category']
max_length = config['max_length']
path_to_dataset = config['path_to_dataset']
epoch = config['epoch']
nw = config['num_workers']
lr = config['lr']
bs = config['batch_size']

pd_train_all, pd_test_all = format_item(path_to_dataset + '/train.csv'), format_item(path_to_dataset + '/test.csv')
train_list, test_list = generate_item(pd_train_all), generate_item(pd_test_all)
train_ds = CustomDataset(*train_list, max_length, args.TRE, args.VCE, args.TCE)
test_ds = CustomDataset(*test_list, max_length, args.TRE, args.VCE, args.TCE)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
test_dl = DataLoader(test_ds, batch_size=bs, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device\n")

model = AnoSRe(category, args.VE, args.TRE, args.FD, args.VCE, args.TCE).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = epoch
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
test_top3_acc_list = []

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dl, model, loss_fn, optimizer, device)

    train_loss, train_correct, _ = test(train_dl, model, loss_fn, device, is_train=True)
    test_loss, test_correct, test_correct_top3 = test(test_dl, model, loss_fn, device, is_train=False)

    train_loss_list.append(train_loss)
    train_acc_list.append(train_correct)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_correct)
    test_top3_acc_list.append(test_correct_top3)

print("Done!")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_path = os.path.join(output_dir, 'model_state_dict.pth')
torch.save(model.state_dict(), model_path)
