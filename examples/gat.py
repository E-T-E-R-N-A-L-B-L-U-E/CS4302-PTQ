import sys
sys.path.append("../")

import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv

from torchsummary import summary

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=8)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_wandb(name=f'GAT-{args.dataset}', heads=args.heads, epochs=args.epochs,
           hidden_channels=args.hidden_channels, lr=args.lr, device=device)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    
    def echo(self):
        def echoConv(conv: GATConv):
            print("lin: ", conv.lin)
            print("lin_src: ", conv.lin_src)
            print("lin_dst: ", conv.lin_dst)
            print("lin_edge: ", conv.lin_edge)
            print("att_src: ", conv.att_src)
            print("att_dst: ", conv.att_dst)
            print("att_edge: ", conv.att_edge)
        print("=====Conv1: ")
        echoConv(self.conv1)
        print("=====Conv2: ")
        echoConv(self.conv2)


model = GAT(dataset.num_features, args.hidden_channels, dataset.num_classes,
            args.heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

@torch.no_grad()
def quantization(model):
    model.eval()
    model.qconfig = torch.ao.quantization.get_default_qconfig('x86')

    model_fp32_fused = torch.ao.quantization.fuse_modules(model, [['conv', 'relu']])

    model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)

    input_fp32 = torch.randn(4, 1, 4, 4)
    model_fp32_prepared(input_fp32)

    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

    return model_int8

# model.eval()
# model.echo()

# times = []
# best_val_acc = final_test_acc = 0
# for epoch in range(1, args.epochs + 1):
#     start = time.time()
#     loss = train()
#     train_acc, val_acc, tmp_test_acc = test(model)
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         test_acc = tmp_test_acc
#     log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
#     times.append(time.time() - start)
# print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

# # Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# # Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])

# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict()
# }, "./checkpoints/gat.pth")

# start quantization
# print("====== start quantization ======")
# model_int8 = quantization(model)
# train_acc, val_acc, tmp_test_acc = test(model_int8)
# log(Train=train_acc, Val=val_acc, Test=tmp_test_acc)

# Linear: lin, lin_src, lin_dst, lin_edge
# Parameter: att_src, att_dst, att_edge, bias