from torch.utils.data import DataLoader

from torch import Tensor
from config import *
import torch
import torch.nn as nn
import dgl
from dataset import LuDataset
from model.gatnet import GatNet
from utils.assess import calculate_indicators, calculate_auc


def evaluate_val_accuracy_gpu(epoch, net, data_iter, device=None):
    pred_labs = []
    real_labs = []
    val_l = 0.
    net.eval()
    loss = nn.CrossEntropyLoss()
    for batch_idx, (seqmatrix, label, pssms, dssps, concatdata, emd, graph) in enumerate(data_iter):
        # data = [d.to(device) for d in data]
        y_hat = net(dgl.batch(graph).to(device), pad_dmap(emd))
        pred = y_hat.argmax(dim=1)
        l = loss(y_hat, label.to(device))
        list_len = len(pred)
        li = []
        for i in range(list_len):
            li.append([pred[i]])
        one_hot_list = torch.LongTensor(li).to(device)
        lab = torch.zeros(list_len, 2).to(device).scatter_(1, one_hot_list, 1)
        pred_labs.append(lab)
        real_labs.append(label.to(device))
        val_l += l
    assess = calculate_indicators(epoch, pred_labs, real_labs)
    print(
        "Val   Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}\t Precision: {:.6f}\t Recall: {:.6f}\t F1: {:.6f}\t Sensitivity: {:.6f}\t Specificity: {:.6f}\t".format(
            epoch, val_l / len(data_iter), assess.get("accuracy"), assess.get("precision"), assess.get("recall"),
            assess.get("f1"), assess.get("sensitivity"), assess.get("specificity")))


def evaluate_test_accuracy_gpu(epoch, net, data_iter, device=None):
    pred_labs = []
    real_labs = []
    val_l = 0.
    net.eval()
    roc_poslist = []
    reallab = []
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (seqmatrix, label, pssms, dssps, concatdata, emd, graph) in enumerate(data_iter):
            # data = [d.to(device) for d in data]
            y_hat = net(dgl.batch(graph).to(device), pad_dmap(emd))
            for yh in y_hat:
                roc_poslist.append(Tensor.cpu(yh[1]).numpy())
            real_l = label.argmax(dim=1)
            for rl in real_l:
                reallab.append(Tensor.cpu(rl).numpy())
            pred = y_hat.argmax(dim=1)
            l = loss(y_hat, label.to(device))
            list_len = len(pred)
            li = []
            for i in range(list_len):
                li.append([pred[i]])
            one_hot_list = torch.LongTensor(li).to(device)
            lab = torch.zeros(list_len, 2).to(device).scatter_(1, one_hot_list, 1)
            pred_labs.append(lab)
            real_labs.append(label.to(device))
            val_l += l
    val_loss.append(val_l / len(data_iter))
    assess = calculate_indicators(epoch, pred_labs, real_labs)
    auccal = calculate_auc(roc_poslist, reallab)
    print(
        "Test  Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}\t Precision: {:.6f}\t Recall: {:.6f}\t F1: {:.6f}\t Sensitivity: {:.6f}\t Specificity: {:.6f}\t".format(
            epoch, val_l / len(data_iter), assess.get("accuracy"), assess.get("precision"), assess.get("recall"),
            assess.get("f1"), assess.get("sensitivity"), assess.get("specificity")))
    print(
        "---------------------------------------------------------------------------------------------------------------------------------------------------")

def collate(samples):
    seqmatrix, label, pssms, dssps, concatdata, emd, graph = map(list, zip(*samples))
    labels = []
    for i in label:
        labels.append(i)
    labels = torch.tensor(labels)
    return seqmatrix, labels, pssms, dssps, concatdata, emd, graph

def train_deepptm(train_neg_file, train_pos_file, train_file_name, num_epochs=num_epochs, lr=lr,
                  device=torch.device('cuda')):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv1d:
            nn.init.xavier_uniform_(m.weight)
    # 批量装载数据集
    train_data = LuDataset(train_file_name=train_file)
    train_iter = DataLoader(dataset=train_data, batch_size=128, shuffle=True, drop_last=True, collate_fn=collate)
    print("加载训练数据集 batch_size=128")

    val_data = LuDataset(train_file_name=val_file)
    val_iter = DataLoader(dataset=val_data, batch_size=128, shuffle=True, drop_last=True, collate_fn=collate)
    print("加载验证数据集 batch_size=128")

    test_data = LuDataset(train_file_name=test_file)
    test_iter = DataLoader(dataset=test_data, batch_size=128, shuffle=True, drop_last=True, collate_fn=collate)
    print("加载测试数据集 batch_size=128")

  #初始化模型
    net = GatNet()
    net.apply(init_weights)
    if torch.cuda.device_count()>1:
        print(f'let us use{torch.cuda.device_count()} GPUS')
        net=nn.DataParallel(net)
    net.to(device)
    print(f'let us use{torch.cuda.device_count()} GPUS')
    print("模型装载到显存....")
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.0001)
    loss = nn.CrossEntropyLoss()
    # 开始训练
    print("开始训练....")
    for epoch in range(num_epochs):
        # 定义指标
        num_correct = 0
        train_l = 0.
        pred_labs = []
        real_labs = []
        net.train()
        for batch_idx, (seqmatrix, label, pssms, dssps, concatdata, emd, graph) in enumerate(train_iter):
            optimizer.zero_grad()
            # data = [d.to(device) for d in data]
            # seqmatrix, label, pssms, dssps, concatdata, emd, graph=data[0],data[1],data[2],data[3],data[4],data[5],data[6]
            y_hat = net(dgl.batch(graph).to(device), pad_dmap(emd))
            pred = y_hat.argmax(dim=1)
            li = []
            for i in range(len(pred)):
                li.append([pred[i]])
            one_hot_list = torch.LongTensor(li).to(device)
            pre_lab = torch.zeros(len(pred), 2).to(device).scatter_(1, one_hot_list, 1)
            pred_labs.append(pre_lab)
            l = loss(y_hat, label.to(device))
            l.backward()
            optimizer.step()
            train_l += l
            real_labs.append(label.to(device))
        step = len(train_iter)  # 样本总数/batchsize是走完一个epoch所需的“步数”
        ex_sum = len(train_data)  # 样本总数
        train_loss.append(train_l / len(train_iter))
        assess = calculate_indicators(epoch, pred_labs, real_labs)
        print(
            "Train Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}\t Precision: {:.6f}\t Recall: {:.6f}\t F1: {:.6f}\t Sensitivity: {:.6f}\t Specificity: {:.6f}\t".format(
                epoch, train_l / len(train_iter), assess.get("accuracy"), assess.get("precision"), assess.get("recall"),
                assess.get("f1"), assess.get("sensitivity"), assess.get("specificity")))
        evaluate_val_accuracy_gpu(epoch, net, val_iter, device)
        evaluate_test_accuracy_gpu(epoch, net, test_iter, device)
def pad_dmap(dmaplist):
    pad_dmap_tensors = torch.zeros((len(dmaplist), 1000, 1024)).float()
    for idx, d in enumerate(dmaplist):
        d = d.float().cpu()
        pad_dmap_tensors[idx] = torch.FloatTensor(d)
    pad_dmap_tensors = pad_dmap_tensors.unsqueeze(1).cuda()
    return pad_dmap_tensors
if __name__ == '__main__':
    train_loss = []
    val_loss = []
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    train_deepptm(train_neg_file=None, train_pos_file=None, train_file_name=None)