import os, random

import numpy as np

from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from torch.optim import Adam
from Collombet.pytorch_tools import EarlyStopping
from focal_loss import MultiClassFocalLossWithAlpha


def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def seed_torch(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_bin():
    f = open("../mm10.main.nochrM.chrom.sizes")
    index = {}
    resolution = 1000000
    lines = f.readlines()
    for line in lines:
        chr_name, length = line.split()
        chr_name = chr_name
        max_len = int(int(length) / resolution)
        index[chr_name] = max_len + 1
        f.seek(0, 0)
    f.close()
    return index


def load_loader(train_dataset, val_dataset, val_size):
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=32,
                              shuffle=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_size,
                            shuffle=False)
    return train_loader, val_loader


def load_BOP_data(BOP, idX, Y):
    index = generate_bin()
    chr_list = sorted(index.keys())
    X = []
    for cell in idX:
        bop = []
        for chr in chr_list:
            bop.append(BOP[cell[0]][chr])
        X.append(np.concatenate(bop).tolist())
    print(np.array(X).shape)  # 414 * 2660
    deal_dataset = TensorDataset(torch.from_numpy(np.array(X).astype(float)), torch.from_numpy(np.array(Y).astype(int)))
    return deal_dataset, np.array(X).shape[0]


def load_BCP_data(BCP, idX, Y):
    index = generate_bin()
    chr_list = sorted(index.keys())
    X = []
    for cell in idX:
        bcp = []
        for chr in chr_list:
            bcp.append(BCP[cell[0]][chr])
        X.append(np.concatenate(bcp).tolist())
    print(np.array(X).shape)  # 414 * 2660
    deal_dataset = TensorDataset(torch.from_numpy(np.array(X).astype(float)), torch.from_numpy(np.array(Y).astype(int)))
    return deal_dataset, np.array(X).shape[0]


def load_BOSP_data(BOSP, idX, Y):
    index = generate_bin()
    chr_list = sorted(index.keys())
    X = []
    for cell in idX:
        bcp = []
        for chr in chr_list:
            bcp.append(BOSP[cell[0]][chr])
        X.append(np.concatenate(bcp).tolist())
    print(np.array(X).shape)  # 414 * 2660
    deal_dataset = TensorDataset(torch.from_numpy(np.array(X).astype(float)), torch.from_numpy(np.array(Y).astype(int)))
    return deal_dataset, np.array(X).shape[0]


def com_linearsize(linear_size, Con_layer, kernel_size):
    for i in range(Con_layer):
        linear_size = int(((linear_size + 2 * 1 - kernel_size) / 1 + 1) // 2)
    if Con_layer == 0:
        linear_size = 0
    return linear_size


class create_model(nn.Module):
    def __init__(self, model_para):
        super(create_model, self).__init__()
        kernel_size, cnn_feature, dp, out_feature, Con_layer, linear_layer = model_para
        self.linear_layer = linear_layer
        linear_size_BOP_init = 2660
        linear_size_BCP_init = 2660
        linear_size_BOSP_init = 2660
        self.Con_layer_BOP, self.Con_layer_BCP , self.Con_layer_BOSP = Con_layer
        linear_size_BOP = com_linearsize(linear_size_BOP_init, self.Con_layer_BOP, kernel_size)
        linear_size_BCP = com_linearsize(linear_size_BCP_init, self.Con_layer_BCP, kernel_size)
        linear_size_BOSP = com_linearsize(linear_size_BOSP_init,self.Con_layer_BOSP,kernel_size)
        self.linear_size_BOP = linear_size_BOP
        self.linear_size_BCP = linear_size_BCP
        self.linear_size_BOSP = linear_size_BOSP
        self.cnn_feature = cnn_feature  # 通道数
        if self.Con_layer_BOP != 0:
            self.conv1_BOP = nn.Conv1d(in_channels=1, out_channels=cnn_feature, kernel_size=kernel_size, stride=1,
                                       padding=1)
            self.bn1_BOP = nn.BatchNorm1d(num_features=cnn_feature)
            self.rule1_BOP = nn.ReLU()
            self.pool_BOP = nn.MaxPool1d(kernel_size=2)
            # self.dropout_BCP = nn.Dropout(dp)
            self.Con_BOP = nn.Sequential()
            for i in range(self.Con_layer_BOP - 1):
                layer_id = str(i + 2)
                self.Con_BOP.add_module("conv%s" % layer_id,
                                        nn.Conv1d(in_channels=cnn_feature, out_channels=cnn_feature,
                                                  kernel_size=kernel_size, stride=1, padding=1))
                self.Con_BOP.add_module("bach%s" % layer_id, nn.BatchNorm1d(num_features=cnn_feature))
                self.Con_BOP.add_module("relu%s" % layer_id, nn.ReLU())
                self.Con_BOP.add_module("maxp%s" % layer_id, nn.MaxPool1d(kernel_size=2))
        if self.Con_layer_BCP != 0:
            self.conv1_BCP = nn.Conv1d(in_channels=1, out_channels=cnn_feature, kernel_size=kernel_size, stride=1,
                                       padding=1)
            self.bn1_BCP = nn.BatchNorm1d(num_features=cnn_feature)
            self.rule1_BCP = nn.ReLU()
            self.pool_BCP = nn.MaxPool1d(kernel_size=2)
            # self.dropout_BCP = nn.Dropout(dp)
            self.Con_BCP = nn.Sequential()
            for i in range(self.Con_layer_BCP - 1):
                layer_id = str(i + 2)
                self.Con_BCP.add_module("conv%s" % layer_id,
                                        nn.Conv1d(in_channels=cnn_feature, out_channels=cnn_feature,
                                                  kernel_size=kernel_size, stride=1, padding=1))
                self.Con_BCP.add_module("bach%s" % layer_id, nn.BatchNorm1d(num_features=cnn_feature))
                self.Con_BCP.add_module("relu%s" % layer_id, nn.ReLU())
                self.Con_BCP.add_module("maxp%s" % layer_id, nn.MaxPool1d(kernel_size=2))
                # self.Con_BCP.add_module("drop%s" % layer_id,nn.Dropout(dp))

        if self.Con_layer_BOSP != 0:
            self.conv1_BOSP = nn.Conv1d(in_channels=1, out_channels=cnn_feature, kernel_size=kernel_size, stride=1,
                                       padding=1)
            self.bn1_BOSP = nn.BatchNorm1d(num_features=cnn_feature)
            self.rule1_BOSP = nn.ReLU()
            self.pool_BOSP = nn.MaxPool1d(kernel_size=2)
            # self.dropout_BCP = nn.Dropout(dp)
            self.Con_BOSP = nn.Sequential()
            for i in range(self.Con_layer_BOSP - 1):
                layer_id = str(i + 2)
                self.Con_BOSP.add_module("conv%s" % layer_id,
                                        nn.Conv1d(in_channels=cnn_feature, out_channels=cnn_feature,
                                                  kernel_size=kernel_size, stride=1, padding=1))
                self.Con_BOSP.add_module("bach%s" % layer_id, nn.BatchNorm1d(num_features=cnn_feature))
                self.Con_BOSP.add_module("relu%s" % layer_id, nn.ReLU())
                self.Con_BOSP.add_module("maxp%s" % layer_id, nn.MaxPool1d(kernel_size=2))

        if linear_layer == 1:
            self.fc = nn.Linear(in_features=cnn_feature * (linear_size_BOP + linear_size_BCP + linear_size_BOSP),
                                out_features=5)
        else:
            self.fc1 = nn.Linear(in_features=cnn_feature * (linear_size_BOP + linear_size_BCP + linear_size_BOSP),
                                 out_features=out_feature)
            self.relu2 = nn.ReLU()
            self.dropout = nn.Dropout(dp)
            self.Linear = nn.Sequential()
            for i in range(linear_layer - 2):
                l_layer_id = str(i + 2)
                self.Linear.add_module("linear%s" % l_layer_id,
                                       nn.Linear(in_features=out_feature, out_features=out_feature))
                self.Linear.add_module("linear_relu%s" % l_layer_id, nn.ReLU())
                self.Linear.add_module("linear_dropout%s" % l_layer_id, nn.Dropout(dp))
            self.fc2 = nn.Linear(in_features=out_feature, out_features=5)

        self.layer_1 = nn.LayerNorm(out_feature,eps=1e-6)
        self.lstm = nn.LSTM(input_size=out_feature,hidden_size=out_feature,num_layers=3,bidirectional=False)
        self.linear3 = nn.Sequential(
            nn.Linear(in_features=out_feature  , out_features= out_feature),
            nn.ReLU(),
            nn.Dropout(dp)
        )
        self.layer_2 = nn.LayerNorm(out_feature , eps=1e-6)
    def forward(self, x1, x2, x3):
        if self.Con_layer_BOP != 0:
            x1 = self.rule1_BOP(self.bn1_BOP(self.conv1_BOP(x1)))
            x1 = self.pool_BOP(x1)
            # x1 = self.dropout_BCP(x1)
            x1 = self.Con_BOP(x1)
            x1 = x1.view(-1, self.cnn_feature * self.linear_size_BOP)

        if self.Con_layer_BCP != 0:
            x2 = self.rule1_BCP(self.bn1_BCP(self.conv1_BCP(x2)))
            x2 = self.pool_BCP(x2)
            # x1 = self.dropout_BCP(x2)
            x2 = self.Con_BCP(x2)
            x2 = x2.view(-1, self.cnn_feature * self.linear_size_BCP)

        if self.Con_layer_BOSP != 0:
            x3 = self.rule1_BOSP(self.bn1_BOSP(self.conv1_BOP(x3)))
            x3 = self.pool_BOSP(x3)
            # x3 = self.dropout_BCP(x3)
            x3 = self.Con_BOSP(x3)
            x3 = x3.view(-1, self.cnn_feature * self.linear_size_BOSP)

        x = torch.cat((x1, x2 , x3), 1)

        if self.linear_layer == 1:
            x = self.fc(x)
        else:
            x = self.fc1(x)
            x = self.layer_1(x)
            x = self.dropout(x)
            x = self.relu2(x)
            x = self.Linear(x)
            x, _ = self.lstm(x)
            x = self.linear3(x)
            x = self.layer_2(x)
            x = self.fc2(x)
        # x = nn.functional.log_softmax(x, dim=1)
        return x


def CNN_train(epoch, model, optimizer, train_loader, loss_fn, device):
    train_loader_BOP, train_loader_BCP, train_loader_BOSP = train_loader
    for (images_BOP, labels_BOP), (images_BCP, labels_BCP), (images_BOSP, labels_BOSP) \
            in zip(train_loader_BOP, train_loader_BCP, train_loader_BOSP):
        i = 0
        optimizer.zero_grad()
        labels = torch.Tensor(labels_BOP.type(torch.FloatTensor)).long()
        images_BOP = torch.unsqueeze(images_BOP.type(torch.FloatTensor), dim=1)
        images_BCP = torch.unsqueeze(images_BCP.type(torch.FloatTensor), dim=1)
        images_BOSP = torch.unsqueeze(images_BOSP.type(torch.FloatTensor), dim=1)
        images_BOP = images_BOP.to(device)
        images_BCP = images_BCP.to(device)
        images_BOSP = images_BOSP.to(device)
        labels = labels.to(device)
        outputs = model(images_BOP, images_BCP, images_BOSP)
        train_loss = loss_fn(outputs, labels)
        train_loss.backward()
        optimizer.step()
        train_loss += train_loss.cpu().data * images_BOP.size(0)
        _, prediction = torch.max(outputs.data, 1)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(images_BOP), len(train_loader_BOP.dataset),
                   100. * i / len(train_loader_BOP), train_loss.cpu().data * images_BOP.size(0)))
    return model, optimizer


def CNN_val(epoch, model, test_loader, loss_fn, test_size, device):
    test_loader_BOP, test_loader_BCP, test_loader_BOSP = test_loader
    i = 0
    for (images_BOP, labels_BOP), (images_BCP, labels_BCP), (images_BOSP, labels_BOSP) \
            in zip(test_loader_BOP, test_loader_BCP, test_loader_BOSP):
        images_BOP = torch.unsqueeze(images_BOP.type(torch.FloatTensor), dim=1)
        images_BCP = torch.unsqueeze(images_BCP.type(torch.FloatTensor), dim=1)
        images_BOSP = torch.unsqueeze(images_BOSP.type(torch.FloatTensor), dim=1)
        # images = images.to(device)train_loader
        # labels = labels.to(device)
        images_BOP = images_BOP.to(device)
        images_BCP = images_BCP.to(device)
        images_BOSP = images_BOSP.to(device)
        labels = torch.Tensor(labels_BOP.type(torch.FloatTensor)).long()
        labels = labels.to(device)
        outputs = model(images_BOP, images_BCP, images_BOSP)
        val_loss = loss_fn(outputs, labels)
        _, prediction = torch.max(outputs.data, 1)
        label_pred = prediction.cpu().numpy()
        label = labels.data.cpu().numpy()
        prediction_num = int(torch.sum(prediction == labels.data))
        val_accuracy = prediction_num / test_size
        print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(images_BOP), len(test_loader_BOP.dataset),
                   100. * i / len(test_loader_BOP), val_loss.cpu().data * images_BCP.size(0)))
        i = i + 1
    return label_pred, label, val_loss, model, val_accuracy


def CNN_1D_montage(BOP, BCP, BOSP, tr_x, tr_y, test_x, test_y, lr, model_para, alpha, gamma):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_torch()
    train_dataset_BOP, train_size_BOP = load_BOP_data(BOP, tr_x, tr_y)
    test_dataset_BOP, test_size_BOP = load_BOP_data(BOP, test_x, test_y)
    train_dataset_BCP, train_size_BCP = load_BCP_data(BCP, tr_x, tr_y)
    test_dataset_BCP, test_size_BCP = load_BCP_data(BCP, test_x, test_y)
    train_dataset_BOSP, train_size_BOSP = load_BOSP_data(BOSP, tr_x, tr_y)
    test_dataset_BOSP, test_size_BOSP = load_BOSP_data(BOSP, test_x, test_y)

    # train_loader_BCP, val_loader_BCP = load_loader(train_dataset_BCP, val_dataset_BCP, val_size_BCP)
    # train_loader_CDP, val_loader_CDP = load_loader(train_dataset_CDP, val_dataset_CDP, val_size_CDP)
    # train_loader_SBCP, val_loader_SBCP = load_loader(train_dataset_SBCP, val_dataset_SBCP, val_size_SBCP)

    train_loader_BOP, test_loader_BOP = load_loader(train_dataset_BOP,test_dataset_BOP, test_size_BOP)
    train_loader_BCP, test_loader_BCP = load_loader(train_dataset_BCP, test_dataset_BCP, test_size_BCP)
    train_loader_BOSP, test_loader_BOSP = load_loader(train_dataset_BOSP, test_dataset_BOSP, test_size_BOSP)
    device = try_gpu(3)
    print(device)
    model = create_model(model_para)
    print(model)
    model.to(device)
    train_loader = [train_loader_BOP, train_loader_BCP, train_loader_BOSP]
    val_loader = [test_loader_BOP, test_loader_BCP, test_loader_BOSP]
    early_stopping = EarlyStopping(patience=10, verbose=True)  # 只有10个验证集的loss没有降低，early_stop才会返回True
    num_epochs = 50
    min_loss = 100000.0
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MultiClassFocalLossWithAlpha(alpha=alpha, gamma=gamma)
    for epoch in range(num_epochs):
        model.train()
        model, optimizer = CNN_train(epoch, model, optimizer, train_loader, loss_fn, device)
        model.eval()
        test_label, label, test_loss, model, test_accuracy = CNN_val(epoch, model, val_loader,
                                                                  loss_fn, test_size_BOP, device)

    torch.cuda.empty_cache()

    return test_accuracy,test_label,label
