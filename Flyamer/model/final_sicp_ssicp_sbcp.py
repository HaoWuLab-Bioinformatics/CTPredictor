import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from Nagano.nagano_network_two_feature import CNN_1D_struct_two_features
import xlsxwriter
from test_method import CNN_1D_montage
from sklearn.metrics import f1_score,precision_score
from collections import Counter

def load_Bosicp_dict():
    file_path = "../features/bosicp/bosicp.npy"
    Data = np.load(file_path, allow_pickle=True).item()  # 返回的长度为细胞数量
    return Data


def load_Bosp_dict():
    file_path = '../features/bosp/bosp.npy'
    Data = np.load(file_path, allow_pickle=True).item()  # 返回的长度为细胞数量
    return Data

def load_Bcp_dict():
    file_path = "../features/bcp/bcp.npy"
    Data = np.load(file_path, allow_pickle=True).item()  # 返回的长度为细胞数量
    return Data

def load_Sicp_dict():
    file_path = "../features/sicp/sicp.npy"
    Data = np.load(file_path, allow_pickle=True).item()  # 返回的长度为细胞数量
    return Data

def load_Ssicp_dict():
    file_path = "../features/ssicp/ssicp.npy"
    Data = np.load(file_path, allow_pickle=True).item()  # 返回的长度为细胞数量
    return Data


def load_Sbop_dict():
    file_path = '../features/sbop/sbop.npy'
    Data = np.load(file_path, allow_pickle=True).item()  # 返回的长度为细胞数量
    return Data

def load_Sbcp_dict():
    file_path = "../features/sbcp/sbcp.npy"
    Data = np.load(file_path, allow_pickle=True).item()  # 返回的长度为细胞数量
    return Data

def load_Sicp_dict():
    file_path = "../features/sicp/sicp.npy"
    Data = np.load(file_path, allow_pickle=True).item()  # 返回的长度为细胞数量
    return Data


def main():
    SBCP = load_Sbcp_dict()
    SICP = load_Sicp_dict()
    SBOP = load_Sbop_dict()
    SSICP = load_Ssicp_dict()
    types = {'oocytes':114,'ZygM':32,'ZygP':32}  # 一共648个细胞
    Y = []
    X = []
    resolutions = [1000000]
    tps = sorted(types)
    alpha = []
    for value in types.values():
        ds = 1 / value
        alpha.append(ds)
    print(alpha)
    # print(tps) ['1CSE', '2CSE', '4CSE', '64CSE', '8CSE']
    f = open('../mm10.main.nochrM.chrom.sizes')
    index = {}
    for resolution in resolutions:
        index[resolution] = {}
        lines = f.readlines()
        for line in lines:
            chr_name, length = line.split()
            # 经过下面这行代码以后，chr_name的值由human_chr1变为chr1
            # chr_name = chr_name.split('_')[1]
            # max_len+1是指一个长度为length的染色体在resolution分辨率下能分成max_len+1块
            # 为什么要+1？因为int(10/3)=3，是向下取整的，多出来的那一截，也要做一块。
            max_len = int(int(length) / resolution)
            # index二维字典中index[resolution][chr_name]存的是染色体chr_name在分别率为resolution时能分出的块数
            index[resolution][chr_name] = max_len + 1
        f.seek(0, 0)
    f.close()
    cell_number = 0
    for type in tps:
        for c_num in range(types[type]):
            cell_number += 1
            X.append(str(cell_number))
            if type == 'oocytes':
                Y.append(0)
            elif type == 'ZygM':
                Y.append(1)
            elif type == 'ZygP':
                Y.append(2)
    X = np.array(X).reshape(cell_number, 1)
    Y = np.array(Y)


    Con_layer_SBCP = 2
    Con_layer_SSICP = 2
    Con_layer_SICP = 2

    Con_layer = [Con_layer_SSICP,Con_layer_SBCP,Con_layer_SICP]
    # linear_layer = 3
    # kernel_size = 7
    # cnn_feature = 32
    # out_feature = 64
    # dp = 0.3
    # lr = 0.0001
    # gamma = 5
    linear_layer = 3
    kernel_size = 11
    cnn_feature = 32
    out_feature = 128
    dp = 0.5
    lr = 0.0001
    gamma = 3
    model_para = [kernel_size,cnn_feature,dp,out_feature,Con_layer,linear_layer]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4096, stratify=Y)
    test_acc , test_label , real_label = CNN_1D_montage(SICP,SSICP,SBCP,X_train,y_train,X_test,y_test,lr ,model_para,alpha,gamma)
    print('test_acc: ', test_acc)
    label_count = []
    for i, j in zip(test_label, real_label):
        if i == j:
            label_count.append(i)
    print("预测结果：", Counter(label_count))

    from sklearn import metrics
    micro_F1 = f1_score(real_label, test_label, average = 'micro')
    print("micro_F1：", micro_F1)
    macro_F1 = f1_score(real_label, test_label, average='macro')
    print("macro_F1：", macro_F1)
    micro_Precision = precision_score(real_label, test_label,average = 'micro')
    print("micro_Precision：", micro_Precision)
    macro_Precision = precision_score(real_label, test_label, average='macro')
    print("macro_Precision：", macro_Precision)
    Precision = metrics.precision_score(real_label, test_label, average='weighted')
    print("Precision：", Precision)
    Mcc = metrics.matthews_corrcoef(real_label,test_label)
    print("Mcc：", Mcc)
    F1 = metrics.f1_score(real_label,test_label,average='weighted')
    print("F1：",F1 )
    ari = metrics.adjusted_rand_score(real_label,test_label)
    print("ARI：",ari )
    Bacc = metrics.balanced_accuracy_score(real_label,test_label)
    print("Bacc：", Bacc)
    Nmi = metrics.normalized_mutual_info_score(real_label,test_label)
    print("NMI", Nmi)

if __name__ == '__main__':
    main()