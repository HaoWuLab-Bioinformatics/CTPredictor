import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from Nagano.nagano_network_two_feature import CNN_1D_struct_two_features
import xlsxwriter
from test_method import CNN_1D_montage
from sklearn.metrics import f1_score,precision_score
from collections import Counter

def load_Ssicp_dict():
    file_path = "../Collombet_features/bosicp/bosicp.npy"
    Data = np.load(file_path, allow_pickle=True).item()  # 返回的长度为细胞数量
    return Data


def load_Sbop_dict():
    file_path = '../Collombet_features/bin_linear_space_p/Bin_contact_linear_space_p.npy'
    Data = np.load(file_path, allow_pickle=True).item()  # 返回的长度为细胞数量
    return Data

def load_Sbcp_dict():
    file_path = "../Collombet_features/sbcp/sbcp.npy"
    Data = np.load(file_path, allow_pickle=True).item()  # 返回的长度为细胞数量
    return Data

def load_Sicp_dict():
    file_path = "../Collombet_features/Sicp/sicp.npy"
    Data = np.load(file_path, allow_pickle=True).item()  # 返回的长度为细胞数量
    return Data

def load_Nbcp_dict():
    file_path = "../Collombet_features/nbcp/nbcp.npy"
    Data = np.load(file_path, allow_pickle=True).item()  # 返回的长度为细胞数量
    return Data


def main():

    SICP = load_Sicp_dict()
    SSICP = load_Ssicp_dict()
    SBCP = load_Sbcp_dict()
    types = {'1CSE':85,'2CSE':114,'4CSE':121,'64CSE':205,'8CSE':123} # 一共648个细胞
    Y = []
    X = []
    resolutions = [1000000]
    tps = sorted(types)
    # print(tps) ['1CSE', '2CSE', '4CSE', '64CSE', '8CSE']

    # alpha用于计算focal_loss
    # 每个类别对应的alpha=该类别出现频率的倒数
    alpha = []
    for value in types.values():
        ds = 1 / value
        alpha.append(ds)
    print(alpha)


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
            if type == '1CSE':
                Y.append(0)
            elif type == '2CSE':
                Y.append(1)
            elif type == '4CSE':
                Y.append(2)
            elif type == '8CSE':
                Y.append(3)
            elif type == '64CSE':
                Y.append(4)
    X = np.array(X).reshape(cell_number, 1)
    Y = np.array(Y)

    Con_layer_NBCP = 2
    Con_layer_SICP = 2
    Con_layer_SSICP = 2

    Con_layer = [Con_layer_NBCP,Con_layer_SICP,Con_layer_SSICP]
    linear_layer = 3
    kernel_size = 11
    cnn_feature = 32
    out_feature = 128
    dp = 0.2
    lr = 0.0001
    model_para = [kernel_size,cnn_feature,dp,out_feature,Con_layer,linear_layer]
    # 以上所有参数都要待定！！！！！,包括gamma，要等到跑完上一步那个多层for循环以后才能确定。
    gamma = 1
    # 验证集参数
    # linear_layer = 3
    # kernel_size = 11
    # cnn_feature = 32
    # out_feature = 128
    # dp = 0.2
    # lr = 0.0001
    # model_para = [kernel_size,cnn_feature,dp,out_feature,Con_layer,linear_layer]
    # # 以上所有参数都要待定！！！！！,包括gamma，要等到跑完上一步那个多层for循环以后才能确定。
    # gamma = 1

    # ！！！随机种子也待定
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4096, stratify=Y)
    test_acc , test_label , real_label = CNN_1D_montage(SBCP,SICP,SSICP,X_train,y_train,X_test,y_test,lr ,model_para,alpha,gamma)
    print('test_acc: ', test_acc)
    label_count = []
    for i, j in zip(test_label, real_label):
        if i == j:
            label_count.append(i)
    print("预测结果：", Counter(label_count))

    from sklearn import metrics
    micro_F1 = f1_score(real_label, test_label, average='micro')
    print("micro_F1：", micro_F1)
    macro_F1 = f1_score(real_label, test_label, average='macro')
    print("macro_F1：", macro_F1)
    micro_Precision = precision_score(real_label, test_label, average='micro')
    print("micro_Precision：", micro_Precision)
    macro_Precision = precision_score(real_label, test_label, average='macro')
    print("macro_Precision：", macro_Precision)
    Precision = metrics.precision_score(real_label, test_label, average='weighted')
    print("Precision：", Precision)
    Mcc = metrics.matthews_corrcoef(real_label, test_label)
    print("Mcc：", Mcc)
    F1 = metrics.f1_score(real_label, test_label, average='weighted')
    print("F1：", F1)
    ari = metrics.adjusted_rand_score(real_label, test_label)
    print("ARI：", ari)
    Bacc = metrics.balanced_accuracy_score(real_label, test_label)
    print("Bacc：", Bacc)
    Nmi = metrics.normalized_mutual_info_score(real_label, test_label)
    print("NMI", Nmi)

if __name__ == '__main__':
    main()
