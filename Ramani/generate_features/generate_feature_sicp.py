import numpy as np
# from origin_contact import calculate_bin_origin
import numpy as np
from multiprocessing import Pool
from itertools import chain
import sys


def find_contact_list(list):
    # 寻找每个bin的接触列表   寻找当前bin对应的邻居
    con = []
    for i in list:
        if i != 0:
            con.append(i)
    return con


def sum_matrix(matrix):
    # 计算染色质的总接触
    U = np.triu(matrix, 1)
    # 返回矩阵的上三角部分，不包括对角线
    D = np.diag(np.diag(matrix))  # 只有对角线元素，其余元素均为0
    # diag返回对角线元元素
    # array是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵
    # array是一个二维矩阵时，结果输出矩阵的对角线元素
    return sum(sum(U + D))  # 按列相加，最后再把198列都加起来
    # sum(U + D) （198，）
    # sum(sum(U + D)) 就一个数了


def Small_Domain_Struct_Contact_pro(contact_matrix, index, scale):
    # 计算单个细胞中单个染色质的染色质接触概率
    contact_matrix = np.array(contact_matrix)  # 染色体的接触矩阵
    new_matrix = np.zeros((index + 2 * scale, index + 2 * scale))  # 扩增接触矩阵
    SICP = []
    chr_total = sum_matrix(contact_matrix)  # 所有接触数的总和
    for i in range(index):
        for j in range(index):
            new_matrix[i + scale, j + scale] = contact_matrix[i, j]  # 这是把接触信息放进去
    for i in range(index):
        bin = i + scale
        a = sum_matrix(new_matrix[bin - scale:bin + scale + 1, bin - scale:bin + scale + 1])  # 每次选出3 * 3 的矩阵
        # [i : i + 2 * scale + 1 , i : i + 2 * scale + 1] # 类似于TAD域
        if a == 0:
            SICP.append(float(0))
        else:
            SICP.append(float(a / chr_total))
    return SICP


def con_ran(cell_id, type, chr_name, max_length, cell_dict):
    # con_ran(cell_id, type, chr_name, max_len, cell_dict)

    file_path = "../Data_filter/Ramani/%s/cell_%s_%s.txt" % (type, str(cell_id), chr_name)
    chr_file = open(file_path)
    scale = 1
    lines = chr_file.readlines()
    contact_matrix = np.zeros((max_length, max_length))
    for line in lines:
        # bin1，bin2是两个染色体片段的编号，num是bin1和bin2的接触数
        bin1, bin2, num = line.split()
        contact_matrix[int(bin1), int(bin2)] += int(float(num))
        if bin1 != bin2:
            contact_matrix[int(bin2), int(bin1)] += int(float(num))

    Sicp = Small_Domain_Struct_Contact_pro(contact_matrix, max_length, scale)

    return Sicp


def main():
    types = {'GM12878':44,'HAP1':214,'HeLa':258,'K562':110} # 一共648个细胞
    resolutions = [1000000]
    tps = sorted(types)
    # print(tps) ['1CSE', '2CSE', '4CSE', '64CSE', '8CSE']
    f = open('combo_hg19.genomesize')
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
    # 关闭文件流
    f.close()
    print(index)
    # 1mbp就是指，1M（1000000）个碱基对作为分辨率，所以rindex就是用来标注index的分辨率的
    rindex = "1mbp"
    # p = Pool(20) # 小鼠需要弄到20
    cell_dict = {}
    cell_number = 0
    chr_list = sorted(index[resolution].keys())
    for type in tps:
        for c_num in range(types[type]):
            cell_id = c_num + 1
            cell_number += 1
            cell_dict[str(cell_number)] = {}
            print(c_num)
            for chr in chr_list:
                # c_num在循环中是0-43（因为range（44）是0-43）、0-213、0-257、0-109
                max_len = index[resolution][chr]
                # print(index[resolution])
                # args = [[rindex, cell_id, type, chr_name, index[resolution][chr_name]] for chr_name in
                #         index[resolution]]

                # print(args)
                Bop = con_ran(cell_id, type, chr, max_len, cell_dict)
                cell_dict[str(cell_number)][chr] = Bop
    out_path = './Ramani_features/sicp/sicp.npy'
    np.save(out_path, cell_dict)


if __name__ == '__main__':
    main()
    # 特征思路：  (1):当前接触数在指定TAD域中的概率 (2) bin在对角线上的概率  （3）：线性邻居+空间邻居
