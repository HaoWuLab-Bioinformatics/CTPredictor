import numpy as np
# from origin_contact import calculate_bin_origin
import numpy as np
from multiprocessing import Pool
from itertools import chain
import sys

def find_contact_list(list):
    #寻找每个bin的接触列表   寻找当前bin对应的邻居
    con = []
    for i in list:
        if i!=0:
            con.append(i)
    return con

def sum_matrix(matrix):
    #计算染色质的总接触
    U = np.triu(matrix, 1)
    # 返回矩阵的上三角部分，不包括对角线
    D = np.diag(np.diag(matrix))  # 只有对角线元素，其余元素均为0
    # diag返回对角线元元素
    # array是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵
    # array是一个二维矩阵时，结果输出矩阵的对角线元素
    return sum(sum(U + D))  # 按列相加，最后再把198列都加起来
    # sum(U + D) （198，）
    # sum(sum(U + D)) 就一个数了
def smooth_mean(list):
    # 在numpy的sum函数中axis为0是压缩行，即将每一列的元素相加,将矩阵压缩为一行
    # 以m = numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])为例
    # print(numpy.sum(m,axis = 0))结果为[22 26 30]
    # 在下面那一行代码中该函数对当前bin的空间相邻（线性+有相互作用的）bin的接触信息组成的矩阵的每一列进行了求和
    # 求和后再除以List列表的长度，它的长度为b+1，其中b为邻居的个数，再加上自己。
    return np.sum(np.array(list), axis=0) / len(list)

def calculate_sbcp(contact_matrix, max_length):
    # 先平滑
    matrix1 = contact_matrix
    # len(contact_matrix)为接触矩阵的行数
    for i in range(len(contact_matrix)):
        # 列表space_neighbour_contact_infor用来存放当前bin的空间相邻bin(线性相邻+相互作用即空间相邻)的接触信息。
        space_neighbour_contact_infor = []
        # 列表index1用来存放与当前bin线性相邻的bin的编号
        index1 = []
        # 先把当前bin自己的接触信息放进去
        space_neighbour_contact_infor.append(contact_matrix[i, :])

        # 该循环用于取得与当前bin线性相邻的bin(左右邻居)的接触信息
        # m的取值为i-1、i、i+1。
        for m in range(i - 1, i + 2):
            # 因为数组是从0号元素开始的(即接触矩阵是从0行开始的)所以m不能<0
            # 因为数组最多到len(matrix)-1号元素(即接触矩阵最多到len(matrix)-1行），所以m不能>len(matrix)-1
            # 因为此循环外，上边一行代码已经取了染色体片段i的接触信息了，此处不能再取一次，所以m不能等于i。
            if m < 0 or m > len(contact_matrix) - 1 or m == i:
                continue
            space_neighbour_contact_infor.append(contact_matrix[m, :])
            # 记录与当前bin线性相邻的bin的编号
            index1.append(m)

        # 此循环的作用是取得与当前bin相互作用(即空间相邻)的bin的接触信息。
        for n in range(len(contact_matrix[0])):
            # 如果n已经是当前bin的线性邻居的话就跳过
            if n in index1:
                continue
            # matrix[i, n] ！= 0则说明当前bin与bin_n发生相互作用，即空间相邻。
            # 而且因为此前已经取了当前bin的接触信息了，此处不能再取一次，所以i!=n。
            if contact_matrix[i, n] != 0 and i != n:
                space_neighbour_contact_infor.append(contact_matrix[n, :])
        # 用平滑处理过后的bin接触信息替换原来bin对应的接触信息
        matrix1[i, :] = smooth_mean(space_neighbour_contact_infor)
    m = np.triu(matrix1, 1)
    n = m.T
    # np.diag(matrix1)提取了方阵matrix1的主对角线上的元素。
    # np.diag(np.diag(matrix1))将以np.diag(matrix1)提取出的方阵matrix1的主对角线上的元素作为主对角线，构建一个除主对角线外的其他元素都为0的方阵。
    smoothed_matrix = m + n + np.diag(np.diag(matrix1))
    # ------------------------------------------------------------
    # 再求特征
    sbcp = []
    # sum_matrix用于计算单个染色质上的总接触数，因为矩阵是对称的，所以算染色质上的总接触数只需要计算上三角元素+对角线元素。
    chr_total = sum_matrix(smoothed_matrix)
    for i in range(max_length):
        con_list = find_contact_list(smoothed_matrix[i])
        if len(con_list) == 0:
            sbcp.append(0)
        else:
            con_pro = sum(con_list) / chr_total
            sbcp.append(con_pro)
    return sbcp

def con_ran(cell_id,type,chr_name,max_length,cell_dict):
    # con_ran(cell_id, type, chr_name, max_len, cell_dict)

    file_path = "../Data_filter/Flymer_three/%s/cell_%s_%s.txt" % (type, str(cell_id), chr_name)
    chr_file = open(file_path)
    lines = chr_file.readlines()
    contact_matrix = np.zeros((max_length,max_length))
    for line in lines:
        # bin1，bin2是两个染色体片段的编号，num是bin1和bin2的接触数
        bin1, bin2, num = line.split()
        contact_matrix[int(bin1), int(bin2)] += int(float(num))
        if bin1 != bin2:
            contact_matrix[int(bin2), int(bin1)] += int(float(num))
    Bop = calculate_sbcp(contact_matrix,max_length)

    return Bop


def main():
    types = {'oocytes':114,'ZygM':32,'ZygP':32}  # 一共648个细胞
    resolutions = [1000000]
    tps = sorted(types)
    # print(tps) ['1CSE', '2CSE', '4CSE', '64CSE', '8CSE']
    f = open('mm10.main.nochrM.chrom.sizes')
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
                Bcp = con_ran(cell_id,type,chr,max_len,cell_dict)
                cell_dict[str(cell_number)][chr] = Bcp
    out_path = './features/sbcp/sbcp.npy'
    np.save(out_path,cell_dict)
if __name__ == '__main__':
    main()
    # 特征思路：  (1):当前接触数在指定TAD域中的概率 (2) bin在对角线上的概率  （3）：线性邻居+空间邻居