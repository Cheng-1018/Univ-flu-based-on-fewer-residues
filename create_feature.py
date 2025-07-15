# 本脚本为seq文件中每条序列生成了物化特征矩阵以及突变0/1向量
# chem_feature(Name)函数返回该亚型抗原数据特征矩阵以及标签，维度(样本数，序列长度，8)
# mutation_feature(Name)函数返回该亚型抗原数据突变0/1向量以及标签，维度(样本数，序列长度)
# region_feature(Name)函数返回该亚型抗原数据按区域sum并展平，维度(样本数，80)
import numpy as np
from read_pdb import region


# 获取序列
def read_seq(Name):
    dict_seq = {}
    with open(f"{Name}-seq", "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            name = lines[i].replace("\n", "")
            name = name.replace(">", "")
            sequence = lines[i + 1].replace("\n", "")
            dict_seq[name] = sequence
    return dict_seq


# 寻找糖基化位点
def gly_site(sequences_dict):
    """
    :param sequences_dict: 序列比对后的字典{name：seq}
    :return: 糖基化位点{name：【0，10，20】}索引从零开始
    """
    gly_dict = {}
    for name, post_seq in sequences_dict.items():
        pre_seq = post_seq.replace('-', '')  #将-去掉得到本来的序列
        ys_dict = {}  # 映射字典{原序：比对后序}
        j = 0
        for i in range(len(post_seq)):
            if post_seq[i] != '-':
                ys_dict[j] = i
                j += 1
        # print(ys_dict)
        sites = []
        for j in range(len(pre_seq) - 2):
            triplet = pre_seq[j:j + 3]
            if triplet[0] == 'N' and triplet[2] in ['S', 'T'] and triplet[1] != 'P':
                sites.append(ys_dict[j])
        gly_dict[name] = sites
    return gly_dict


# 氨基酸物化特征,对‘-’位置填充0
amino_feat1 = {
    "A": 0.25,
    "R": -1.76,
    "N": -0.64,
    "D": -0.72,
    "C": 0.04,
    "Q": -0.69,
    "E": -0.62,
    "G": 0.16,
    "H": -0.40,
    "I": 0.73,
    "K": -1.10,
    "L": 0.53,
    "M": 0.26,
    "F": 0.61,
    "P": -0.07,
    "S": -0.26,
    "T": -0.18,
    "W": 0.37,
    "Y": 0.02,
    "V": 0.54,
    '-': 0
}
amino_feat2 = {
    'A': 12.28,
    'L': 14.10,
    'R': 11.49,
    'K': 10.80,
    'N': 11.00,
    'M': 14.33,
    'D': 10.97,
    'F': 13.43,
    'C': 14.93,
    'P': 11.19,
    'Q': 11.28,
    'S': 11.26,
    'E': 11.19,
    'T': 11.65,
    'G': 12.01,
    'W': 12.95,
    'H': 12.84,
    'Y': 13.29,
    'I': 14.77,
    'V': 15.07,
    '-': 0
}
amino_feat3 = {
    'A': 0.61,
    'L': 1.53,
    'R': 0.60,
    'K': 1.15,
    'N': 0.06,
    'M': 1.18,
    'D': 0.46,
    'F': 2.02,
    'C': 1.07,
    'P': 1.95,
    'Q': 0.00,
    'S': 0.05,
    'E': 0.47,
    'T': 0.05,
    'G': 0.07,
    'W': 2.65,
    'H': 0.61,
    'Y': 1.88,
    'I': 2.22,
    'V': 1.32,
    '-': 0
}
amino_feat4 = {
    'A': 0.75,
    'L': 2.40,
    'R': 0.75,
    'K': 1.50,
    'N': 0.69,
    'M': 1.30,
    'D': 0.00,
    'F': 2.65,
    'C': 1.00,
    'P': 2.60,
    'Q': 0.59,
    'S': 0.00,
    'E': 0.00,
    'T': 0.45,
    'G': 0.00,
    'W': 3.00,
    'H': 0.00,
    'Y': 2.85,
    'I': 2.95,
    'V': 1.70,
    '-': 0
}
amino_feat5 = {
    'A': 1.00,
    'L': 4.00,
    'R': 6.13,
    'K': 4.77,
    'N': 2.95,
    'M': 4.43,
    'D': 2.78,
    'F': 5.89,
    'C': 2.43,
    'P': 2.72,
    'Q': 3.95,
    'S': 1.60,
    'E': 3.78,
    'T': 2.60,
    'G': 0.00,
    'W': 8.08,
    'H': 4.66,
    'Y': 6.47,
    'I': 4.00,
    'V': 3.00,
    '-': 0
}
amino_feat6 = {
    'A': 5.2,
    'L': 7.0,
    'R': 6.0,
    'K': 6.0,
    'N': 5.0,
    'M': 6.8,
    'D': 5.0,
    'F': 7.1,
    'C': 6.1,
    'P': 6.2,
    'Q': 6.0,
    'S': 4.9,
    'E': 6.0,
    'T': 5.0,
    'G': 4.2,
    'W': 7.6,
    'H': 6.0,
    'Y': 7.1,
    'I': 7.0,
    'V': 6.4,
    '-': 0
}
amino_feat7 = {
    'A': 6.00,
    'L': 5.98,
    'R': 10.76,
    'K': 9.74,
    'N': 5.41,
    'M': 5.74,
    'D': 2.77,
    'F': 5.48,
    'C': 5.05,
    'P': 6.30,
    'Q': 5.65,
    'S': 5.68,
    'E': 3.22,
    'T': 5.66,
    'G': 5.97,
    'W': 5.89,
    'H': 7.59,
    'Y': 5.66,
    'I': 6.02,
    'V': 5.96,
    '-': 0
}


# 为每条序列分配特征矩阵
def get_index_martix(Name):
    dict_seq = read_seq(Name)
    gly_dict = gly_site(dict_seq)
    index_fea = {}
    for name, seq in dict_seq.items():
        fea = []
        for i in range(len(seq)):
            res = seq[i]
            index = []
            index.append(amino_feat1[res])
            index.append(amino_feat2[res])
            index.append(amino_feat3[res])
            index.append(amino_feat4[res])
            index.append(amino_feat5[res])
            index.append(amino_feat6[res])
            index.append(amino_feat7[res])
            if i in gly_dict[name]:
                index.append(1)
            else:
                index.append(0)
            fea.append(index)
        index_fea[name] = np.array(fea)
    return index_fea


# 读取抗原关系文件，每对序列计算特征矩阵的绝对差值
def chem_feature(Name):
    index_fea = get_index_martix(Name)
    relation = []
    feature = []
    with open(f"{Name}-antigen", "r") as file:
        lines = file.readlines()
        for i in range(len(lines)):
            parts = lines[i].strip().split("\t")
            name1 = parts[0]
            name2 = parts[1]
            feature1 = index_fea[name1]
            feature2 = index_fea[name2]
            feature.append(abs(feature1 - feature2))
            relation.append(int(parts[2]))
    # feature = [matrix.flatten() for matrix in feature]#展平
    return np.array(feature), np.array(relation)


# 对一对序列，根据突变返回0/1向量，name1，2为病毒株名称
def mutation(name1, name2, sequences):
    seq1 = sequences[name1]
    seq2 = sequences[name2]
    vector = []
    for i in range(len(seq1)):
        if seq1[i] == seq2[i]:
            vector.append(1)
        else:
            vector.append(0)
    return vector


def mutation_feature(Name):
    sequences = read_seq(Name)
    relation = []
    feature = []
    with open(rf"{Name}-antigen", "r") as file:
        lines = file.readlines()
        for i in range(len(lines)):
            parts = lines[i].strip().split("\t")
            name1 = parts[0]
            name2 = parts[1]
            vector = mutation(name1, name2, sequences)
            feature.append(vector)
            relation.append(int(parts[2]))
    # feature = [matrix.flatten() for matrix in feature]#展平
    return np.array(feature), np.array(relation)


def region_feature(Name):
    index_fea = get_index_martix(Name)
    relation = []
    feature = []
    with open(f"{Name}-antigen", "r") as file:
        lines = file.readlines()
        for i in range(len(lines)):
            parts = lines[i].strip().split("\t")
            name1 = parts[0]
            name2 = parts[1]
            feature1 = index_fea[name1]
            feature2 = index_fea[name2]
            feature.append(abs(feature1 - feature2))
            relation.append(int(parts[2]))
    # feature = [matrix.flatten() for matrix in feature]#展平
    region_dict = region(Name)
    new_fea = np.zeros((feature.shape[0], 10, 8))
    for region_id, list in region_dict.items():
        region_fea = feature[:, list, :]
        region_fea = np.sum(region_fea, axis=1)
        new_fea[:, region_id - 1, :] = region_fea
    new_fea = new_fea.reshape(new_fea.shape[0], -1)
    return new_fea, np.array(relation)
