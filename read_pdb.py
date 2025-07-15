# 该脚本实现了从PDB中读取三维结构信息并计算各氨基酸距离，再将其映射到多序列比对后的序列中，对缺失部分的距离信息进行线性插值
# all(Name) 函数接收亚型名称，返回补全的距离字典{res_id:distance}
# all(Name) 函数接收亚型名称，返回分区字典{0:[190,287....}
import numpy as np
import Bio.PDB

# 氨基酸三字母到单字母表示的映射
amino_acids = {
    'ALA': 'A',  # 丙氨酸 (Alanine)
    'ARG': 'R',  # 精氨酸 (Arginine)
    'ASN': 'N',  # 天冬酰胺 (Asparagine)
    'ASP': 'D',  # 天冬氨酸 (Aspartic acid)
    'CYS': 'C',  # 半胱氨酸 (Cysteine)
    'GLN': 'Q',  # 谷氨酰胺 (Glutamine)
    'GLU': 'E',  # 谷氨酸 (Glutamic acid)
    'GLY': 'G',  # 甘氨酸 (Glycine)
    'HIS': 'H',  # 组氨酸 (Histidine)
    'ILE': 'I',  # 异亮氨酸 (Isoleucine)
    'LEU': 'L',  # 亮氨酸 (Leucine)
    'LYS': 'K',  # 赖氨酸 (Lysine)
    'MET': 'M',  # 甲硫氨酸 (Methionine)
    'PHE': 'F',  # 苯丙氨酸 (Phenylalanine)
    'PRO': 'P',  # 脯氨酸 (Proline)
    'SER': 'S',  # 丝氨酸 (Serine)
    'THR': 'T',  # 苏氨酸 (Threonine)
    'TRP': 'W',  # 色氨酸 (Tryptophan)
    'TYR': 'Y',  # 酪氨酸 (Tyrosine)
    'VAL': 'V'  # 缬氨酸 (Valine)
}


# 读取PDB文件
def read_pdb(Name):
    '''

    :param Name:
    :return: {11: ['ASP', array([-20.541,  43.585, -67.855], dtype=float32)],
    12: ['THR', array([-18.857,  44.085, -64.472], dtype=float32)]}
    '''
    if Name == 'H1N1':
        Name = '3LZG'
    elif Name == 'H3N2':
        Name = '6AOU'
    else:
        Name = '2IBX'
    # 加载PDB文件
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(Name, f"{Name}.pdb.txt")

    # 获取A链
    chain = structure[0]['A']
    # 获取每个氨基酸的ID，名称，CA坐标
    res_dict = {}
    for residue in chain:
        res_name = residue.get_resname()
        res_id = residue.get_id()[1]
        for atom in residue:
            atom_name = atom.get_name()
            atom_coord = atom.get_coord()
            if atom_name == 'CA':
                res_dict[res_id] = [res_name, atom_coord]  # 11: ['ASP', array([-20.541,  43.585, -67.855]
    # 排序
    res_dict = {k: res_dict[k] for k in sorted(res_dict)}
    return res_dict


# 获取RBS区域坐标并确定抗原中心，计算每个氨基酸到中心的距离,返回序号与距离的字典
def RBS(res_dict):
    """

    :param res_dict: 字典{'pdb顺序'：【'三字母缩写'，【X,Y,Z】】}
    :return: 字典【region；【PDB顺序】】

    """
    rbs_region = {'130_loop': [i for i in range(133, 139)], '190_helix': [i for i in range(187, 197)],
                  '220_loop': [i for i in range(220, 230)]}

    rbs_coord = []
    for region, id in rbs_region.items():
        for i in id:
            rbs_coord.append(res_dict[i][1])
            # print(f'{region} {i} {res_dict[i][0]} {res_dict[i][1]}')
    # 确定抗原中心
    core_all = np.mean(rbs_coord, axis=0)

    # 各区域中心
    core_130 = np.mean([rbs_coord[i] for i in range(0, 6)], axis=0)
    core_190 = np.mean([rbs_coord[i] for i in range(6, 16)], axis=0)
    core_220 = np.mean([rbs_coord[i] for i in range(16, 26)], axis=0)

    # 抗原中心到各区域距离

    dist_130 = np.linalg.norm(core_all - core_130)
    # print(f'130-core {dist_130}')
    dist_190 = np.linalg.norm(core_all - core_190)
    # print(f'190-core {dist_190}')
    dist_220 = np.linalg.norm(core_all - core_220)
    # print(f'220-core {dist_220}')

    # 计算氨基酸与抗原中心的距离，返回字典{res_id:dist}
    dist_dict = {}
    for res_id, value in res_dict.items():
        dist = np.linalg.norm(core_all - value[1])
        dist_dict[res_id] = dist
    return dist_dict


# 读取多序列比对后的序列文件
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


# 比对后的序列中含有‘-’，建立原始PDB序列到比对后的序列的映射序号字典map_dict
def mapping(res_dict, sequences_dict):
    '''

    :param res_dict:
    :param sequences_dict:
    :param region_dict:
    :return: 映射字典{PDB顺序：比对顺序}{region：【比对顺序-从0开始】}
    '''

    pre_seq = ''  # 原始模板序列
    order = []  # 原始顺序
    for id, value in res_dict.items():
        pre_seq += amino_acids[value[0]]
        order.append(id)
    post_seq = sequences_dict['template']  #比对后的模板序列
    # print(post_seq)
    map_dict = {}  #映射字典
    j = 0
    for i in range(len(post_seq)):
        if post_seq[i] != '-':
            map_dict[order[j]] = i
            j += 1
    # print(f'映射字典PDB顺序-对比后顺序{map_dict}')
    return map_dict


def linear_interpolation(dictionary, n):
    """
    使用线性插值补全字典中缺失的键值

    参数:
        dictionary: 输入的字典，键为0到n的整数，部分缺失
        n: 最大的键值

    返回:
        补全后的完整字典
    """
    # 创建完整的键范围
    full_keys = range(n + 1)

    # 初始化结果字典
    result = {}

    # 遍历所有键
    for key in full_keys:
        if key in dictionary:
            # 如果键存在，直接使用原值
            result[key] = dictionary[key]
        else:
            # 寻找前一个存在的键
            prev_key = key - 1
            while prev_key >= 0 and prev_key not in result:
                prev_key -= 1

            # 寻找后一个存在的键
            next_key = key + 1
            while next_key <= n and next_key not in dictionary:
                next_key += 1

            # 根据前后键是否存在进行插值
            if prev_key >= 0 and next_key <= n:
                # 前后都有值，进行线性插值
                prev_val = result[prev_key]
                next_val = dictionary[next_key]

                # 计算插值
                slope = (next_val - prev_val) / (next_key - prev_key)
                interpolated_val = prev_val + slope * (key - prev_key)
                result[key] = interpolated_val
            elif prev_key >= 0:
                # 只有前一个值，使用前一个值
                result[key] = result[prev_key]
            elif next_key <= n:
                # 只有后一个值，使用后一个值
                result[key] = dictionary[next_key]
            else:
                # 没有前后值（理论上不会发生，因为n是最大键）
                result[key] = 0  # 默认值

    return result


def divide_into_regions(original_dict, region_size=4, num_regions=10):
    regions = {i: [] for i in range(1, num_regions + 1)}

    for serial, distance in original_dict.items():
        if distance >= num_regions * region_size:
            continue  # 忽略超过最大区域距离的值
        region = min((distance // region_size) + 1, num_regions)
        regions[region].append(serial)

    return regions


def region(Name):
    original_dict = all(Name)
    return divide_into_regions(original_dict)


def all(Name):
    res_dict = read_pdb(Name)  # {11: ['ASP', array([-20.541,  43.585, -67.855], dtype=float32)]}
    dist_dict = RBS(res_dict)  # pdb id : dist {11: 104.65737, 12: 101.287254}
    sequences_dict = read_seq(Name)  # 与模板序列对序列比对后的seq
    seq_len = len(sequences_dict['template'])
    map_dict = mapping(res_dict, sequences_dict)  # PDB id:sequence id{11: 0, 12: 1, 13: 2}
    dist = {}
    for id, dis in dist_dict.items():
        try:
             dist[map_dict[id]] = dis
        except KeyError:
            pass
    dist = linear_interpolation(dist, seq_len - 1)  # 缺失的距离信息线性插值
    # print(dist)
    # print(len(dist))
    return dist
