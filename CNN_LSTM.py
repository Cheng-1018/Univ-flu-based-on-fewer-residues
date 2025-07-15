#导入包
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from create_feature import chem_feature
from read_pdb import all
import csv

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')
# 如果使用 GPU，确保 PyTorch 能够看到所有可用的 GPU
if device.type == 'cuda':
    print(torch.cuda.device_count(), "个 GPU 可用")
    print('当前使用的 GPU:', torch.cuda.get_device_name(0))


def set_seed(seed=42):
    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# 使用
set_seed(42)


# 封装数据
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(np.array(X))  # 转为Tensor
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)  # 必须实现：返回数据总量

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # 必须实现：返回单个样本.unsqueeze(0)


class ProteinMutationPredictor(nn.Module):
    def __init__(self, cnn_channels=64, lstm_hidden=64):
        super().__init__()

        # 1D CNN 提取局部特征
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=cnn_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # BiLSTM 捕捉序列依赖
        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入x形状: [batch_size, seq_len, 8]
        x = x.permute(0, 2, 1)  # -> [batch, 8, seq_len]
        cnn_features = self.cnn(x)  # -> [batch, cnn_channels*2, seq_len//2]
        cnn_features = cnn_features.permute(0, 2, 1)  # -> [batch, seq_len//2, cnn_channels*2]

        lstm_out, _ = self.lstm(cnn_features)  # -> [batch, seq_len//2, lstm_hidden*2]
        pooled = lstm_out.mean(dim=1)  # -> [batch, lstm_hidden*2]

        return self.classifier(pooled)  # -> [batch, 1]


# 一个train_dataloader的训练函数
def train_func(train_dataloader, model, optimizer, criterion):
    """
    训练模型的一个epoch，并计算训练指标。

    参数:
    - train_dataloader: 训练数据的DataLoader
    - model: 要训练的模型
    - optimizer: 优化器
    - criterion: 损失函数

    返回:
    - 一个包含训练指标（损失、准确率、精确率、召回率、F1分数、AUC）的字典
    """
    model.train()  # 设置模型为训练模式
    all_labels = []  # 存储所有真实标签
    all_probs = []  # 存储所有预测概率
    all_loss = 0  # 累计损失

    for feature, label in train_dataloader:
        feature, label = feature.to(device), label.to(device)  # 将数据移动到 GPU
        optimizer.zero_grad()  # 梯度清零
        outputs = model(feature)  # 前向传播
        loss = criterion(outputs.squeeze(), label)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        all_loss += loss.item()  # 累计损失

        prob = outputs.detach().cpu().numpy()  # 将预测结果转移到CPU并转为numpy数组
        all_probs.extend(prob)  # 存储预测概率
        all_labels.extend(label.cpu().numpy())  # 存储真实标签

    avg_loss = all_loss / len(train_dataloader)  # 计算平均损失
    all_preds = np.array(all_probs) > 0.5  # 将预测概率转为二分类结果

    # 计算各种评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


# 一个val_dataloader的评测函数
def eval_func(val_dataloader, model, criterion):
    """
    评估模型在一个epoch上的验证集性能。

    参数:
    - val_dataloader: 验证数据的DataLoader
    - model: 要评估的模型
    - optimizer: 优化器
    - criterion: 损失函数

    返回:
    - 一个包含验证指标（损失、准确率、精确率、召回率、F1分数、AUC）的字典
    """
    model.eval()  # 设置模型为评估模式
    all_labels = []  # 存储所有真实标签
    all_probs = []  # 存储所有预测概率
    all_loss = 0  # 累计损失

    with torch.no_grad():  # 无需计算梯度
        for feature, label in val_dataloader:
            feature, label = feature.to(device), label.to(device)  # 将数据移动到 GPU
            outputs = model(feature)  # 前向传播
            loss = criterion(outputs.squeeze(), label)  # 计算损失
            all_loss += loss.item()  # 累计损失

            prob = outputs.detach().cpu().numpy()  # 将预测结果转移到CPU并转为numpy数组
            all_probs.extend(prob)  # 存储预测概率
            all_labels.extend(label.cpu().numpy())  # 存储真实标签

    all_preds = np.array(all_probs) > 0.5  # 将预测概率转为二分类结果
    avg_loss = all_loss / len(val_dataloader)  # 计算平均损失

    # 计算各种评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


# 生成五折交叉验证的索引
def five_fold_generate(X, y, n_splits):
    # 初始化StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    index_dict = {}
    # 进行交叉验证
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        index_dict[f'{fold}train'] = train_idx
        index_dict[f'{fold}test'] = test_idx
    # 返回每折使用的索引
    return index_dict


# 一折训练的函数
def single_five_fold(i, j, feature_lists, label_lists, fold_index):
    """
        进行单亚型五折交叉验证

        参数:
        - i: 训练数据的亚型索引
        - j: 折数
        返回:
        - 最佳评估指标
        """
    subtype = {0: 'H1', 1: 'H3', 2: 'H5'}  # 亚型映射
    fea = feature_lists[i]  # 训练特征
    lab = label_lists[i]  # 训练标签

    train_dataset = CustomDataset(fea[fold_index[f'{j}train']], lab[fold_index[f'{j}train']])
    val_dataset = CustomDataset(fea[fold_index[f'{j}test']], lab[fold_index[f'{j}test']])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    # 模型和优化器初始化
    model = ProteinMutationPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    # 记录每个epoch的评估结果
    all_train_metric = {}
    all_val_metric = {}
    epochs = 20
    m = 0
    best_metric = {}
    early_num = 0
    for k in range(epochs):
        train_metric = train_func(train_dataloader, model, optimizer, criterion)  # 训练一个epoch
        val_metric = eval_func(val_dataloader, model, criterion)  # 评估一个epoch
        val_metric['epoch'] = k + 1
        val_loss = val_metric['loss']
        all_train_metric[k] = train_metric
        all_val_metric[k] = val_metric

        scheduler.step(val_loss)  # 根据验证损失调整学习率
        # print(f'{subtype[i]}->self{j} epoch{k + 1} val{val_metric}')  # 输出当前epoch的验证结果
        auc_acc_mean = (val_metric['accuracy'] + val_metric['auc']) / 2  # 计算AUC和准确率的平均

        if auc_acc_mean > m:
            m = auc_acc_mean
            best_metric[f'{subtype[i]}->self{j + 1}fold'] = val_metric  # 更新最佳指标
            early_num = 0
        else:
            early_num += 1
            if early_num > 5:
                break

    # 实验完成后，清空模型和相关对象
    del model
    del optimizer
    del scheduler
    del train_dataloader
    del val_dataloader
    del fea
    del lab
    torch.cuda.empty_cache()
    gc.collect()

    return best_metric


# 单亚型-单亚型
def single_experiment(i, j, feature_lists, label_lists):
    """
    进行单亚型到单亚型的实验。

    参数:
    - i: 训练数据的亚型索引
    - j: 验证数据的亚型索引

    返回:
    - 最佳评估指标
    """
    subtype = {0: 'H1', 1: 'H3', 2: 'H5'}  # 亚型映射

    train_fea = feature_lists[i]  # 训练特征
    train_lab = label_lists[i]  # 训练标签

    # 拆分测试亚型数据集，测试集占30%
    val_fea, test_fea, val_lab, test_lab = train_test_split(
        feature_lists[j], label_lists[j],
        test_size=0.5,
        random_state=42,
        shuffle=True,
        stratify=label_lists[j]
    )

    train_dataset = CustomDataset(train_fea, train_lab)
    val_dataset = CustomDataset(val_fea, val_lab)
    test_dataset = CustomDataset(test_fea, test_lab)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    # 模型和优化器初始化
    model = ProteinMutationPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    # 记录每个epoch的评估结果
    all_train_metric = {}
    all_val_metric = {}
    epochs = 20
    m = 0
    best_val_metric = {}
    current_test_metric = {}
    early_num = 0
    for k in range(epochs):
        train_metric = train_func(train_dataloader, model, optimizer, criterion)  # 训练一个epoch
        val_metric = eval_func(val_dataloader, model, criterion)  # 验证集评估一个epoch
        val_metric['epoch'] = k + 1
        test_metric = eval_func(test_dataloader, model, criterion)  # 测试集评估一个epoch
        test_metric['epoch'] = k + 1
        val_loss = val_metric['loss']  # 验证集损失
        all_train_metric[k] = train_metric
        all_val_metric[k] = val_metric

        scheduler.step(val_loss)  # 根据验证损失调整学习率
        # print(f'{subtype[i]}->{subtype[j]} epoch{k + 1} val{val_metric}')  # 输出当前epoch的验证结果
        auc_acc_mean = (val_metric['accuracy'] + val_metric['auc']) / 2  # 根据验证集计算AUC和准确率的平均

        if auc_acc_mean > m:
            m = auc_acc_mean
            best_val_metric[f'{subtype[i]}->{subtype[j]} 60%val'] = val_metric  # 更新验证集最佳指标
            current_test_metric[f'{subtype[i]}->{subtype[j]} 40%test'] = test_metric  # 更新测试集当前指标指标
        else:
            early_num += 1
            if early_num > 5:
                break
    # 实验完成后，清空模型和相关对象
    del model
    del optimizer
    del scheduler
    del train_dataloader
    del val_dataloader
    del train_fea
    del train_lab
    del val_fea
    del val_lab
    torch.cuda.empty_cache()
    gc.collect()

    return best_val_metric, current_test_metric


# 修改后的DOUBLE ,亚型之间特征维度不统一，无法直接合并，采用分俩次训练的方式解决
def double_experiment(i, feature_lists, label_lists):
    """
    进行多亚型到单亚型的实验。

    参数:
    - i: 验证亚型的亚型索引

    返回:
    - 最佳评估指标
    """
    subtype = {0: 'H3H5->H1', 1: 'H1H5->H3', 2: 'H1H3->H5'}  # 多亚型映射
    # 提取除验证亚型外的所有训练数据
    train_fea = [feature_lists[j] for j in range(3) if j != i]

    train_lab = [label_lists[j] for j in range(3) if j != i]
    # 拆分测试亚型数据集，测试集占30%
    val_fea, test_fea, val_lab, test_lab = train_test_split(
        feature_lists[i], label_lists[i],
        test_size=0.5,
        random_state=42,
        shuffle=True,
        stratify=label_lists[i]
    )

    train_dataset0 = CustomDataset(train_fea[0], train_lab[0])
    train_dataset1 = CustomDataset(train_fea[1], train_lab[1])
    val_dataset = CustomDataset(val_fea, val_lab)
    test_dataset = CustomDataset(test_fea, test_lab)

    train_dataloader0 = DataLoader(train_dataset0, batch_size=32, shuffle=True)
    train_dataloader1 = DataLoader(train_dataset1, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    # 模型和优化器初始化
    model = ProteinMutationPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    # 记录每个epoch的评估结果
    all_train_metric = {}
    all_val_metric = {}
    epochs = 20
    m = 0
    early_num = 0
    best_val_metric = {}
    current_test_metric = {}
    for k in range(epochs):
        train_metric1 = train_func(train_dataloader0, model, optimizer, criterion)  # 训练一个epoch
        train_metric2 = train_func(train_dataloader1, model, optimizer, criterion)  # 训练一个epoch
        val_metric = eval_func(val_dataloader, model, criterion)  # 验证集评估一个epoch
        val_metric['epoch'] = k + 1
        test_metric = eval_func(test_dataloader, model, criterion)  # 测试集评估一个epoch
        test_metric['epoch'] = k + 1
        val_loss = val_metric['loss']
        all_train_metric[k] = [train_metric1, train_metric2]
        all_val_metric[k] = val_metric

        scheduler.step(val_loss)  # 根据验证损失调整学习率
        # print(f'{subtype[i]} epoch{k + 1} val{val_metric}')  # 输出当前epoch的验证结果
        auc_acc_mean = (val_metric['accuracy'] + val_metric['auc']) / 2  # 计算AUC和准确率的平均

        if auc_acc_mean > m:
            m = auc_acc_mean
            best_val_metric[f'{subtype[i]} 60%val'] = val_metric  # 更新验证集最佳指标
            current_test_metric[f'{subtype[i]} 40%test'] = test_metric  # 更新此时的测试集指标
        else:
            early_num += 1
            if early_num > 5:
                break

    # 实验完成后，清空模型和相关对象
    del model
    del optimizer
    del scheduler
    del train_dataloader0
    del train_dataloader1
    del val_dataloader
    del train_fea
    del train_lab
    del val_fea
    del val_lab
    torch.cuda.empty_cache()
    gc.collect()

    return best_val_metric, current_test_metric


# 由距离排序后的索引

H1_dist = all('H1N1')
H3_dist = all('H3N2')
H5_dist = all('H5N1')

# 按照距离排序字典
H1_list = [k for k, v in sorted(H1_dist.items(), key=lambda item: item[1])]
H3_list = [k for k, v in sorted(H3_dist.items(), key=lambda item: item[1])]
H5_list = [k for k, v in sorted(H5_dist.items(), key=lambda item: item[1])]

intersection_H1 = list(set(sorted(H1_list[0:320])[0:190]) & set(H1_list[0:90]))
intersection_H3 = list(set(sorted(H3_list[0:320])[0:190]) & set(H3_list[0:90]))
intersection_H5 = list(set(sorted(H5_list[0:320])[0:190]) & set(H5_list[0:90]))
save_list = []
H1_fea, H1_lab = chem_feature('H1N1')
H3_fea, H3_lab = chem_feature('H3N2')
H5_fea, H5_lab = chem_feature('H5N1')


def try_num(num, H1_fea, H1_lab, H3_fea, H3_lab, H5_fea, H5_lab):
    # 特征和标签的获取

    # 取前n个位置,再按原来的相对顺序
    H1_fea, H1_lab = H1_fea[:, sorted(H1_list[0:num]), :], H1_lab
    H3_fea, H3_lab = H3_fea[:, sorted(H3_list[0:num]), :], H3_lab
    H5_fea, H5_lab = H5_fea[:, sorted(H5_list[0:num]), :], H5_lab

    feature_lists = [H1_fea, H3_fea, H5_fea]  # 存储不同亚型的特征
    label_lists = [H1_lab, H3_lab, H5_lab]  # 存储不同亚型的标签
    # 进行所有实验
    all_metric = []

    # 单亚型的交叉验证
    subtype = {0: 'H1', 1: 'H3', 2: 'H5'}  # 亚型映射
    for i in [0, 1, 2]:
        fold_index = five_fold_generate(feature_lists[i], label_lists[i], 5)  # 生成该亚型五折交叉的索引，按类别均衡
        best_metric = {f'{subtype[i]}->self': {}}  # 存储该亚型五折交叉平均指标
        avg_loss = 0
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        auc = 0
        for j in range(5):
            best_metric0 = single_five_fold(i, j, feature_lists, label_lists, fold_index)
            avg_loss += best_metric0[f'{subtype[i]}->self{j + 1}fold']['loss']
            accuracy += best_metric0[f'{subtype[i]}->self{j + 1}fold']['accuracy']
            precision += best_metric0[f'{subtype[i]}->self{j + 1}fold']['precision']
            recall += best_metric0[f'{subtype[i]}->self{j + 1}fold']['recall']
            f1 += best_metric0[f'{subtype[i]}->self{j + 1}fold']['f1']
            auc += best_metric0[f'{subtype[i]}->self{j + 1}fold']['auc']
            all_metric.append(best_metric0)
        avg_loss /= 5
        accuracy /= 5
        precision /= 5
        recall /= 5
        f1 /= 5
        auc /= 5
        best_metric[f'{subtype[i]}->self']['loss'] = avg_loss
        best_metric[f'{subtype[i]}->self']['accuracy'] = accuracy
        best_metric[f'{subtype[i]}->self']['precision'] = precision
        best_metric[f'{subtype[i]}->self']['recall'] = recall
        best_metric[f'{subtype[i]}->self']['f1'] = f1
        best_metric[f'{subtype[i]}->self']['auc'] = auc
        best_metric[f'{subtype[i]}->self']['epoch'] = 'average'
        #存储起来
        all_metric.append(best_metric)

    # 单亚型到单亚型，测试亚型划分出50%做验证
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            if i != j:
                best_val_metric1, current_test_metric1 = single_experiment(i, j, feature_lists,
                                                                           label_lists)  # 进行单亚型到单亚型的实验
                all_metric.append(best_val_metric1)
                all_metric.append(current_test_metric1)

    # 双亚型到单亚型，测试亚型划分出50做验证
    for i in [0, 1, 2]:
        best_val_metric2, current_test_metric2 = double_experiment(i, feature_lists, label_lists)  # 进行多亚型到单亚型的实验
        all_metric.append(best_val_metric2)
        all_metric.append(current_test_metric2)
    # 输出所有实验的评估指标
    print(f'取前{num}个，model,epoch,accuracy,precision,recall,f1,auc')
    for metric in all_metric:
        for model, metr in metric.items():
            str = f'CNN_BiLSTM,{num},{model},{metr["epoch"]},{metr["accuracy"]:.4f},{metr["precision"]:.4f},{metr["recall"]:.4f},{metr["f1"]:.4f},{metr["auc"]:.4f}'
            print(str)
            save_list.append(str)


try_num(100, H1_fea, H1_lab, H3_fea, H3_lab, H5_fea, H5_lab)
# 准备写入CSV的数据
rows = []
for item in save_list:
    # 分割字符串为列表
    row = item.split(',')
    rows.append(row)

# 写入CSV文件
with open('results.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 写入数据
    writer.writerows(rows)

print("CSV文件已保存为 results.csv")
