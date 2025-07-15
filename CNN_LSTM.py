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

H1_list = [224, 181, 182, 186, 222, 223, 225, 180, 187, 183, 179, 190, 132, 221, 189, 226, 90, 134, 185, 215, 133, 130,
           213, 214, 184, 91, 218, 131, 246, 216, 188, 220, 219, 191, 149, 151, 178, 92, 211, 135, 212, 217, 248, 129,
           89, 227, 143, 247, 150, 245, 141, 192, 177, 142, 152, 194, 128, 136, 88, 193, 148, 196, 144, 93, 228, 210,
           126, 249, 195, 127, 147, 157, 140, 197, 244, 176, 198, 153, 155, 154, 209, 94, 63, 243, 125, 137, 156, 87,
           64, 61, 229, 145, 250, 139, 146, 62, 86, 58, 95, 158, 175, 57, 160, 251, 65, 199, 159, 138, 208, 242, 122,
           124, 241, 85, 118, 230, 200, 60, 59, 207, 66, 252, 97, 123, 161, 174, 116, 55, 54, 96, 121, 56, 162, 119,
           117, 253, 84, 240, 231, 68, 115, 201, 67, 53, 114, 120, 206, 100, 98, 163, 173, 239, 83, 205, 101, 254, 99,
           69, 164, 113, 112, 232, 51, 202, 52, 80, 172, 81, 82, 233, 104, 238, 70, 102, 71, 103, 204, 255, 165, 111,
           203, 266, 50, 79, 105, 78, 110, 264, 166, 171, 49, 256, 237, 109, 234, 265, 267, 108, 72, 268, 263, 106, 107,
           168, 258, 167, 77, 235, 170, 257, 236, 75, 74, 48, 73, 262, 169, 269, 259, 76, 261, 270, 271, 281, 47, 260,
           40, 297, 282, 299, 46, 298, 280, 272, 42, 41, 39, 279, 43, 283, 296, 295, 44, 273, 300, 38, 284, 45, 278,
           274, 275, 276, 294, 37, 301, 277, 302, 36, 285, 305, 293, 304, 303, 292, 307, 306, 35, 286, 287, 34, 291, 33,
           308, 309, 290, 32, 289, 288, 310, 31, 311, 30, 312, 29, 313, 16, 15, 28, 14, 13, 21, 314, 17, 18, 22, 315,
           12, 27, 20, 23, 19, 24, 25, 11, 316, 26, 10, 317, 318, 319, 9, 8, 320, 321, 7, 322, 323, 324, 325, 326, 6, 5,
           4, 3, 2, 1, 0]
H3_list = [227, 184, 185, 189, 225, 226, 228, 183, 182, 186, 190, 193, 135, 97, 229, 218, 192, 188, 137, 98, 216, 217,
           224, 187, 136, 249, 134, 221, 219, 191, 133, 152, 194, 222, 99, 154, 181, 223, 138, 215, 96, 251, 214, 146,
           220, 153, 230, 250, 248, 198, 144, 145, 180, 195, 155, 100, 147, 197, 132, 151, 196, 139, 95, 231, 150, 131,
           130, 252, 213, 199, 179, 247, 94, 160, 158, 72, 143, 156, 129, 200, 101, 201, 212, 148, 140, 246, 70, 159,
           73, 232, 149, 71, 253, 157, 142, 67, 102, 178, 161, 93, 254, 162, 128, 163, 126, 141, 211, 74, 66, 202, 245,
           233, 69, 104, 244, 68, 203, 75, 127, 103, 210, 177, 64, 63, 76, 255, 124, 92, 164, 165, 125, 65, 107, 234,
           62, 256, 108, 243, 105, 121, 123, 204, 122, 176, 91, 209, 79, 106, 257, 78, 166, 119, 242, 77, 117, 167, 120,
           111, 235, 90, 60, 61, 88, 89, 109, 80, 205, 110, 208, 175, 118, 236, 112, 258, 241, 116, 81, 168, 259, 268,
           86, 87, 59, 266, 114, 169, 207, 115, 206, 174, 58, 113, 267, 82, 237, 240, 269, 270, 265, 83, 260, 85, 238,
           170, 173, 264, 171, 239, 57, 261, 84, 271, 272, 283, 172, 273, 56, 262, 263, 299, 284, 50, 301, 282, 300, 52,
           55, 285, 281, 49, 298, 274, 297, 51, 48, 302, 286, 53, 275, 54, 280, 296, 279, 276, 46, 47, 278, 287, 303,
           304, 307, 277, 295, 306, 45, 305, 294, 308, 309, 288, 44, 289, 43, 293, 311, 310, 42, 292, 290, 291, 312, 41,
           313, 40, 314, 39, 315, 26, 38, 25, 31, 316, 24, 27, 28, 23, 32, 317, 37, 30, 29, 33, 22, 34, 35, 318, 21, 36,
           319, 320, 321, 20, 19, 18, 322, 323, 17, 16, 324, 325, 326, 327, 328, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
           4, 0, 1, 2, 3]
H5_list = [223, 180, 181, 222, 185, 221, 224, 179, 178, 182, 186, 189, 131, 225, 90, 188, 220, 212, 184, 91, 129, 133,
           217, 130, 245, 132, 213, 183, 214, 187, 215, 190, 218, 219, 92, 148, 177, 211, 150, 216, 210, 134, 128, 226,
           142, 247, 246, 89, 149, 140, 244, 141, 191, 176, 151, 193, 88, 93, 135, 192, 143, 147, 126, 227, 195, 209,
           127, 194, 125, 248, 196, 146, 139, 156, 154, 243, 63, 175, 94, 152, 197, 208, 124, 242, 155, 87, 144, 136,
           64, 61, 228, 153, 58, 249, 138, 95, 145, 62, 86, 57, 65, 157, 174, 250, 159, 158, 85, 198, 207, 137, 118,
           241, 121, 123, 229, 60, 59, 66, 240, 97, 55, 199, 96, 251, 54, 206, 122, 84, 56, 116, 119, 160, 173, 117, 67,
           120, 161, 83, 252, 53, 230, 70, 239, 98, 115, 200, 100, 114, 162, 69, 172, 205, 238, 253, 68, 101, 99, 204,
           51, 163, 82, 52, 81, 231, 112, 113, 71, 80, 201, 232, 171, 104, 237, 102, 254, 72, 103, 110, 203, 164, 50,
           111, 79, 263, 49, 265, 165, 78, 202, 255, 267, 105, 170, 233, 236, 266, 264, 109, 107, 73, 234, 262, 269,
           106, 268, 169, 108, 166, 167, 256, 48, 77, 75, 74, 235, 257, 270, 261, 168, 280, 76, 47, 259, 258, 40, 260,
           296, 281, 271, 279, 46, 41, 298, 42, 297, 39, 282, 272, 278, 295, 43, 294, 38, 283, 273, 45, 299, 277, 274,
           44, 293, 275, 37, 276, 36, 284, 304, 300, 301, 292, 306, 305, 303, 35, 291, 302, 285, 34, 307, 290, 33, 308,
           286, 32, 289, 288, 287, 309, 310, 31, 30, 311, 29, 312, 16, 15, 28, 18, 21, 14, 17, 22, 13, 313, 20, 314, 19,
           23, 12, 27, 24, 25, 11, 315, 26, 316, 317, 10, 318, 9, 319, 8, 7, 6, 5, 4, 3, 2, 1, 0]

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
