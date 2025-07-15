import numpy as np
from sklearn.naive_bayes import GaussianNB
from create_feature import chem_feature
from read_pdb import all
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer  # 用于数据分布调整
from sklearn.pipeline import Pipeline
import csv


# #单亚型内交叉验证

def five_fold(X, y, n_splits):
    # 初始化StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc, recall, precision, f1, auc = 0, 0, 0, 0, 0
    # 进行交叉验证
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_X, train_y = X[train_idx], y[train_idx]
        test_X, test_y = X[test_idx], y[test_idx]
        gnb = GaussianNB()

        # 训练模型
        gnb.fit(train_X, train_y)
        # 在测试集上进行预测
        pred_y = gnb.predict(test_X)  # 类别预测
        pred_proba_y = gnb.predict_proba(test_X)[:, 1]  # 概率预测(用于AUC计算)
        # 计算各项指标
        acc += accuracy_score(test_y, pred_y)
        recall += recall_score(test_y, pred_y)
        precision += precision_score(test_y, pred_y, zero_division=0)
        f1 += f1_score(test_y, pred_y)
        auc += roc_auc_score(test_y, pred_proba_y)
    acc /= n_splits
    recall /= n_splits
    precision /= n_splits
    f1 /= n_splits
    auc /= n_splits

    return {'param': '', 'accuracy': acc, 'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1}


def cross_subtype(train_X, train_y, val_test_X, val_test_y):
    # 划分验证集和测试集
    val_X, test_X, val_y, test_y = train_test_split(
        val_test_X, val_test_y,
        test_size=0.5,
        random_state=42,
        shuffle=True,
        stratify=val_test_y
    )

    # GaussianNB本身参数较少，主要通过特征工程优化
    param_grid = {
        'preprocess__method': ['none', 'standard', 'yeo-johnson'],  # 预处理方法
        'clf__var_smoothing': np.logspace(-12, -2, 20)  # 唯一可调的核心参数
    }

    max_auc_acc = 0
    # 初始化记录指标
    val_metrics = {'accuracy': 0, 'auc': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    test_metrics = {'accuracy': 0, 'auc': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    best_param = ''

    for method in param_grid['preprocess__method']:
        for var_smoothing in param_grid['clf__var_smoothing']:
            # 构建预处理管道
            if method == 'none':
                preprocessor = Pipeline([('identity', 'passthrough')])
            elif method == 'standard':
                preprocessor = Pipeline([('scaler', StandardScaler())])
            else:
                preprocessor = Pipeline([('transformer', PowerTransformer(method='yeo-johnson'))])

            model = Pipeline([
                ('preprocess', preprocessor),
                ('clf', GaussianNB(var_smoothing=var_smoothing))
            ])

            model.fit(train_X, train_y)

            # 验证集评估
            pred_y = model.predict(val_X)
            pred_proba = model.predict_proba(val_X)[:, 1]

            acc = accuracy_score(val_y, pred_y)
            precision = precision_score(val_y, pred_y, zero_division=0)
            recall = recall_score(val_y, pred_y)
            f1 = f1_score(val_y, pred_y)
            auc = roc_auc_score(val_y, pred_proba)

            # 更新最佳参数
            if (auc + acc) / 2 > max_auc_acc:
                max_auc_acc = (auc + acc) / 2
                best_param = f'preprocess={method}_var_smoothing={var_smoothing:.2e}'

                # 记录验证集指标
                val_metrics = {
                    'accuracy': acc,
                    'auc': auc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

                # 记录测试集指标
                test_pred = model.predict(test_X)
                test_proba = model.predict_proba(test_X)[:, 1]
                test_metrics = {
                    'accuracy': accuracy_score(test_y, test_pred),
                    'auc': roc_auc_score(test_y, test_proba),
                    'precision': precision_score(test_y, test_pred, zero_division=0),
                    'recall': recall_score(test_y, test_pred),
                    'f1': f1_score(test_y, test_pred)
                }

    return {'param': best_param, **val_metrics}, {'param': best_param, **test_metrics}


# 由距离排序后的氨基酸索引
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

# 获取特征矩阵
H1_X, H1_y = chem_feature('H1N1')
H3_X, H3_y = chem_feature('H3N2')
H5_X, H5_y = chem_feature('H5N1')


def all_experiment(num, H1_X, H1_y, H3_X, H3_y, H5_X, H5_y):
    # 取前n个位置,再按原来的相对顺序
    H1_X, H1_y = H1_X[:, sorted(H1_list[0:num]), :], H1_y
    H3_X, H3_y = H3_X[:, sorted(H3_list[0:num]), :], H3_y
    H5_X, H5_y = H5_X[:, sorted(H5_list[0:num]), :], H5_y

    # 序列维度取平均
    H1_X = np.mean(H1_X, axis=1)
    H3_X = np.mean(H3_X, axis=1)
    H5_X = np.mean(H5_X, axis=1)

    feature_list = [H1_X, H3_X, H5_X]
    label_list = [H1_y, H3_y, H5_y]
    # 单亚型的交叉验证
    subtype = {0: 'H1', 1: 'H3', 2: 'H5'}  # 亚型映射
    cross_stype = {0: 'H3H5->H1', 1: 'H1H5->H3', 2: 'H1H3->H5'}
    all_metric = []
    # 单亚型五折交叉验证
    for i in range(3):
        metric = five_fold(feature_list[i], label_list[i], 5)
        metric['model'] = f"{subtype[i]}->self"
        all_metric.append(metric)
    for i in range(3):
        for j in range(3):
            if i != j:
                val_metric, test_metric = cross_subtype(feature_list[i], label_list[i],
                                                        feature_list[j], label_list[j])
                val_metric['model'] = f'{subtype[i]}->{subtype[j]} 50%val'
                test_metric['model'] = f'{subtype[i]}->{subtype[j]} 50%test'
                all_metric.append(val_metric)
                all_metric.append(test_metric)
    for i in range(3):
        trainX_list = [feature_list[j] for j in range(3) if j != i]
        trainy_list = [label_list[j] for j in range(3) if j != i]

        train_X = np.vstack(trainX_list)
        train_y = np.concatenate(trainy_list)

        test_X = feature_list[i]
        test_y = label_list[i]

        val_metric, test_metric = cross_subtype(train_X, train_y, test_X, test_y)
        val_metric['model'] = f'{cross_stype[i]} 50%val'
        test_metric['model'] = f'{cross_stype[i]} 50%test'
        all_metric.append(val_metric)
        all_metric.append(test_metric)
    # 输出所有实验的评估指标
    print(f'algorithm,num,model,param,accuracy,precision,recall,f1,auc')
    for metric in all_metric:
        str = f'byes,{num},{metric["model"]},{metric["param"]},{metric["accuracy"]:.4f},{metric["precision"]:.4f},{metric["recall"]:.4f},{metric["f1"]:.4f},{metric["auc"]:.4f}'
        print(str)
        save_list.append(str)


all_experiment(100, H1_X, H1_y, H3_X, H3_y, H5_X, H5_y)
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
