from create_feature import chem_feature
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import csv


#单亚型五折交叉验证
def five_fold(X, y, n_splits):
    # 初始化StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc, recall, precision, f1, auc = 0, 0, 0, 0, 0
    # 进行交叉验证
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_X, train_y = X[train_idx], y[train_idx]
        test_X, test_y = X[test_idx], y[test_idx]
        # 创建包含标准化的管道
        model = make_pipeline(
            StandardScaler(),  # 自动处理训练/测试标准化
            svm.SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            )
        )
        model.fit(train_X, train_y)
        # 4. 预测
        pred_y = model.predict(test_X)
        pred_proba_y = model.predict_proba(test_X)[:, 1]

        # 5. 评估
        acc += accuracy_score(test_y, pred_y)
        precision += precision_score(test_y, pred_y, zero_division=0)
        recall += recall_score(test_y, pred_y)
        f1 += f1_score(test_y, pred_y)
        auc += roc_auc_score(test_y, pred_proba_y)
    acc /= n_splits
    recall /= n_splits
    precision /= n_splits
    f1 /= n_splits
    auc /= n_splits
    return {'param':'','accuracy': acc, 'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1}


# 跨亚型训练测试
def cross_subtype(train_X, train_y, val_test_X, val_test_y):
    # 训练数据为测试亚型的全部数据，测试亚型分出验证集和测试集，在验证集上调整参数，记录最好参数
    # 被测试亚型划分出50%
    val_X, test_X, val_y, test_y = train_test_split(val_test_X, val_test_y,
                                                    test_size=0.5,
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=val_test_y
                                                    )
    # 定义参数网格
    param_grid = {
        'clf__C': np.logspace(-2, 2, 5),
        'clf__gamma': np.logspace(-3, 0, 4),  # 核函数系数
        'clf__kernel': ['rbf', 'linear'],  # 核函数类型
    }
    max_auc_acc = 0
    Acc, Auc, Precision, Recall, F1 = 0, 0, 0, 0, 0  # 记录验证集指标
    Acc1, Auc1, Precision1, Recall1, F11 = 0, 0, 0, 0, 0  # 记录测试集指标
    best_param = ''
    for gamma in param_grid['clf__gamma']:
        for C in param_grid['clf__C']:
            for kernel in param_grid['clf__kernel']:
                print(f'参数{gamma}_{C}_{kernel}')
                model = make_pipeline(
                    StandardScaler(),
                    svm.SVC(
                        C=C,
                        kernel=kernel,
                        gamma=gamma,
                        class_weight=None,
                        random_state=42,
                        probability=True
                    )
                )
                model.fit(train_X, train_y)
                # 在验证集上预测
                pred_y = model.predict(val_X)
                pred_proba_y = model.predict_proba(val_X)[:, 1]

                acc = accuracy_score(val_y, pred_y)
                precision = precision_score(val_y, pred_y, zero_division=0)
                recall = recall_score(val_y, pred_y)
                f1 = f1_score(val_y, pred_y)
                auc = roc_auc_score(val_y, pred_proba_y)
                if (auc + acc) / 2 > max_auc_acc:
                    # 更新验证集最佳指标
                    max_auc_acc = (auc + acc) / 2
                    best_param = f'{gamma}_{C}_{kernel}'
                    Acc, Auc, Precision, Recall, F1 = acc, auc, precision, recall, f1

                    # 记录此时测试集指标
                    # 在测试集上预测
                    pred_y = model.predict(test_X)
                    pred_proba_y = model.predict_proba(test_X)[:, 1]

                    Acc1 = accuracy_score(test_y, pred_y)
                    Precision1 = precision_score(test_y, pred_y, zero_division=0)
                    Recall1 = recall_score(test_y, pred_y)
                    F11 = f1_score(test_y, pred_y)
                    Auc1 = roc_auc_score(test_y, pred_proba_y)
    val_metric = {'param': best_param, 'accuracy': Acc, 'auc': Auc, 'precision': Precision, 'recall': Recall, 'f1': F1}
    test_metric = {'param': best_param, 'accuracy': Acc1, 'auc': Auc1, 'precision': Precision1, 'recall': Recall1,
                   'f1': F11}

    return val_metric, test_metric



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
save_list = [f'algorithm,num,model,param,accuracy,precision,recall,f1,auc']

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
        str = f'SVM,{num},{metric["model"]},{metric["param"]},{metric["accuracy"]:.4f},{metric["precision"]:.4f},{metric["recall"]:.4f},{metric["f1"]:.4f},{metric["auc"]:.4f}'
        print(str)
        save_list.append(str)


all_experiment(90, H1_X, H1_y, H3_X, H3_y, H5_X, H5_y)
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