import pandas as pd
import matplotlib.pyplot as plt

model_self = ['H1->self', 'H3->self', 'H5->self']
model_val = ['H1->H3 50%val', 'H1->H5 50%val', 'H3->H1 50%val', 'H3->H5 50%val',
             'H5->H1 50%val', 'H5->H1 50%val', 'H1H3->H5 50%val', 'H1H5->H3 50%val', 'H3H5->H1 50%val']
model_test = ['H1->H3 50%test', 'H1->H5 50%test', 'H3->H1 50%test', 'H3->H5 50%test',
              'H5->H1 50%test', 'H5->H1 50%test', 'H1H3->H5 50%test', 'H1H5->H3 50%test', 'H3H5->H1 50%test']

df = pd.read_csv('results.csv', sep=',')

df_self = df[df['model'].isin(model_self)]
df_val = df[df['model'].isin(model_val)]

plt.figure(figsize=(12, 6))
##############################################################
df_logistic_val = df_val[df_val['algorithm'] == 'logistic']
df_logistic_val_mean = df_logistic_val.groupby('num').agg(
    val_acc_mean=pd.NamedAgg(column='accuracy', aggfunc='mean'),
    val_acc_var=pd.NamedAgg(column='accuracy', aggfunc='var'),
    val_auc_mean=pd.NamedAgg(column='auc', aggfunc='mean'),
    val_auc_var=pd.NamedAgg(column='auc', aggfunc='var')
)
df_logistic_val_mean['average'] = (df_logistic_val_mean['val_acc_mean'] + df_logistic_val_mean['val_auc_mean']) / 2

plt.plot(range(5, 325, 5), df_logistic_val_mean['val_acc_mean'].values, color='pink', marker='.',
         label='logistic_Accuracy')
plt.plot(range(5, 325, 5), df_logistic_val_mean['val_auc_mean'].values, color='red', marker='.',
         label='logistic_AUC')
plt.plot(range(5, 325, 5), df_logistic_val_mean['average'].values, color='black', marker='.',
         label='logistic_average')
##############################################################
df_univ_val = df_val[df_val['algorithm'] == 'Univ-Flu']

df_univ_val_mean = pd.DataFrame({
    'val_acc_mean': [df_univ_val['accuracy'].mean()],
    'val_acc_var': [df_univ_val['accuracy'].var()],
    'val_auc_mean': [df_univ_val['auc'].mean()],
    'val_auc_var': [df_univ_val['auc'].var()]
})
df_univ_val_mean['average']=(df_univ_val_mean['val_acc_mean']+df_univ_val_mean['val_auc_mean'])/2
# plt.scatter(100, df_univ_val_mean['val_acc_mean'].values, marker='^', label='Univ-Flu_Accuracy')
# plt.scatter(100, df_univ_val_mean['val_auc_mean'].values, marker='^', label='Univ-Flu_AUC')
plt.scatter(100, df_univ_val_mean['average'].values, marker='.', label='Univ-Flu_average')
##############################################################
df_univ_val = df_val[df_val['algorithm'] == 'byes']

df_univ_val_mean = pd.DataFrame({
    'val_acc_mean': [df_univ_val['accuracy'].mean()],
    'val_acc_var': [df_univ_val['accuracy'].var()],
    'val_auc_mean': [df_univ_val['auc'].mean()],
    'val_auc_var': [df_univ_val['auc'].var()]
})
df_univ_val_mean['average']=(df_univ_val_mean['val_acc_mean']+df_univ_val_mean['val_auc_mean'])/2
# plt.scatter(105, df_univ_val_mean['val_acc_mean'].values, marker='^', label='byes_Accuracy')
# plt.scatter(105, df_univ_val_mean['val_auc_mean'].values, marker='^', label='byes_AUC')
plt.scatter(110, df_univ_val_mean['average'].values, marker='.', label='byes_average')
##############################################################
df_univ_val = df_val[df_val['algorithm'] == 'SVM']

df_univ_val_mean = pd.DataFrame({
    'val_acc_mean': [df_univ_val['accuracy'].mean()],
    'val_acc_var': [df_univ_val['accuracy'].var()],
    'val_auc_mean': [df_univ_val['auc'].mean()],
    'val_auc_var': [df_univ_val['auc'].var()]
})
df_univ_val_mean['average']=(df_univ_val_mean['val_acc_mean']+df_univ_val_mean['val_auc_mean'])/2
# plt.scatter(110, df_univ_val_mean['val_acc_mean'].values, marker='^', label='SVM_Accuracy')
# plt.scatter(110, df_univ_val_mean['val_auc_mean'].values, marker='^', label='SVM_AUC')
plt.scatter(120, df_univ_val_mean['average'].values, marker='.', label='SVM_average')
##############################################################
df_univ_val = df_val[df_val['algorithm'] == 'RF']

df_univ_val_mean = pd.DataFrame({
    'val_acc_mean': [df_univ_val['accuracy'].mean()],
    'val_acc_var': [df_univ_val['accuracy'].var()],
    'val_auc_mean': [df_univ_val['auc'].mean()],
    'val_auc_var': [df_univ_val['auc'].var()]
})
df_univ_val_mean['average']=(df_univ_val_mean['val_acc_mean']+df_univ_val_mean['val_auc_mean'])/2
# plt.scatter(115, df_univ_val_mean['val_acc_mean'].values, marker='^', label='RF_Accuracy')
# plt.scatter(115, df_univ_val_mean['val_auc_mean'].values, marker='^', label='RF_AUC')
plt.scatter(130, df_univ_val_mean['average'].values, marker='.', label='RF_average')
##############################################################
df_univ_val = df_val[df_val['algorithm'] == 'CNN_BiLSTM']

df_univ_val_mean = pd.DataFrame({
    'val_acc_mean': [df_univ_val['accuracy'].mean()],
    'val_acc_var': [df_univ_val['accuracy'].var()],
    'val_auc_mean': [df_univ_val['auc'].mean()],
    'val_auc_var': [df_univ_val['auc'].var()]
})
df_univ_val_mean['average']=(df_univ_val_mean['val_acc_mean']+df_univ_val_mean['val_auc_mean'])/2
# plt.scatter(120, df_univ_val_mean['val_acc_mean'].values, marker='^', label='CNN_BiLSTM_Accuracy')
# plt.scatter(120, df_univ_val_mean['val_auc_mean'].values, marker='^', label='CNN_BiLSTM_AUC')
plt.scatter(140, df_univ_val_mean['average'].values, marker='.', label='CNN_BiLSTM_average')
# 添加图例和标签
# 设置y轴范围为0.2到1
plt.xlabel('num/step=5', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Model Average Performance on subtype_cross  val set (Accuracy and AUC)', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
# 调整布局
plt.tight_layout()
plt.show()

df_val = df[df['model'].isin(model_test)]

plt.figure(figsize=(12, 6))
##############################################################
df_logistic_val = df_val[df_val['algorithm'] == 'logistic']
df_logistic_val_mean = df_logistic_val.groupby('num').agg(
    val_acc_mean=pd.NamedAgg(column='accuracy', aggfunc='mean'),
    val_acc_var=pd.NamedAgg(column='accuracy', aggfunc='var'),
    val_auc_mean=pd.NamedAgg(column='auc', aggfunc='mean'),
    val_auc_var=pd.NamedAgg(column='auc', aggfunc='var')
)
df_logistic_val_mean['average'] = (df_logistic_val_mean['val_acc_mean'] + df_logistic_val_mean['val_auc_mean']) / 2
print('Logistic',
      df_logistic_val_mean['val_acc_mean'].values[19],
      df_logistic_val_mean['val_auc_mean'].values[19])
plt.plot(range(5, 325, 5), df_logistic_val_mean['val_acc_mean'].values, color='pink', marker='.',
         label='logistic_Accuracy')
plt.plot(range(5, 325, 5), df_logistic_val_mean['val_auc_mean'].values, color='red', marker='.',
         label='logistic_AUC')
plt.plot(range(5, 325, 5), df_logistic_val_mean['average'].values, color='black', marker='.',
         label='logistic_average')
##############################################################
df_univ_val = df_val[df_val['algorithm'] == 'Univ-Flu']

df_univ_val_mean = pd.DataFrame({
    'val_acc_mean': [df_univ_val['accuracy'].mean()],
    'val_acc_var': [df_univ_val['accuracy'].var()],
    'val_auc_mean': [df_univ_val['auc'].mean()],
    'val_auc_var': [df_univ_val['auc'].var()]
})
print('Univ-Flu',
      df_univ_val_mean['val_acc_mean'].values,
      df_univ_val_mean['val_auc_mean'].values)
df_univ_val_mean['average']=(df_univ_val_mean['val_acc_mean']+df_univ_val_mean['val_auc_mean'])/2
# plt.scatter(100, df_univ_val_mean['val_acc_mean'].values, marker='^', label='Univ-Flu_Accuracy')
# plt.scatter(100, df_univ_val_mean['val_auc_mean'].values, marker='^', label='Univ-Flu_AUC')
plt.scatter(100, df_univ_val_mean['average'].values, marker='.', label='Univ-Flu_average')
##############################################################
df_univ_val = df_val[df_val['algorithm'] == 'byes']

df_univ_val_mean = pd.DataFrame({
    'val_acc_mean': [df_univ_val['accuracy'].mean()],
    'val_acc_var': [df_univ_val['accuracy'].var()],
    'val_auc_mean': [df_univ_val['auc'].mean()],
    'val_auc_var': [df_univ_val['auc'].var()]
})
print('byes',
      df_univ_val_mean['val_acc_mean'].values,
      df_univ_val_mean['val_auc_mean'].values)
df_univ_val_mean['average']=(df_univ_val_mean['val_acc_mean']+df_univ_val_mean['val_auc_mean'])/2
# plt.scatter(105, df_univ_val_mean['val_acc_mean'].values, marker='^', label='byes_Accuracy')
# plt.scatter(105, df_univ_val_mean['val_auc_mean'].values, marker='^', label='byes_AUC')
plt.scatter(110, df_univ_val_mean['average'].values, marker='.', label='byes_average')
##############################################################
df_univ_val = df_val[df_val['algorithm'] == 'SVM']

df_univ_val_mean = pd.DataFrame({
    'val_acc_mean': [df_univ_val['accuracy'].mean()],
    'val_acc_var': [df_univ_val['accuracy'].var()],
    'val_auc_mean': [df_univ_val['auc'].mean()],
    'val_auc_var': [df_univ_val['auc'].var()]
})
print('SVM',
      df_univ_val_mean['val_acc_mean'].values,
      df_univ_val_mean['val_auc_mean'].values)
df_univ_val_mean['average']=(df_univ_val_mean['val_acc_mean']+df_univ_val_mean['val_auc_mean'])/2
# plt.scatter(110, df_univ_val_mean['val_acc_mean'].values, marker='^', label='SVM_Accuracy')
# plt.scatter(110, df_univ_val_mean['val_auc_mean'].values, marker='^', label='SVM_AUC')
plt.scatter(120, df_univ_val_mean['average'].values, marker='.', label='SVM_average')
##############################################################
df_univ_val = df_val[df_val['algorithm'] == 'RF']

df_univ_val_mean = pd.DataFrame({
    'val_acc_mean': [df_univ_val['accuracy'].mean()],
    'val_acc_var': [df_univ_val['accuracy'].var()],
    'val_auc_mean': [df_univ_val['auc'].mean()],
    'val_auc_var': [df_univ_val['auc'].var()]
})
print('RF',
      df_univ_val_mean['val_acc_mean'].values,
      df_univ_val_mean['val_auc_mean'].values)
df_univ_val_mean['average']=(df_univ_val_mean['val_acc_mean']+df_univ_val_mean['val_auc_mean'])/2
# plt.scatter(115, df_univ_val_mean['val_acc_mean'].values, marker='^', label='RF_Accuracy')
# plt.scatter(115, df_univ_val_mean['val_auc_mean'].values, marker='^', label='RF_AUC')
plt.scatter(130, df_univ_val_mean['average'].values, marker='.', label='RF_average')
##############################################################
df_univ_val = df_val[df_val['algorithm'] == 'CNN_BiLSTM']

df_univ_val_mean = pd.DataFrame({
    'val_acc_mean': [df_univ_val['accuracy'].mean()],
    'val_acc_var': [df_univ_val['accuracy'].var()],
    'val_auc_mean': [df_univ_val['auc'].mean()],
    'val_auc_var': [df_univ_val['auc'].var()]
})
print('CNN_BiLSTM',
      df_univ_val_mean['val_acc_mean'].values,
      df_univ_val_mean['val_auc_mean'].values)
df_univ_val_mean['average']=(df_univ_val_mean['val_acc_mean']+df_univ_val_mean['val_auc_mean'])/2
# plt.scatter(120, df_univ_val_mean['val_acc_mean'].values, marker='^', label='CNN_BiLSTM_Accuracy')
# plt.scatter(120, df_univ_val_mean['val_auc_mean'].values, marker='^', label='CNN_BiLSTM_AUC')
plt.scatter(140, df_univ_val_mean['average'].values, marker='.', label='CNN_BiLSTM_average')
##############################################################
df_univ_val = df_val[df_val['algorithm'] == 'RF_fusion']

df_univ_val_mean = pd.DataFrame({
    'val_acc_mean': [df_univ_val['accuracy'].mean()],
    'val_acc_var': [df_univ_val['accuracy'].var()],
    'val_auc_mean': [df_univ_val['auc'].mean()],
    'val_auc_var': [df_univ_val['auc'].var()]
})
print('RF_fusion',
      df_univ_val_mean['val_acc_mean'].values,
      df_univ_val_mean['val_auc_mean'].values)
df_univ_val_mean['average']=(df_univ_val_mean['val_acc_mean']+df_univ_val_mean['val_auc_mean'])/2
# plt.scatter(120, df_univ_val_mean['val_acc_mean'].values, marker='^', label='RF_fusion_Accuracy')
# plt.scatter(120, df_univ_val_mean['val_auc_mean'].values, marker='^', label='RF_fusion_AUC')
plt.scatter(140, df_univ_val_mean['average'].values, marker='.', label='RF_fusion_average')
# 添加图例和标签
# 设置y轴范围为0.2到1
plt.xlabel('num/step=5', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Model Average Performance on subtype_cross  val set (Accuracy and AUC)', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
# 调整布局
plt.tight_layout()
plt.show()