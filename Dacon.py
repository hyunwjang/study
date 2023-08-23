#!/usr/bin/env python
# coding: utf-8

# In[163]:


df_train = pd.read_csv('datacon/train.csv')
df_test = pd.read_csv('datacon/test.csv')
df_sample = pd.read_csv('datacon/sample_submission.csv')


# In[164]:


import pandas as pd
import seaborn as sns
import matplotlib as plt


# In[165]:


display(df_train.head())
display(df_test.head())


# In[166]:


print(df_train.shape)
print(df_train.info())
display(df_train.describe())
display(df_train.head())


# In[167]:


print(df_test.shape)
print(df_test.info())
display(df_test.describe())
display(df_test.head())


# In[168]:


for df_train_columns in [x for x in df_train.select_dtypes(exclude=object).columns]:
  display(df_train[df_train_columns].value_counts().sort_index(ascending=False).to_frame())
  print(len(df_train[df_train_columns].unique()),"개")
  print('*'*20)


# In[169]:


for df_test_columns in [x for x in df_test.select_dtypes(exclude=object).columns]:
  display(df_test[df_test_columns].value_counts().sort_index(ascending=False).to_frame())
  print(len(df_test[df_test_columns].unique()),"개")
  print('*'*20)


# In[170]:


x = 3
for df_train_columns in [x for x in df_test.select_dtypes(exclude=object).columns]:
#     max_c = df_train.loc[(df_train[df_train_columns] <df_train[df_train_columns].mean() - df_train[df_train_columns].std()*x) | \
#              (df_train[df_train_columns] >df_train[df_train_columns].mean() + df_train[df_train_columns].std()*x)]
#     print(max_c)
    print(df_train_columns, df_train[df_train_columns].max())
    print(len(df_train.loc[df_train[df_train_columns] == df_train[df_train_columns].max()].label) \
          , " / " \
         ,len(df_train.loc[(df_train[df_train_columns] == df_train[df_train_columns].max()) & df_train['label']==1].label), \
         " = " \
         ,int(len(df_train.loc[(df_train[df_train_columns] == df_train[df_train_columns].max()) & df_train['label']==1].label) /len(df_train.loc[df_train[df_train_columns] == df_train[df_train_columns].max()].label) *100) ,"%")
    display(df_train.loc[df_train[df_train_columns] == df_train[df_train_columns].max()].label)
    print('*' * 10)
    


# In[171]:


test = df_train.corr().unstack().to_frame().reset_index().dropna()
test.rename(columns ={0 :"corr"} ,inplace =True)
test = test.loc[test['corr']!=1].sort_values('corr',ascending = False).drop_duplicates('corr')
test.loc[(test.level_0=='label') |(test.level_1=='label')].sort_values('corr',ascending = False)

#헤모글로빈 : 적혈구의 산소 결합도
#헤모글로빈 , 키 , 몸무게가 label 값에서 0.3 이상의 상관관계를 확인함


# In[172]:


sns.boxplot(data = df_train)
#박스 플롯으로 이상값확인


# In[173]:


x = 3
int_t = df_train.select_dtypes(exclude=object).columns
print(int_t)
print(df_train.shape)
for i in int_t[:-1]:
    df_train = df_train.loc[(df_train[i] >df_train[i].mean() - df_train[i].std()*x) & \
             (df_train[i] <df_train[i].mean() + df_train[i].std()*x)]
print(df_train.shape)
#이상값 제거
#평균에서 표준편차 *3 값을 제외 시킴 
sns.boxplot(data = df_train)


# In[174]:


test = df_train.corr().unstack().to_frame().reset_index().dropna()
test.rename(columns ={0 :"corr"} ,inplace =True)
test = test.loc[test['corr']!=1].sort_values('corr',ascending = False).drop_duplicates('corr')
test.loc[(test.level_0=='label') |(test.level_1=='label')].sort_values('corr',ascending = False)
#이상값 제거 후에도 상관관계는 상위 3번째까지는 동일한 형태인것을 확인


# In[175]:


print(df_train.shape)
print(df_train.info())
display(df_train.describe())
display(df_train.head())


# In[176]:


#데이터 쪼개기 
X_train = df_train.iloc[:,1:-1]
X_train_ID = df_train.ID
Y = df_train.label


# In[177]:


X_train_m = X_train.copy()
df_test_m = df_test.copy()

from sklearn.preprocessing import MinMaxScaler
scaler = ['나이', '키(cm)', '몸무게(kg)', 'BMI', '시력', '충치',
          '공복 혈당', '혈압', '중성 지방',
       '혈청 크레아티닌', '콜레스테롤', '고밀도지단백', 
          '저밀도지단백', '헤모글로빈', '요 단백', '간 효소율']
mm = MinMaxScaler()
mm.fit(X_train_m[scaler])

X_train_m[scaler] = mm.transform(X_train_m[scaler])
df_test_m[scaler] = mm.transform(df_test_m[scaler])


# In[178]:


X_train_s = X_train.copy()
df_test_s = df_test.copy()

from sklearn.preprocessing import StandardScaler

scaler = ['나이', '키(cm)', '몸무게(kg)', 'BMI', '시력', '충치',
          '공복 혈당', '혈압', '중성 지방',
       '혈청 크레아티닌', '콜레스테롤', '고밀도지단백', 
          '저밀도지단백', '헤모글로빈', '요 단백', '간 효소율']
st = StandardScaler()
st.fit(X_train_s[scaler])

X_train_s[scaler] = st.transform(X_train[scaler])
df_test_s[scaler] = st.transform(df_test[scaler])


# In[179]:


from sklearn.decomposition import PCA
#pca 주성분검사
pca = PCA(n_components=2)

data = pca.fit_transform(df_train.drop(columns =['ID','label']))


sns.scatterplot(x=data[:, 0], y=data[:, 1], hue = df_train.label ,palette = 'tab10')

#시각화를 통해서 육안으로 분포도 확인


# In[180]:


# X_train.drop(columns = ['몸무게(kg)','콜레스테롤','요 단백','충치'], inplace = True)
# df_test.drop(columns = ['몸무게(kg)','콜레스테롤','요 단백','충치'], inplace = True)


# In[181]:


display(X_train_m.describe())
display(X_train_m.head())


# In[182]:


display(X_train_s.describe())
display(X_train_s.head())


# In[183]:


from sklearn.model_selection import train_test_split
x_train_m, x_val_m, y_train_m, y_val_m = train_test_split(X_train_m, Y,test_size = 0.3, random_state = 42,stratify = Y)


# In[184]:


from sklearn.model_selection import train_test_split
x_train_s, x_val_s, y_train_s, y_val_s = train_test_split(X_train_s, Y,test_size = 0.3, random_state = 42,stratify = Y)


# In[185]:


#데이터 분리


# In[186]:


from sklearn.ensemble import RandomForestClassifier
rf_m = RandomForestClassifier(random_state = 42)
rf_s = RandomForestClassifier(random_state = 42)
#튜닝 전 성능확인


# In[187]:


rf_m.fit(x_train_m, y_train_m)
rf_s.fit(x_train_s, y_train_s)


# In[188]:


#standar
predict_train_s = rf_s.predict(x_train_s)
predict_val_s = rf_s.predict(x_val_s)

#minmax
predict_train_m = rf_m.predict(x_train_m)
predict_val_m = rf_m.predict(x_val_m)


# In[189]:


from sklearn.metrics import accuracy_score ,f1_score, confusion_matrix, classification_report

print("standar")
print(">>>train --------------------------------")
print("accuracy_score : ", accuracy_score(y_train_s, predict_train_s))
print("f1_score : ", f1_score(y_train_s, predict_train_s))
# print(confusion_matrix(y_train, predict_train))
# print(classification_report(y_train, predict_train))
print("\n>>>val ---------------------------------")
print("accuracy_score : ", accuracy_score(y_val_s, predict_val_s))
print("f1_score : ", f1_score(y_val_s, predict_val_s))

print("\nminmax")
print(">>>train --------------------------------")
print("accuracy_score : ", accuracy_score(y_train_m, predict_train_m))
print("f1_score : ", f1_score(y_train_m, predict_train_m))
print("\n>>>val ---------------------------------")
print("accuracy_score : ", accuracy_score(y_val_m, predict_val_m))
print("f1_score : ", f1_score(y_val_m, predict_val_m))


# In[154]:


from sklearn.model_selection import GridSearchCV
#standar
param_grid = {'n_estimators':[50,100,150], 'max_depth' : [10,11,12], 'min_samples_leaf':[7,8,9]}
clf_s = GridSearchCV(rf_s, param_grid, cv = 3)
clf_s.fit(x_train_s, y_train_s)
print('Best Parameters: ', clf_s.best_params_)
print('Best Score: ', clf_s.best_score_)
print('TestSet Score: ', clf_s.score(x_val_s, y_val_s))


# In[192]:


#minmax

param_grid = {'n_estimators':[50,100,150],'max_depth' : [10,11,12], 'min_samples_leaf':[7,8,9]}
clf_m = GridSearchCV(rf_m, param_grid, cv = 3)
clf_m.fit(x_train_m, y_train_m)
print('Best Parameters: ', clf_m.best_params_)
print('Best Score: ', clf_m.best_score_)
print('TestSet Score: ', clf_m.score(x_val_m, y_val_m))


# In[190]:


rf_s = RandomForestClassifier(random_state = 42, n_estimators = 150, \
                            max_depth = 11, min_samples_leaf= 9)
rf_s.fit(x_train_s, y_train_s)

#standar
predict_train_s = rf_s.predict(x_train_s)
predict_val_s = rf_s.predict(x_val_s)

print("standar")
print(">>>train --------------------------------")
print("accuracy_score : ", accuracy_score(y_train_s, predict_train_s))
print("f1_score : ", f1_score(y_train_s, predict_train_s))
# print(confusion_matrix(y_train, predict_train))
# print(classification_report(y_train, predict_train))
print("\n>>>val ---------------------------------")
print("accuracy_score : ", accuracy_score(y_val_s, predict_val_s))
print("f1_score : ", f1_score(y_val_s, predict_val_s))


# In[193]:


rf_m = RandomForestClassifier(random_state = 42, n_estimators = 100, \
                            max_depth = 11, min_samples_leaf= 8)
rf_m.fit(x_train_m, y_train_m)

#standar
predict_train_m = rf_m.predict(x_train_m)
predict_val_m = rf_m.predict(x_val_m)

print("standar")
print(">>>train --------------------------------")
print("accuracy_score : ", accuracy_score(y_train_m, predict_train_m))
print("f1_score : ", f1_score(y_train_m, predict_train_m))
# print(confusion_matrix(y_train, predict_train))
# print(classification_report(y_train, predict_train))
print("\n>>>val ---------------------------------")
print("accuracy_score : ", accuracy_score(y_val_m, predict_val_m))
print("f1_score : ", f1_score(y_val_m, predict_val_m))


# In[ ]:


# rf = RandomForestClassifier(random_state = 42, n_estimators = 100, \
#                             max_depth = 10, min_samples_leaf= 5)
# rf.fit(x_train, y_train)
# predict_train = rf.predict(x_train)
# predict_val = rf.predict(x_val)

# proba_train = rf.predict_proba(x_train)[: ,1]
# proba_val = rf.predict_proba(x_val)[: ,1]     

# print("\n>>>train --------------------------------")
# print("accuracy_score : ", accuracy_score(y_train, predict_train))
# print("f1_score : ", f1_score(y_train, predict_train))
# print("roc_auc_score : ", roc_auc_score(y_train, proba_train))
# print(confusion_matrix(y_train, predict_train))
# print(classification_report(y_train, predict_train))
# print("\n>>>test ---------------------------------")
# print("accuracy_score : ", accuracy_score(y_val, predict_val))
# print("f1_score : ", f1_score(y_val, predict_val))
# print("roc_auc_score : ", roc_auc_score(y_val, proba_val))
# print(confusion_matrix(y_val, predict_val))
# print(classification_report(y_val, predict_val))


# In[ ]:


result = pd.DataFrame(rf.predict(df_test.iloc[:,1:]))
result
df_sample.label = result
df_sample.to_csv('sample_submission.csv', index = False)
check = pd.read_csv("sample_submission.csv")
check.head()


# In[ ]:


df_sample.label = result
df_sample.to_csv('sample_submission.csv', index = False)
check = pd.read_csv("sample_submission.csv")
check.head()


# In[208]:


X_train_d = X_train.loc[:,['키(cm)','몸무게(kg)','헤모글로빈','혈청 크레아티닌','중성 지방']]
df_test_d = df_test.loc[:,['키(cm)','몸무게(kg)','헤모글로빈','혈청 크레아티닌','중성 지방']]


scaler = ['키(cm)','몸무게(kg)','헤모글로빈','혈청 크레아티닌','중성 지방']
st = StandardScaler()
st.fit(X_train_d[scaler])

X_train_d[scaler] = st.transform(X_train_d[scaler])
df_test_d[scaler] = st.transform(df_test_d[scaler])

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X_train_d, Y,test_size = 0.3, random_state = 42,stratify = Y)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 42)
rf.fit(x_train, y_train)

predict_train = rf.predict(x_train)
predict_val = rf.predict(x_val)

print("\n>>>train --------------------------------")
print("accuracy_score : ", accuracy_score(y_train, predict_train))
print("f1_score : ", f1_score(y_train, predict_train))

print("\n>>>test ---------------------------------")
print("accuracy_score : ", accuracy_score(y_val, predict_val))
print("f1_score : ", f1_score(y_val, predict_val))

#상관관계가 20프로 넘는 것으로 머신러닝
#5개의 columns를 하였을때 정확도 상승


# In[209]:


param_grid = {'n_estimators':[150,200,250], 'max_depth' : [11,12,13], 'min_samples_leaf':[8,9,10]}
clf = GridSearchCV(rf, param_grid, cv = 3)
clf.fit(x_train, y_train)
print('Best Parameters: ', clf.best_params_)
print('Best Score: ', clf.best_score_)
print('TestSet Score: ', clf.score(x_val, y_val))

#선택된 파라미터가 중간값이 될때까지 순차적으로 돌려봄


# In[210]:


rf = RandomForestClassifier(random_state = 42, n_estimators = 200, \
                            max_depth = 12, min_samples_leaf= 9)
rf.fit(x_train, y_train)
predict_train = rf.predict(x_train)
predict_val = rf.predict(x_val)

print("\n>>>train --------------------------------")
print("accuracy_score : ", accuracy_score(y_train, predict_train))
print("f1_score : ", f1_score(y_train, predict_train))

print("\n>>>test ---------------------------------")
print("accuracy_score : ", accuracy_score(y_val, predict_val))
print("f1_score : ", f1_score(y_val, predict_val))


# In[211]:


df_test_d


# In[212]:


X_train_d


# In[214]:


result = pd.DataFrame(rf.predict(df_test_d))
result
df_sample.label = result
df_sample.to_csv('sample_submission.csv', index = False)
check = pd.read_csv("sample_submission.csv")
check.head()


# In[ ]:




