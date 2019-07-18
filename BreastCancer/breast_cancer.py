from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, SparsePCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy as np

# Carrego llibreria que conte les dades de features i labels.
data = load_breast_cancer()
features = data['data']
labels = data['target']

# A target, 0 es maligne i 1 es benigne. Normalitzem les features.
features_norm = (features - features.mean(0))/ features.std(0)

# M de maligne i B de benigne.
M = data['target'] == 0
B = data['target'] == 1

# Faig PCA de dos components per tal de visualitzar els dos grups benignes i malignes.

pca = PCA(n_components = 2).fit(features_norm)
features_pca = pca.transform(features_norm)
print (features_pca.shape)
fig = plt.figure()
plt.hist(features_pca[M, 0])
plt.hist(features_pca[B, 0])
plt.legend(['Malign', 'Benign'])
plt.xlabel("Component 1")
fig.tight_layout()
fig.savefig("HistogramPlotPCA1.png")
plt.show()

lda = LinearDiscriminantAnalysis(n_components = 2).fit(features_norm, labels)
features_lda = lda.transform(features_norm)
fig = plt.figure()
plt.hist(features_lda[M, 0])
plt.hist(features_lda[B, 0])
plt.legend(['Malign', 'Benign'])
plt.xlabel("Component 1")
fig.tight_layout()
fig.savefig("HistogramPlotLDA1.png")
plt.show()


fig = plt.figure()
plt.hist(features_lda[M, 1])
plt.hist(features_lda[B, 1])
plt.xlabel("Component 2")
plt.legend(['Malign', 'Benign'])
fig.tight_layout()
fig.savefig("HistogramPlotPCA2.png")
plt.show()

fig = plt.figure()
plt.scatter(features_pca[M, 0], features_pca[M, 1])
plt.scatter(features_pca[B, 0], features_pca[B, 1])
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(['Malign', 'Benign'])
fig.tight_layout()
fig.savefig("PCA.png")
plt.show()

# Faig train_test_split i les prediccions del y_test.
x_train, x_test, y_train, y_test = train_test_split(features_norm, labels, test_size = 0.25, random_state = 10)
model_logistic = LogisticRegression().fit(x_train, y_train)
model_forest = RandomForestClassifier().fit(x_train, y_train)
model_linear = LinearRegression().fit(x_train, y_train)

# Miro la importancia de cada feature.
forest = ExtraTreesClassifier(n_estimators = 100)
forest.fit(x_train,y_train)
importancia = forest.feature_importances_
print(importancia)

# Miro el score de cada model i faig les prediccions de y_train i y_test
train_yhat_linear = model_linear.predict(x_train)
test_yhat_linear = model_linear.predict(x_test)
print (model_linear.score(x_test, y_test))
print (model_linear.score(x_test, y_test))

train_yhat_logistic = model_logistic.predict(x_train)
test_yhat_logistic = model_logistic.predict(x_test)
print (model_logistic.score(x_test, y_test))
print (model_logistic.score(x_test, y_test))

train_yhat_forest = model_forest.predict(x_train)
test_yhat_forest = model_forest.predict(x_test)
print (model_forest.score(x_test, y_test))
print (model_forest.score(x_test, y_test))

# Faig funcio de confusion_matrix.
def confusion_matrix(predicted, real, gamma):
  decisio = np.where(predicted < gamma, 0, 1)
  tp = np.logical_and(real == 1, decisio == 1).sum()
  tn = np.logical_and(real == 0, decisio == 0).sum()
  fp = np.logical_and(real == 0, decisio == 1).sum()
  fn = np.logical_and(real == 1, decisio == 0).sum()
  return tp, fp, tn, fn


# Calculo en una llista la tpr i fpr de cada un.
tpr_train = []
fpr_train = []
fpr_test = []
tpr_test = []

gammas = np.arange(0, 1, 0.01)
for gamma in gammas:
  tp_train, fp_train, tn_train, fn_train = confusion_matrix(train_yhat_linear, y_train, gamma)
  tpr_train.append(tp_train/(tp_train + fn_train))
  fpr_train.append(fp_train/(tn_train + fp_train))
  tp_test, fp_test, tn_test, fn_test = confusion_matrix(test_yhat_linear, y_test, gamma)
  tpr_test.append(tp_test/(tp_test + fn_test))
  fpr_test.append(fp_test/(tn_test + fp_test))
  tp_train, fp_train, tn_train, fn_train = confusion_matrix(train_yhat_logistic, y_train, gamma)
  tpr_train.append(tp_train/(tp_train + fn_train))
  fpr_train.append(fp_train/(tn_train + fp_train))
  tp_test, fp_test, tn_test, fn_test = confusion_matrix(test_yhat_logistic, y_test, gamma)
  tpr_test.append(tp_test/(tp_test + fn_test))
  fpr_test.append(fp_test/(tn_test + fp_test))
  tp_train, fp_train, tn_train, fn_train = confusion_matrix(train_yhat_forest, y_train, gamma)
  tpr_train.append(tp_train/(tp_train + fn_train))
  fpr_train.append(fp_train/(tn_train + fp_train))
  tp_test, fp_test, tn_test, fn_test = confusion_matrix(test_yhat_forest, y_test, gamma)
  tpr_test.append(tp_test/(tp_test + fn_test))
  fpr_test.append(fp_test/(tn_test + fp_test))

# Faig un plot scatter de fpr i tpr per veure la ROC curve.
fig = plt.figure()
plt.scatter(fpr_train, tpr_train)
plt.scatter(fpr_test, tpr_test)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(['Train', 'Test'])
fig.tight_layout()
fig.savefig("fpr_tpr.png")
plt.show()

# Miro el score de la ROC curve (mesurada amb l'area del grafic)
print (roc_auc_score(y_train, train_yhat_linear))
print (roc_auc_score(y_train, train_yhat_logistic))
print (roc_auc_score(y_train, train_yhat_forest))
