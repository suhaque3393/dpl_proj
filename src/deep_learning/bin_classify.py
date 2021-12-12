# Part I: Binary Classifier


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X, y = mnist["data"], mnist["target"]
X.shape
y.shape

import matplotlib as mpl
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = X[:3000], X[3001:3501], y[:3000], y[3001:3501]
y_train_0 = (y_train == '0')
y_test_0 = (y_test == '0')

#SGD CLASSIFIER


# 1. Testing Accuracy

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_0)

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy")

# 2. Confusion Matrix

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_0, cv=3)
 
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_0, y_train_pred)

# 3. Precision



from sklearn.metrics import precision_score, recall_score

precision_score(y_train_0, y_train_pred)

# 4. Recall


recall_score(y_train_0, y_train_pred)

# 5. F1 score


from sklearn.metrics import f1_score

f1_score(y_train_0, y_train_pred)

# 6. ROC curve


y_scores = cross_val_predict(sgd_clf, X_train, y_train_0, cv=3,
                             method="decision_function")
from sklearn.metrics import roc_curve
import numpy as np
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_0, y_scores)
recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

fpr, tpr, thresholds = roc_curve(y_train_0, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) 
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    
    plt.grid(True)                                            

plt.figure(figsize=(8, 6))                                    
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]           
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  
plt.plot([fpr_90], [recall_90_precision], "ro")              

plt.show()     

# 7. The area under the curve

from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_0, y_scores)

#K NEAREST NEIGHBOR

# 1. Accuracy

from sklearn import neighbors
for i in range(3, 16):
  neigh = neighbors.KNeighborsClassifier(i)
  neigh.fit(X_train, y_train_0)
  acc = cross_val_score(neigh, X_train, y_train_0, cv=3, scoring="accuracy")
  ave = sum(acc)/len(acc)
  print(f"K = {i}: {acc}, Average: {ave}")

# Optimal value is K = 3

# 2. Confusion Matrix

neigh = neighbors.KNeighborsClassifier(3)
neigh.fit(X_train, y_train_0)
y_train_pred = cross_val_predict(neigh, X_train, y_train_0, cv=3)
confusion_matrix(y_train_0, y_train_pred)

# 3. Precision

precision_score(y_train_0, y_train_pred)

# 4. Recall

recall_score(y_train_0, y_train_pred)

# 5. F1 Score

f1_score(y_train_0, y_train_pred)

# 6. ROC Curve

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) 
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    
    plt.grid(True)                                            

y_scores = cross_val_predict(neigh, X_train, y_train_0, cv=3, method="predict")

precisions, recalls, thresholds = precision_recall_curve(y_train_0, y_scores)
recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

fpr, tpr, thresholds = roc_curve(y_train_0, y_scores)


plt.figure(figsize=(8, 6))                                    
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]           
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  
plt.plot([fpr_90], [recall_90_precision], "ro")              

plt.show() 

# 7. Area Under Curve

roc_auc_score(y_train_0, y_scores)

# LOGISTIC REGRESSION

# 1. Testing Accuracy

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
log_reg.fit(X_train, y_train_0)
cross_val_score(log_reg, X_train, y_train_0, cv=3, scoring="accuracy")

# 2. Confusion Matrix

y_train_pred = cross_val_predict(log_reg, X_train, y_train_0, cv=3)
confusion_matrix(y_train_0, y_train_pred)

# 3. Precision

precision_score(y_train_0, y_train_pred)

# 4. Recall

recall_score(y_train_0, y_train_pred)

# 5. F1 Score

f1_score(y_train_0, y_train_pred)

# 6. ROC Curve

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) 
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    
    plt.grid(True)                                            

y_scores = cross_val_predict(log_reg, X_train, y_train_0, cv=3, method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_0, y_scores)
recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

fpr, tpr, thresholds = roc_curve(y_train_0, y_scores)


plt.figure(figsize=(8, 6))                                    
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]           
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  
plt.plot([fpr_90], [recall_90_precision], "ro")              

plt.show() 

# 7. Area Under Curve

roc_auc_score(y_train_0, y_scores)

# SVM With Linear Kernel

# 1. Accuracy

from sklearn import svm
for i in range(-3, 4):
  linear_svm = svm.SVC(kernel='linear', C=10**i)
  linear_svm.fit(X_train, y_train_0)
  acc = cross_val_score(linear_svm, X_train, y_train_0, cv=3, scoring="accuracy")
  ave = sum(acc)/len(acc)
  print(f"C = 10^{i}: {acc}, Average: {ave}")

# All accuracy scores are the same.

# 2. Confusion Matrix

y_train_pred = cross_val_predict(linear_svm, X_train, y_train_0, cv=3)
confusion_matrix(y_train_0, y_train_pred)

# 3. Precision

precision_score(y_train_0, y_train_pred)

# 4. Recall

recall_score(y_train_0, y_train_pred)

# 5. F1 Score

f1_score(y_train_0, y_train_pred)

# 6. ROC Curve

y_scores = cross_val_predict(linear_svm, X_train, y_train_0, cv=3,
                             method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_0, y_scores)
recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

fpr, tpr, thresholds = roc_curve(y_train_0, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) 
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    
    plt.grid(True)                                            

plt.figure(figsize=(8, 6))                                    
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]           
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  
plt.plot([fpr_90], [recall_90_precision], "ro")              

plt.show() 

# 7. Area Under Curve

roc_auc_score(y_train_0, y_scores)

#SVM with the 2nd Order Kernel

# 1. Accuracy

for i in range(-3, 4):
  svm_second = svm.SVC(kernel='poly', degree = 2, C=10**i)
  svm_second.fit(X_train, y_train_0)
  acc = cross_val_score(svm_second, X_train, y_train_0, cv=3, scoring="accuracy")
  print(f"C = 10^{i}: {acc}")

# Accuracy score is highest and  the same for i = 1, 2 and 3

# 2. Confusion Matrix

second_svm = svm.SVC(kernel='poly', degree = 2, C=10**3)
second_svm.fit(X_train, y_train_0)
y_train_pred = cross_val_predict(second_svm, X_train, y_train_0, cv=3)
confusion_matrix(y_train_0, y_train_pred)

# 3. Precision

precision_score(y_train_0, y_train_pred)

# 4. Recall

recall_score(y_train_0, y_train_pred)

# 5. F1 Score

f1_score(y_train_0, y_train_pred)

# 6. ROC Curve

y_scores = cross_val_predict(svm_second, X_train, y_train_0, cv=3,
                             method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_0, y_scores)
recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

fpr, tpr, thresholds = roc_curve(y_train_0, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) 
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    
    plt.grid(True)                                            

plt.figure(figsize=(8, 6))                                    
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]           
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  
plt.plot([fpr_90], [recall_90_precision], "ro")              

plt.show() 

# 7. Area Under Curve

roc_auc_score(y_train_0, y_scores)

# SVM with the 3rd Order Kernel

# 1. Accuracy

for i in range(-3, 4):
  svm_third = svm.SVC(kernel='poly', degree = 3, C=10**i)
  svm_third.fit(X_train, y_train_0)
  acc = cross_val_score(svm_third, X_train, y_train_0, cv=3, scoring="accuracy")
  print(f"C = 10^{i}: {acc}")

# Accuracy score is the highest and the same for i = 1, 2, and 3

# 2. Confusion Matrix

third_svm = svm.SVC(kernel='poly', degree = 3, C=10**3)
third_svm.fit(X_train, y_train_0)
y_train_pred = cross_val_predict(third_svm, X_train, y_train_0, cv=3)
confusion_matrix(y_train_0, y_train_pred)

# 3. Precision

precision_score(y_train_0, y_train_pred)

# 4. Recall

recall_score(y_train_0, y_train_pred)

# 5. F1 Score

f1_score(y_train_0, y_train_pred)

# 6. ROC Curve

y_scores = cross_val_predict(svm_third, X_train, y_train_0, cv=3,
                             method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_0, y_scores)
recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

fpr, tpr, thresholds = roc_curve(y_train_0, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) 
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    
    plt.grid(True)                                            

plt.figure(figsize=(8, 6))                                    
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]           
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  
plt.plot([fpr_90], [recall_90_precision], "ro")              

plt.show() 

# 7. Area Under Curve

roc_auc_score(y_train_0, y_scores)

# Part II: Multiclass classification

# SGD CLASSIFIER

# 1. Accuracy

sgd_mult = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_mult.fit(X_train, y_train)
cross_val_score(sgd_mult, X_train, y_train, cv=3, scoring="accuracy")

# 2. Confusion Matrix

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_mult, X_train_scaled, y_train, cv=3, scoring="accuracy")
y_train_pred = cross_val_predict(sgd_mult, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx

# K NEAREST NEIGHBOR

# 1. Accuracy

for i in range(3, 16):
  neigh_mult = neighbors.KNeighborsClassifier(i)
  neigh_mult.fit(X_train, y_train)
  acc = cross_val_score(neigh_mult, X_train, y_train, cv=3, scoring="accuracy")
  ave = sum(acc)/len(acc)
  print(f"K = {i}: {acc}, Average: {ave}")

# Optimal K = 6

# 2. Confusion Matrix

neigh_mult = neighbors.KNeighborsClassifier(6)
neigh_mult.fit(X_train, y_train)
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
y_train_pred = cross_val_predict(neigh_mult, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx

# SOFTMAX REGRESSION

# 1. Accuracy

softmax_reg = LogisticRegression(max_iter = 1000,multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
softmax_reg.fit(X_train, y_train)
cross_val_score(softmax_reg, X_train, y_train, cv=3, scoring="accuracy")

# 2. Confusion Matrix

y_train_pred = cross_val_predict(softmax_reg, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx

# SVM with the Linear Kernel

# 1. Accuracy

for i in range(-3, 4):
  svm_linear_mult = svm.SVC(kernel='linear', C=10**i)
  svm_linear_mult.fit(X_train, y_train)
  acc = cross_val_score(svm_linear_mult, X_train, y_train, cv=3, scoring="accuracy")
  ave = sum(acc)/len(acc)
  print(f"C = 10^{i}: {acc}, Average: {ave}")

# Since the kernel is linear, we can use any value for i

y_train_pred = cross_val_predict(svm_linear_mult, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx

# SVM with 2nd Order Kernel

# 1. Accuracy

for i in range(-3, 4):
  svm_second_mult = svm.SVC(kernel='poly', degree = 2, C=10**i)
  svm_second_mult.fit(X_train, y_train)
  acc = cross_val_score(svm_second_mult, X_train, y_train, cv=3, scoring="accuracy")
  ave = sum(acc)/len(acc)
  print(f"C = 10^{i}: {acc}, Average: {ave}")

# Accuracy score is the same for i = 1, 2, and 3

# 2. Confusion Matrix

y_train_pred = cross_val_predict(svm_second_mult, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx

# SVM with 3rd Order Kernel

# 1. Accuracy

for i in range(-3, 4):
  svm_third_mult = svm.SVC(kernel='poly', degree = 3, C=10**i)
  svm_third_mult.fit(X_train, y_train)
  acc = cross_val_score(svm_third_mult, X_train, y_train, cv=3, scoring="accuracy")
  ave = sum(acc)/len(acc)
  print(f"C = 10^{i}: {acc}, Average: {ave}")

# Accuracy score is the same for i=1,2 and 3

# 2. Confusion Matrix

y_train_pred = cross_val_predict(svm_third_mult, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx