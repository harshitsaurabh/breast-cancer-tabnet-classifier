import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

data = pd.read_csv('/Users/harshitsaurabh/Downloads/breast-cancer-wisconsin-data_data.csv')
data.drop(columns={'id','Unnamed: 32'},inplace=True)
Target = data['diagnosis']
Features = data.drop(columns={'diagnosis'})

encoder=LabelEncoder()
Target = encoder.fit_transform(Target)
print("class Labels :", encoder.classes_)

scaler = StandardScaler()
Features = scaler.fit_transform(Features).squeeze()
Features.shape, Target.shape

encoder=LabelEncoder()
Target = encoder.fit_transform(Target)
print("class Labels :", encoder.classes_)

scaler = StandardScaler()
Features = scaler.fit_transform(Features).squeeze()
Features.shape, Target.shape


X_train,X_test,y_train,y_test=train_test_split (Features,Target,test_size=0.3, random_state=41)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

TN_model= TabNetClassifier(optimizer_fn=torch.optim.Adam,
                       scheduler_params={"step_size":10,
                                         "gamma":0.95},
                       scheduler_fn=torch.optim.lr_scheduler.StepLR,
                      )
TN_model.fit(
    X_train ,y_train,
    eval_set=[(X_train, y_train), (X_test , y_test)],
    eval_name=['train', 'test'],
    eval_metric=['auc','balanced_accuracy'],
    max_epochs=200, patience=60,
    batch_size=512, virtual_batch_size=512,
    num_workers=0,
    weights=1,
    drop_last=False
)

y_pred = TN_model.predict(X_test)
Acc = accuracy_score(y_test, y_pred)

print("\n================= Test Result ========================")
print(f"Accuracy: %{100*Acc:6.2f}  ")
print("_______________________________________________________\n Classification Report:")
print(classification_report(y_test, y_pred))
print("_______________________________________________________\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
