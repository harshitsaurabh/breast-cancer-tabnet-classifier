import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# =======================
# Load and Preprocess Data
# =======================
data = pd.read_csv('data/breast-cancer-wisconsin-data_data.csv')
data.drop(columns={'id', 'Unnamed: 32'}, inplace=True)

Target = data['diagnosis']
Features = data.drop(columns={'diagnosis'})

encoder = LabelEncoder()
Target = encoder.fit_transform(Target)
print("Class Labels:", encoder.classes_)

scaler = StandardScaler()
Features = scaler.fit_transform(Features)

# =======================
# Train-Test Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    Features, Target, test_size=0.3, random_state=41
)

# =======================
# Model Training
# =======================
TN_model = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    scheduler_params={"step_size": 10, "gamma": 0.95},
    scheduler_fn=torch.optim.lr_scheduler.StepLR
)

TN_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_name=['train', 'test'],
    eval_metric=['auc', 'balanced_accuracy'],
    max_epochs=200, patience=60,
    batch_size=512, virtual_batch_size=512,
    num_workers=0,
    weights=1,
    drop_last=False
)

# =======================
# Model Evaluation
# =======================
y_pred = TN_model.predict(X_test)
y_prob = TN_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
print("\n================= Test Result ========================")
print(f"Accuracy: %{100*acc:6.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC Curve
auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title("ROC Curve")
plt.savefig("roc_curve.png")
plt.show()

# Feature Importance
feature_importances = TN_model.feature_importances_
plt.bar(np.arange(len(feature_importances)), feature_importances)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("TabNet Feature Importances")
plt.savefig("feature_importance.png")
plt.show()

# Save Model
TN_model.save_model('tabnet_bc_model.zip')
