import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Impor library untuk Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# ==========================================
# STEP 1: READ THE DATASET & CLEANING
# ==========================================
print("--- Membaca dan Membersihkan Data ---")
data = pd.read_csv("BreastCancerData (4).csv")

# Membuang kolom yang tidak perlu (Unnamed: 32 dan id)
data.drop(labels=['Unnamed: 32', 'id'], axis=1, inplace=True, errors='ignore')

# Mapping fitur target 'diagnosis' menjadi biner (Benign = 0, Malignant = 1)
data['diagnosis'] = data['diagnosis'].replace({'B': 0, 'M': 1})

# ==========================================
# STEP 2: FEATURE SELECTION (ABSOLUT PEARSON CORRELATION)
# ==========================================
print("\n--- Feature Selection menggunakan Pearson Correlation (Absolut) ---")
corr_matrix = data.corr()

# Mengambil nilai absolut korelasi untuk mempertimbangkan korelasi negatif yang kuat
abs_corr_with_diagnosis = corr_matrix['diagnosis'].abs()

# Mengambil 8 fitur teratas (index 0 adalah diagnosis, jadi kita ambil index 1 sampai 9)
strong_relation_features = abs_corr_with_diagnosis.nlargest(9).index[1:]
print(f"Top 8 Fitur Terpilih: \n{list(strong_relation_features)}")

data_selected = data[list(strong_relation_features) + ['diagnosis']]

# ==========================================
# STEP 3: PLOT DISTRIBUSI (OVERLAPPING ANALYSIS)
# ==========================================
# Mengadaptasi script asli Anda untuk melihat overlap dari 8 fitur terbaik
print("\n--- Plot Distribusi Fitur (Ideal Distribution Analysis) ---")
fig, axes = plt.subplots(4, 2, figsize=(15, 20))
axes = axes.flatten()

for i, feature in enumerate(strong_relation_features):
    # Plot histogram + KDE (density)
    sns.histplot(data=data, x=feature, hue='diagnosis', stat='density', bins=30, kde=True,
                 palette=['#1f77b4', '#d62728'], element='step', ax=axes[i])
    axes[i].set_title(f'Distribution of {feature}')
    
plt.tight_layout()
plt.show()

# ==========================================
# STEP 4: PLOT HEATMAP FITUR TERPILIH
# ==========================================
plt.figure(figsize=(10, 8))
mask = np.zeros_like(data_selected.corr())
mask[np.triu_indices_from(mask)] = True

# Menampilkan korelasi ASLI (dengan vmin=-1, vmax=1)
sns.heatmap(data=data_selected.corr(), vmin=-1, vmax=1, mask=mask, 
            square=True, annot=True, cmap="vlag")
plt.title("Heatmap 8 Fitur Terpilih (Korelasi Asli)")
plt.show()

# ==========================================
# STEP 5: DATA SPLITTING & SCALING
# ==========================================
X = data_selected.drop('diagnosis', axis=1)
y = data_selected['diagnosis']

# Membagi data 75% Train, 25% Test secara Stratified (seimbang rasionya)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Feature Scaling (Wajib terutama untuk SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# STEP 6: MODEL TRAINING & PERFORMANCE EVALUATION
# ==========================================
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "Support Vector Machine (SVM)": SVC(kernel='linear', random_state=42)
}

for name, model in models.items():
    print(f"\n{'='*50}\nMODEL: {name}\n{'='*50}")
    
    # Training Model
    model.fit(X_train_scaled, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test_scaled)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Memecah isi confusion matrix (berlaku untuk binary classification 2x2)
    # Ravel mengurutkan dari: TN (0,0), FP (0,1), FN (1,0), TP (1,1)
    TN, FP, FN, TP = cm.ravel()
    
    # Visualisasi Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign (0)', 'Malignant (1)'],
                yticklabels=['Benign (0)', 'Malignant (1)'])
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # Perhitungan Manual Sesuai Slide 35-38 (Measures of Test Performance)
    accuracy    = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)  # Disebut juga True Positive Rate (TPR) / Recall
    specificity = TN / (TN + FP)  # Disebut juga True Negative Rate (TNR)
    precision   = TP / (TP + FP) if (TP + FP) != 0 else 0
    
    print(f"--- Performance Analysis untuk {name} ---")
    print(f"True Positives (TP) : {TP}")
    print(f"False Negatives (FN): {FN}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN) : {TN}")
    print("-" * 40)
    print(f"Accuracy                   : {accuracy * 100:.2f}%")
    print(f"Sensitivity (Recall/TPR)   : {sensitivity * 100:.2f}%")
    print(f"Specificity (TNR)          : {specificity * 100:.2f}%")
    print(f"Precision                  : {precision * 100:.2f}%\n")