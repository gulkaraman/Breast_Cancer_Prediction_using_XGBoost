# Gerekli kütüphanelerin yüklenmesi
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Veri Kümesini Yükleme
# Veri setini CSV dosyasından okuma
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
# columns = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
# data = pd.read_csv(url, names=columns)
data = pd.read_csv("breast_cancer.csv")

# Veri hakkında genel bilgiler
total_data = len(data)
print("Toplam veri sayısı:", total_data)
print(data.info())

# Veri kümesinde eksik değerleri "?" ile temsil ediyor, bu değerleri NaN olarak değiştirelim
data.replace("?", np.nan, inplace=True)

# Eksik değerleri ortalama değerle doldurma ve sütunu sayısal bir formata dönüştürme
data["Bare Nuclei"] = data["Bare Nuclei"].astype(float)  # Önce float'a dönüştürmek gerekebilir
data["Bare Nuclei"] = data["Bare Nuclei"].fillna(data["Bare Nuclei"].median())
data["Bare Nuclei"] = data["Bare Nuclei"].astype(int)

# "Class" sütunundaki değerleri 0 ve 1 olarak değiştirelim (0: benign, 1: malignant)
data["Class"] = data["Class"].map({2: 0, 4: 1})

# Hastalikli ve saglikli veri sayilari
count_classes = pd.value_counts(data['Class'], sort=True)
count_classes.plot(kind='bar', rot=0)
plt.title("Hastalikli ve Saglikli Veri Sayisi")
plt.xticks(range(2), ['Saglikli', 'Hastalikli'])
plt.xlabel("Durum")
plt.ylabel("Frekans")
plt.show()

# Veriyi görselleştirme
sns.countplot(x='Class', data=data)
plt.title("Hastalikli ve Saglikli Veri Sayisi")
plt.xlabel("Durum")
plt.ylabel("Frekans")
plt.show()

# 2. Veri Ön İşleme
# Veri setini bağımsız değişkenler (X) ve hedef değişken (y) olarak bölelim
X = data.drop(["Sample code number", "Class"], axis=1)
y = data["Class"]

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. XGBoost Modelini Oluşturma
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# 4. Modeli Eğitme
model.fit(X_train, y_train)

# 5. Modelin Değerlendirilmesi
y_pred = model.predict(X_test)

# 6. Performans Metriklerinin Hesaplanması
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Confusion matrix hesaplayalım
cm = confusion_matrix(y_test, y_pred)

# Veri setinin ilk 10 satırını sütunlarıyla birlikte gösterme
print("Veri Setinin İlk 10 Satırı:\n", data.head(10).to_string())

# 7. Görselleştirmeler
# Confusion matrix görselleştirmesi
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("Karmaşıklık Matrisi")
plt.show()

# Diğer performans metriklerini gösterelim
print("Doğruluk (Accuracy):", accuracy)
print("F1 Skoru:", f1)
print("Hassasiyet (Precision):", precision)
print("Duyarlılık (Recall):", recall)

# Özelliklerin dağılımını görselleştirme - Histogram
plt.figure(figsize=(12, 12))
for i, feature in enumerate(X.columns):
    plt.subplot(4, 5, i + 1)
    sns.histplot(data[feature], kde=True)
    plt.title(feature)
plt.tight_layout()
plt.show()

# Özelliklerin dağılımını görselleştirme - Violin Plot
plt.figure(figsize=(12, 12))
for i, feature in enumerate(X.columns):
    plt.subplot(4, 5, i + 1)
    sns.violinplot(y=data[feature])
    plt.title(feature)
plt.tight_layout()
plt.show()

# Hedef değişkenin dağılımını görselleştirme
plt.figure(figsize=(6, 4))
sns.countplot(x=data["Class"])
plt.title("Sınıf Dağılımı")
plt.xlabel("Sınıf")
plt.ylabel("Sayı")
plt.show()
