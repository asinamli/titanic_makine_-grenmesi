import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid", palette="pastel")
plt.rcParams['figure.figsize'] = (10, 6)


df = pd.read_csv("titanic.csv")

#  Eksik Veriler
plt.figure(figsize=(8, 5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Eksik Verilerin Görselleştirilmesi")
plt.show()


df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(['Cabin', 'Ticket'], axis=1, inplace=True)


print("Eksik Verilerin Sayısı (Doldurulduktan Sonra):\n", df.isnull().sum())


plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=30, kde=True, color="skyblue")
plt.title("Yaş Dağılımı")
plt.xlabel("Yaş")
plt.ylabel("Frekans")
plt.show()

# ( one-hot-encoding)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)


X = df.drop(['Survived', 'Name', 'PassengerId'], axis=1)   # X: Modelin öğrenmesi için kullanılacak özellikler (yaş, cinsiyet, bilet sınıfı vb.).
y = df['Survived']   # y: Tahmin edilmesi gereken hedef değişken


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  (PCA öncesi ilk iki özellik)
plt.figure(figsize=(10, 5))
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], color='blue', label='Eğitim Verisi', alpha=0.5)
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], color='red', label='Test Verisi', alpha=0.5)
plt.title("Eğitim ve Test Verilerinin Dağılımı")
plt.xlabel("Özellik 1")
plt.ylabel("Özellik 2")
plt.legend()
plt.show()


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. PCA Uygulama
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# PCA Sonrası
plt.figure(figsize=(8, 5))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], color='blue', label='Eğitim Verisi', alpha=0.5)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], color='red', label='Test Verisi', alpha=0.5)
plt.title("PCA Sonrası Eğitim ve Test Verileri")
plt.xlabel("Bileşen 1")
plt.ylabel("Bileşen 2")
plt.legend()
plt.show()


log_reg_model = LogisticRegression()
log_reg_model.fit(X_train_scaled, y_train)
y_pred_original = log_reg_model.predict(X_test_scaled)

log_reg_model.fit(X_train_pca, y_train)
y_pred_pca = log_reg_model.predict(X_test_pca)

accuracy_original = accuracy_score(y_test, y_pred_original)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

print(f"Lojistik Regresyon Doğruluk Oranı (Orijinal): {accuracy_original:.2f}")
print(f"Lojistik Regresyon Doğruluk Oranı (PCA): {accuracy_pca:.2f}")


kmeans_original = KMeans(n_clusters=2, random_state=42)
kmeans_clusters_original = kmeans_original.fit_predict(X_test_scaled)

kmeans_pca = KMeans(n_clusters=2, random_state=42)
kmeans_clusters_pca = kmeans_pca.fit_predict(X_test_pca)

silhouette_original = silhouette_score(X_test_scaled, kmeans_clusters_original)
silhouette_pca = silhouette_score(X_test_pca, kmeans_clusters_pca)

print(f"K-Means Silhouette Skoru (Orijinal): {silhouette_original:.2f}")
print(f"K-Means Silhouette Skoru (PCA): {silhouette_pca:.2f}")


plt.figure(figsize=(8, 5))
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=kmeans_clusters_original, cmap='viridis', s=50)
plt.title("K-Means Kümeleme - Orijinal Veriler")
plt.xlabel("Özellik 1")
plt.ylabel("Özellik 2")
plt.colorbar(label="Küme")
plt.show()



cm_original = confusion_matrix(y_test, y_pred_original)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_original, annot=True, fmt='d', cmap="Blues", xticklabels=["Ölmedi", "Öldü"], yticklabels=["Ölmedi", "Öldü"])
plt.title("Confusion Matrix (Lojistik Regresyon - Orijinal Veriler)")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()


plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred_original, cmap='coolwarm', s=50)
plt.title("Orijinal Veriler - Logistic Regression")

plt.subplot(2, 2, 2)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred_pca, cmap='coolwarm', s=50)
plt.title("PCA Verileri - Logistic Regression")

plt.subplot(2, 2, 3)
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=kmeans_clusters_original, cmap='viridis', s=50)
plt.title("Orijinal Veriler - K-Means")

plt.subplot(2, 2, 4)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=kmeans_clusters_pca, cmap='viridis', s=50)
plt.title("PCA Verileri - K-Means")

plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Sex_male', hue='Survived', palette='Set1')
plt.title("Cinsiyete Göre Hayatta Kalma Oranı")
plt.xlabel("Cinsiyet (0: Kadın, 1: Erkek)")
plt.ylabel("Frekans")
plt.legend(title="Hayatta Kalma Durumu", labels=["Ölmedi", "Öldü"])
plt.show()


embarked_columns = ['Embarked_C', 'Embarked_Q', 'Embarked_S']


for col in embarked_columns:
    if col in df.columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=col, hue='Survived', palette='Set1')
        plt.title(f"{col} Değişkenine Göre Hayatta Kalma Oranı")
        plt.xlabel(col)
        plt.ylabel("Frekans")
        plt.legend(title="Hayatta Kalma Durumu", labels=["Ölmedi", "Öldü"])
        plt.show()


sns.barplot(x='Pclass', y='Survived', data=df, hue='Pclass', palette="muted")
plt.title("Bilet Sınıfına Göre Hayatta Kalma Oranı")
plt.xlabel("Bilet Sınıfı")
plt.ylabel("Hayatta Kalma Oranı")
plt.show()


plt.figure(figsize=(8, 5))
plt.bar(['Lojistik Regresyon (Orijinal)', 'Lojistik Regresyon (PCA)', 'K-Means (Orijinal)', 'K-Means (PCA)'],
        [accuracy_original, accuracy_pca, silhouette_original, silhouette_pca],
        color=['blue', 'cyan', 'green', 'lime'])
plt.title('Sınıflandırma ve Kümeleme Doğruluk Oranları Karşılaştırması')
plt.ylabel('Skor')

plt.show()
