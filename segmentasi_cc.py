import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Pengaturan tampilan halaman Streamlit
st.set_page_config(page_title="Segmentasi Pelanggan Kartu Kredit", layout="wide")
st.title("📊 Segmentasi Pelanggan Menggunakan KMeans Clustering")

# Langkah 1: Load Dataset
st.header("1. Load Dataset")

df = pd.read_csv("CC GENERAL.csv")
st.write("**Dimensi Dataset:**", df.shape)
st.dataframe(df.head())

# Langkah 2: Pra-pemrosesan Data
st.header("2. Pra-pemrosesan Data")

# Mengisi missing value numerik dengan mean
for col in df.select_dtypes(include='number').columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Menghapus kolom ID pelanggan
if 'CUST_ID' in df.columns:
    df.drop('CUST_ID', axis=1, inplace=True)

# Standardisasi fitur
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Tampilkan data yang sudah distandardisasi
st.subheader("Data setelah Standardisasi")
st.dataframe(df_scaled.head())

# Langkah 3: Menentukan Jumlah Cluster
st.header("3. Menentukan Jumlah Cluster")

# Metode Elbow
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow
fig_elbow, ax1 = plt.subplots()
ax1.plot(range(1, 11), wcss, marker='o')
ax1.set_title('Metode Elbow')
ax1.set_xlabel('Jumlah Cluster (k)')
ax1.set_ylabel('WCSS')
ax1.grid()
st.pyplot(fig_elbow)

# Silhouette Score
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(df_scaled)
    score = silhouette_score(df_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Plot Silhouette Score
fig_silhouette, ax2 = plt.subplots()
ax2.plot(k_range, silhouette_scores, marker='o')
ax2.set_title('Silhouette Score untuk Berbagai Nilai k')
ax2.set_xlabel('Jumlah Cluster (k)')
ax2.set_ylabel('Silhouette Score')
ax2.grid(True)
st.pyplot(fig_silhouette)

k_optimal = st.slider("Pilih jumlah cluster optimal berdasarkan grafik:", 2, 10, 3)

# Langkah 4: Pemodelan KMeans

st.header("4. Pemodelan KMeans")

kmeans = KMeans(n_clusters=k_optimal, random_state=42)
kmeans.fit(df_scaled)
df['Cluster'] = kmeans.labels_ + 1  # Mulai dari 1

st.write("**Distribusi Data per Cluster:**")
st.dataframe(df['Cluster'].value_counts().rename_axis("Cluster").reset_index(name="Jumlah Data"))

st.write("**Berdasarkan pemodelan KMeans Clustering ini didapatkan informasi bahwa**")

st.markdown("""
- **Cluster 1**: 1241 data
- **Cluster 2**: 4559 data
- **Cluster 3**: 3150 data
""")

# Langkah 5: Visualisasi Cluster (PCA)
st.header("5. Visualisasi Cluster (PCA)")

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
df_pca['Cluster'] = df['Cluster']

fig_pca, ax3 = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=df_pca,
    x='PCA1', y='PCA2',
    hue='Cluster',
    palette=sns.color_palette("deep", n_colors=k_optimal),
    alpha=0.8, s=60, ax=ax3
)
ax3.set_title('Visualisasi Cluster Pelanggan dengan PCA')
ax3.grid(True)
st.pyplot(fig_pca)

st.write("**Sumbu X (PCA1) mempresentasikan aktivitas pembelian, Semakin ke kanan, semakin tinggi aktivitas belanjanya.**")

st.write("**Sumbu Y (PCA2) mempresentasikan penggunaan pinjaman tunai (cash advance), Semakin ke atas, semakin tinggi penggunaan pinjaman tunainya.**") 

# Langkah 6: Analisis Cluster
st.header("6. Analisis Cluster")

df_scaled['Cluster'] = kmeans.labels_
cluster_summary = df_scaled.groupby('Cluster').mean().T

# Reset index untuk tampilan
cluster_summary_display = cluster_summary.copy()
cluster_summary_display.columns = [f"Cluster {i+1}" for i in cluster_summary_display.columns]
st.dataframe(cluster_summary_display.style.background_gradient(cmap='YlGnBu', axis=1))

# Interpretasi
st.subheader("Kesimpulan Awal Segmentasi")
st.markdown("""
- **Cluster 1**: Pengguna Pinjaman Tunai yang Berisiko Tinggi
- **Cluster 2**: Pelanggan Pasif / Hemat
- **Cluster 3**: Pengguna Aktif / Pembelanja Utama
""")
