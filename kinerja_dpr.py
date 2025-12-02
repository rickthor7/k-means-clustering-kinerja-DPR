import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

np.random.seed(42)

N = 100 # Jumlah Anggota DPR RI🐀
nama_anggota = [f'Anggota_{i+1}' for i in range(N)]
fraksi_list = ['PDI-P', 'Golkar', 'Gerindra', 'NasDem', 'PKB', 'Demokrat', 'PKS', 'PAN', 'PPP']
fraksi = np.random.choice(fraksi_list, size=N, p=[0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.08, 0.07, 0.05])

indices = np.arange(N)
np.random.shuffle(indices)
N_tinggi, N_sedang, N_rendah = 30, 40, 30
idx_tinggi = indices[:N_tinggi]
idx_sedang = indices[N_tinggi:N_tinggi + N_sedang]
idx_rendah = indices[N_tinggi + N_sedang:]

kehadiran, ruu, partisipasi = np.zeros(N), np.zeros(N), np.zeros(N)

kehadiran[idx_tinggi] = np.random.uniform(90, 100, N_tinggi).round(2)
ruu[idx_tinggi] = np.random.randint(4, 9, N_tinggi)
partisipasi[idx_tinggi] = np.random.randint(35, 51, N_tinggi)

kehadiran[idx_sedang] = np.random.uniform(75, 90, N_sedang).round(2)
ruu[idx_sedang] = np.random.randint(2, 5, N_sedang)
partisipasi[idx_sedang] = np.random.randint(15, 36, N_sedang)

kehadiran[idx_rendah] = np.random.uniform(50, 75, N_rendah).round(2)
ruu[idx_rendah] = np.random.randint(0, 3, N_rendah)
partisipasi[idx_rendah] = np.random.randint(1, 16, N_rendah)

df_kinerja_dpr = pd.DataFrame({
    'Nama_Anggota': nama_anggota,
    'Fraksi': fraksi,
    'Kehadiran_Rapat_Persen': kehadiran,
    'RUU_Diinisiasi_Jumlah': ruu,
    'Partisipasi_Aktivitas_Skor': partisipasi
})

FITUR = ['Kehadiran_Rapat_Persen', 'RUU_Diinisiasi_Jumlah', 'Partisipasi_Aktivitas_Skor']
X = df_kinerja_dpr[FITUR]

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_normalized)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(9, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Metode Siku (Elbow Method) untuk Menentukan K Optimal')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show() 

K_OPTIMAL = 3 

kmeans = KMeans(n_clusters=K_OPTIMAL, init='k-means++', max_iter=300, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_normalized)
df_kinerja_dpr['Cluster_ID'] = cluster_labels

centroid_analysis = df_kinerja_dpr.groupby('Cluster_ID')[FITUR].mean()
centroid_analysis_sorted = centroid_analysis.sort_values(by='Kehadiran_Rapat_Persen', ascending=False)

print("----------------------------------------------------------")
print(f"Hasil Clustering K-Means dengan K={K_OPTIMAL}")
print("----------------------------------------------------------")
print("\n[Tabel Centroid] Rata-rata Kinerja Anggota per Cluster (Output Clustering):")

print("\n[Tabel Ringkasan] Jumlah Anggota dalam Setiap Cluster:")
print(df_kinerja_dpr['Cluster_ID'].value_counts().sort_index())
print("----------------------------------------------------------")



plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_kinerja_dpr['Kehadiran_Rapat_Persen'], 
                      df_kinerja_dpr['Partisipasi_Aktivitas_Skor'], 
                      c=df_kinerja_dpr['Cluster_ID'], 
                      cmap='viridis', s=100, alpha=0.8)

centroids = kmeans.cluster_centers_
centroid_denormalized = scaler.inverse_transform(centroids) 
df_centroids = pd.DataFrame(centroid_denormalized, columns=FITUR)

plt.scatter(df_centroids['Kehadiran_Rapat_Persen'], 
            df_centroids['Partisipasi_Aktivitas_Skor'], 
            marker='X', s=300, color='red', label='Centroid')

plt.title(f'Visualisasi Cluster Kinerja Anggota (K={K_OPTIMAL})')
plt.xlabel('Kehadiran Rapat (%)')
plt.ylabel('Partisipasi dan Aktivitas (Skor)')

unique_clusters = sorted(df_kinerja_dpr['Cluster_ID'].unique())
legend_labels = []

sorted_cluster_ids = centroid_analysis_sorted.index.tolist()
if len(sorted_cluster_ids) == 3:
    label_map = {
        sorted_cluster_ids[0]: 'Kinerja Tinggi',
        sorted_cluster_ids[1]: 'Kinerja Sedang',
        sorted_cluster_ids[2]: 'Kinerja Rendah'
    }
    legend_labels = [label_map.get(id, f'Cluster {id}') for id in sorted_cluster_ids]
else:
    legend_labels = [f'Cluster {id}' for id in unique_clusters]


plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Kelompok Kinerja")
plt.grid(True)
plt.show() 
