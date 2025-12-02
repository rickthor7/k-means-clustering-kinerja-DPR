import pandas as pd
import numpy as np

# Set seed untuk hasil yang bisa direproduksi
np.random.seed(42)

# n buat Jumlah Anggota
N = 100

# 1. GENERASI DATA ANGGOTA (Nama dan Fraksi)
nama_anggota = [f'Anggota_{i+1}' for i in range(N)]
fraksi_list = ['PDI-P', 'Golkar', 'Gerindra', 'NasDem', 'PKB', 'Demokrat', 'PKS', 'PAN', 'PPP']
fraksi = np.random.choice(fraksi_list, size=N, p=[0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.08, 0.07, 0.05]) # Distribusi mirip pemilu

# 2. GENERASI VARIABEL KINERJA (Fokus pada Penurunan/Variasi)

# 3 kelompok (Cluster) yang berbeda secara implisit dalam data:
# Cluster 1: Kinerja Tinggi (sekitar 30% dari anggota)
# Cluster 2: Kinerja Sedang (sekitar 40% dari anggota)
# Cluster 3: Kinerja Rendah (sekitar 30% dari anggota)

indices = np.arange(N)
np.random.shuffle(indices)

# Alokasi jumlah anggota per kelompok
N_tinggi = 30
N_sedang = 40
N_rendah = 30

idx_tinggi = indices[:N_tinggi]
idx_sedang = indices[N_tinggi:N_tinggi + N_sedang]
idx_rendah = indices[N_tinggi + N_sedang:]

# Inisialisasi array kosong untuk fitur
kehadiran = np.zeros(N)
ruu = np.zeros(N)
partisipasi = np.zeros(N)

# A. Data Kinerja Tinggi (Kehadiran Tinggi, RUU Banyak, Partisipasi Tinggi)
kehadiran[idx_tinggi] = np.random.uniform(90, 100, N_tinggi).round(2)
ruu[idx_tinggi] = np.random.randint(4, 9, N_tinggi)
partisipasi[idx_tinggi] = np.random.randint(35, 51, N_tinggi)

# B. Data Kinerja Sedang (Kehadiran Menengah, RUU Sedang, Partisipasi Sedang)
kehadiran[idx_sedang] = np.random.uniform(75, 90, N_sedang).round(2)
ruu[idx_sedang] = np.random.randint(2, 5, N_sedang)
partisipasi[idx_sedang] = np.random.randint(15, 36, N_sedang)

# C. Data Kinerja Rendah (Kehadiran Rendah/Menurun, RUU Sedikit, Partisipasi Rendah)
kehadiran[idx_rendah] = np.random.uniform(50, 75, N_rendah).round(2)
ruu[idx_rendah] = np.random.randint(0, 3, N_rendah)
partisipasi[idx_rendah] = np.random.randint(1, 16, N_rendah)

# 3. BUAT DATAFRAME AKHIR
df_kinerja_dummy = pd.DataFrame({
    'Nama_Anggota': nama_anggota,
    'Fraksi': fraksi,
    'Kehadiran_Rapat_Persen': kehadiran,
    'RUU_Diinisiasi_Jumlah': ruu,
    'Partisipasi_Aktivitas_Skor': partisipasi
})

# Tampilkan 10 data teratas dan informasi umum
print(f"--- Data Dummy Kinerja Anggota DPR RI (Total {N} Anggota) ---")
print("\n10 Data Teratas:")
print(df_kinerja_dummy.head(10))

print("\nStatistik Deskriptif (Ringkasan Variasi Data):")
print(df_kinerja_dummy[['Kehadiran_Rapat_Persen', 'RUU_Diinisiasi_Jumlah', 'Partisipasi_Aktivitas_Skor']].describe().T)

df_kinerja_dummy.to_csv('kinerja_anggota_dpr.csv', index=False)
