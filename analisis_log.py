import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime

def parse_sar_log(filename):
    data = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
            # Cari baris yang berisi data (bukan header)
            data_started = False
            for line in lines:
                # Cari baris yang berisi timestamp dengan format waktu
                if re.match(r'^\d{1,2}:\d{2}:\d{2}\s+(AM|PM|\d)', line) or re.match(r'^\d{1,2}:\d{2}:\d{2}', line):
                    parts = line.split()
                    if len(parts) >= 10:
                        try:
                            timestamp = parts[0]
                            cpu_idle = float(parts[7])
                            mem_used = float(parts[9])  # %memused
                            data.append([timestamp, cpu_idle, mem_used])
                        except (ValueError, IndexError):
                            continue
    except FileNotFoundError:
        print(f"Error: File {filename} tidak ditemukan.")
        return pd.DataFrame(columns=['Time', 'CPU_Idle', 'Mem_Used'])
    
    return pd.DataFrame(data, columns=['Time', 'CPU_Idle', 'Mem_Used'])

# Load data
print("Memuat data log...")
df_no_selinux = parse_sar_log('no_selinux_cpu_mem.log')
df_with_selinux = parse_sar_log('with_selinux_cpu_mem.log')

# Periksa apakah data berhasil dimuat
if df_no_selinux.empty or df_with_selinux.empty:
    print("Error: Data tidak dapat dimuat. Pastikan file log ada dan formatnya benar.")
    exit()

# Konversi waktu ke format datetime untuk plotting yang lebih baik
try:
    time_format = '%H:%M:%S'
    df_no_selinux['Time'] = pd.to_datetime(df_no_selinux['Time'], format=time_format)
    df_with_selinux['Time'] = pd.to_datetime(df_with_selinux['Time'], format=time_format)
except ValueError:
    print("Format waktu tidak sesuai. Menggunakan indeks sebagai sumbu X.")
    # Jika format waktu tidak sesuai, gunakan indeks
    df_no_selinux['Time'] = df_no_selinux.index
    df_with_selinux['Time'] = df_with_selinux.index

# Hitung statistik
cpu_avg_no_selinux = (100 - df_no_selinux['CPU_Idle']).mean()
cpu_avg_with_selinux = (100 - df_with_selinux['CPU_Idle']).mean()
mem_avg_no_selinux = df_no_selinux['Mem_Used'].mean()
mem_avg_with_selinux = df_with_selinux['Mem_Used'].mean()

print(f"Rata-rata CPU tanpa SELinux: {cpu_avg_no_selinux:.2f}%")
print(f"Rata-rata CPU dengan SELinux: {cpu_avg_with_selinux:.2f}%")
print(f"Rata-rata RAM tanpa SELinux: {mem_avg_no_selinux:.2f}%")
print(f"Rata-rata RAM dengan SELinux: {mem_avg_with_selinux:.2f}%")

# Plotting
plt.figure(figsize=(15, 10))

# CPU Usage
plt.subplot(2, 1, 1)
plt.plot(df_no_selinux['Time'], 100 - df_no_selinux['CPU_Idle'], 'b-', label='Tanpa SELinux', alpha=0.7)
plt.plot(df_with_selinux['Time'], 100 - df_with_selinux['CPU_Idle'], 'r-', label='Dengan SELinux', alpha=0.7)
plt.ylabel('Penggunaan CPU (%)')
plt.title('Perbandingan Penggunaan CPU: SELinux vs Non-SELinux')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Tambah garis rata-rata
plt.axhline(y=cpu_avg_no_selinux, color='blue', linestyle='--', alpha=0.5, label=f'Rata-rata tanpa SELinux: {cpu_avg_no_selinux:.2f}%')
plt.axhline(y=cpu_avg_with_selinux, color='red', linestyle='--', alpha=0.5, label=f'Rata-rata dengan SELinux: {cpu_avg_with_selinux:.2f}%')
plt.legend()

# Memory Usage
plt.subplot(2, 1, 2)
plt.plot(df_no_selinux['Time'], df_no_selinux['Mem_Used'], 'b-', label='Tanpa SELinux', alpha=0.7)
plt.plot(df_with_selinux['Time'], df_with_selinux['Mem_Used'], 'r-', label='Dengan SELinux', alpha=0.7)
plt.ylabel('Penggunaan RAM (%)')
plt.xlabel('Waktu')
plt.title('Perbandingan Penggunaan Memori: SELinux vs Non-SELinux')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Tambah garis rata-rata
plt.axhline(y=mem_avg_no_selinux, color='blue', linestyle='--', alpha=0.5, label=f'Rata-rata tanpa SELinux: {mem_avg_no_selinux:.2f}%')
plt.axhline(y=mem_avg_with_selinux, color='red', linestyle='--', alpha=0.5, label=f'Rata-rata dengan SELinux: {mem_avg_with_selinux:.2f}%')
plt.legend()

plt.tight_layout()

# Simpan gambar dengan kualitas tinggi
plt.savefig('selinux_performance_comparison.png', dpi=300, bbox_inches='tight')
print("Grafik berhasil disimpan sebagai 'selinux_performance_comparison.png'")

# Tampilkan grafik (opsional, bisa dihapus jika tidak diperlukan)
# plt.show()