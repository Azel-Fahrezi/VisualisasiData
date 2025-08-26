import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime
import numpy as np

def parse_sar_log(log_content, label):
    """
    Parse log SAR dan kembalikan DataFrame dengan label.
    Versi ini memperbaiki kesalahan parsing timestamp dan indeks data.
    """
    lines = log_content.strip().split('\n')
    
    cpu_data = []
    mem_data = []
    
    for line in lines:
        # Mencocokkan baris yang dimulai dengan format waktu (contoh: 03:47:38 PM)
        if re.match(r'^\d{2}:\d{2}:\d{2} [AP]M', line):
            parts = line.split()
            
            # PERBAIKAN 1: Format timestamp yang benar adalah 'HH:MM:SS AM/PM'
            # Sebelumnya: Anda menambahkan 'parts[2]' yang bisa berupa 'all' atau angka,
            # sehingga formatnya menjadi salah.
            timestamp = parts[0] + ' ' + parts[1]
            
            # PERBAIKAN 2: Kondisi untuk data CPU. 'all' ada di indeks ke-2, bukan ke-3.
            if len(parts) > 2 and parts[2] == 'all':
                # Pastikan baris memiliki cukup kolom untuk data CPU
                if len(parts) >= 9:
                    # PERBAIKAN 3: Indeks data CPU digeser satu ke kiri (mulai dari 3, bukan 4).
                    cpu_data.append([
                        timestamp,
                        float(parts[3]),  # %user
                        float(parts[4]),  # %nice
                        float(parts[5]),  # %system
                        float(parts[6]),  # %iowait
                        float(parts[7]),  # %steal
                        float(parts[8])   # %idle
                    ])
            # Cek apakah ini baris data memori (bukan header atau baris CPU)
            # Baris data memori akan memiliki angka di indeks ke-2.
            elif len(parts) > 2 and parts[2].isdigit():
                 # Pastikan baris memiliki cukup kolom untuk data memori
                if len(parts) >= 13:
                    # PERBAIKAN 4: Indeks data memori digeser satu ke kiri (mulai dari 2, bukan 3).
                    mem_data.append([
                        timestamp,
                        int(parts[2]),   # kbmemfree
                        int(parts[3]),   # kbavail
                        int(parts[4]),   # kbmemused
                        float(parts[5]), # %memused
                        int(parts[6]),   # kbbuffers
                        int(parts[7]),   # kbcached
                        int(parts[8]),   # kbcommit
                        float(parts[9]), # %commit
                        int(parts[10]),  # kbactive
                        int(parts[11]),  # kbinact
                        int(parts[12])   # kbdirty
                    ])
    
    # Buat DataFrames
    cpu_columns = ['Time', '%user', '%nice', '%system', '%iowait', '%steal', '%idle']
    mem_columns = ['Time', 'kbmemfree', 'kbavail', 'kbmemused', '%memused', 'kbbuffers', 
                   'kbcached', 'kbcommit', '%commit', 'kbactive', 'kbinact', 'kbdirty']
    
    cpu_df = pd.DataFrame(cpu_data, columns=cpu_columns)
    mem_df = pd.DataFrame(mem_data, columns=mem_columns)
    
    # Tambahkan label
    cpu_df['Config'] = label
    mem_df['Config'] = label
    
    # PERBAIKAN 5: Format konversi waktu disesuaikan. Hapus '%Z' karena tidak ada info zona waktu.
    cpu_df['Time'] = pd.to_datetime(cpu_df['Time'], format='%I:%M:%S %p', errors='coerce')
    mem_df['Time'] = pd.to_datetime(mem_df['Time'], format='%I:%M:%S %p', errors='coerce')
    
    # Hapus baris yang gagal di-parse waktunya (jika ada)
    cpu_df.dropna(subset=['Time'], inplace=True)
    mem_df.dropna(subset=['Time'], inplace=True)

    # Buat indeks waktu yang dimulai dari 0 (dalam menit)
    if not cpu_df.empty:
        cpu_df['Time_Index'] = (cpu_df['Time'] - cpu_df['Time'].min()).dt.total_seconds() / 60
    if not mem_df.empty:
        mem_df['Time_Index'] = (mem_df['Time'] - mem_df['Time'].min()).dt.total_seconds() / 60
    
    return cpu_df, mem_df

# --- Sisa kode Anda (tidak perlu diubah) ---

# Muat dan parse kedua file log
try:
    with open('with_selinux_cpu_mem.log', 'r') as file:
        with_selinux_content = file.read()

    with open('no_selinux_cpu_mem.log', 'r') as file:
        no_selinux_content = file.read()

    # Parse log
    cpu_df_selinux, mem_df_selinux = parse_sar_log(with_selinux_content, 'With SELinux')
    cpu_df_no_selinux, mem_df_no_selinux = parse_sar_log(no_selinux_content, 'Without SELinux')

    # Gabungkan data
    cpu_combined = pd.concat([cpu_df_selinux, cpu_df_no_selinux], ignore_index=True)
    mem_combined = pd.concat([mem_df_selinux, mem_df_no_selinux], ignore_index=True)

    # Buat visualisasi
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(16, 12))
    
    # Perbandingan Penggunaan CPU
    plt.subplot(2, 2, 1)
    for config in cpu_combined['Config'].unique():
        config_data = cpu_combined[cpu_combined['Config'] == config]
        plt.plot(config_data['Time_Index'], 100 - config_data['%idle'], 
                 label=config, linewidth=2, alpha=0.8)
    plt.title('CPU Utilization Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('CPU Usage (%)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Perbandingan Penggunaan Memori
    plt.subplot(2, 2, 2)
    for config in mem_combined['Config'].unique():
        config_data = mem_combined[mem_combined['Config'] == config]
        plt.plot(config_data['Time_Index'], config_data['%memused'], 
                 label=config, linewidth=2, alpha=0.8)
    plt.title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Memory Used (%)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Perbandingan Komitmen Memori
    plt.subplot(2, 2, 3)
    for config in mem_combined['Config'].unique():
        config_data = mem_combined[mem_combined['Config'] == config]
        plt.plot(config_data['Time_Index'], config_data['%commit'], 
                 label=config, linewidth=2, alpha=0.8)
    plt.title('Memory Commit Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Commit Percentage (%)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Perbandingan I/O Wait
    plt.subplot(2, 2, 4)
    for config in cpu_combined['Config'].unique():
        config_data = cpu_combined[cpu_combined['Config'] == config]
        plt.plot(config_data['Time_Index'], config_data['%iowait'], 
                 label=config, linewidth=2, alpha=0.8)
    plt.title('I/O Wait Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('I/O Wait (%)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(pad=3.0)
    plt.suptitle('SELinux Performance Comparison: CPU & Memory', fontsize=18, fontweight='bold')
    plt.subplots_adjust(top=0.92)
    plt.savefig('selinux_comparison_corrected.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Buat ringkasan statistik
    print("CPU Statistics Summary:")
    cpu_summary = cpu_combined.groupby('Config').agg({
        '%user': ['mean', 'max'],
        '%system': ['mean', 'max'],
        '%iowait': ['mean', 'max'],
        '%idle': ['mean', 'min']
    }).round(2)
    print(cpu_summary)

    print("\nMemory Statistics Summary:")
    mem_summary = mem_combined.groupby('Config').agg({
        '%memused': ['mean', 'max'],
        '%commit': ['mean', 'max'],
        'kbcached': 'mean'
    }).round(2)
    print(mem_summary)

    # Hitung perbedaan
    print("\nPerformance Differences (Mean Values):")
    mem_diff = mem_summary.loc['With SELinux', ('%memused', 'mean')] - mem_summary.loc['Without SELinux', ('%memused', 'mean')]
    commit_diff = mem_summary.loc['With SELinux', ('%commit', 'mean')] - mem_summary.loc['Without SELinux', ('%commit', 'mean')]
    iowait_diff = cpu_summary.loc['With SELinux', ('%iowait', 'mean')] - cpu_summary.loc['Without SELinux', ('%iowait', 'mean')]
    
    print(f"Memory Usage: {mem_diff:.2f}% {'higher' if mem_diff > 0 else 'lower'} with SELinux")
    print(f"Memory Commit: {commit_diff:.2f}% {'higher' if commit_diff > 0 else 'lower'} with SELinux")
    print(f"I/O Wait: {iowait_diff:.2f}% {'higher' if iowait_diff > 0 else 'lower'} with SELinux")

except FileNotFoundError as e:
    print(f"Error: File tidak ditemukan - {e}. Pastikan file log berada di direktori yang sama.")
except Exception as e:
    print(f"Terjadi kesalahan: {e}")