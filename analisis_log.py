import argparse
from pathlib import Path
import re
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Fungsi PARSER milik Anda (dipertahankan)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Utilitas: ekstraksi label dari nama file
# ---------------------------------------------------------------------------

def _normalize_scenario(raw: str) -> str:
    s = raw.replace('shuwdown', 'shutdown')
    s = s.replace('-', ' ').replace('_', ' ').strip()
    # Perbaiki frasa umum
    s = re.sub(r'\s+', ' ', s)
    return s.title()

def extract_meta_from_filename(fname: str):
    """
    Mengembalikan (label, scenario, protection).
    Label = "<Scenario Title> — <Protection Title>"
    """
    base = Path(fname).name
    name_wo_ext = base.rsplit('.', 1)[0]
    parts = name_wo_ext.split('_')
    # Default
    protection = 'No Protection'
    # Deteksi proteksi
    lower = name_wo_ext.lower()
    if 'selinux' in lower:
        protection = 'SELinux'
    elif 'apparmor' in lower:
        protection = 'AppArmor'
    elif 'no-protection' in lower or 'noprotection' in lower or 'no_protection' in lower:
        protection = 'No Protection'
    # Ekstrak skenario dari token setelah 2 token timestamp
    scenario_tokens = []
    if len(parts) > 2:
        # Ambil token mulai indeks 2 hingga sebelum token yang berisi proteksi/cpu/mem
        stop_words = {'selinux', 'apparmor', 'no-protection', 'noprotection', 'no', 'protection',
                      'cpu', 'mem', 'cpu_mem', 'cpumem'}
        for tok in parts[2:]:
            if tok.lower() in stop_words:
                break
            scenario_tokens.append(tok)
    scenario_raw = ' '.join(scenario_tokens) if scenario_tokens else 'unknown'
    scenario = _normalize_scenario(scenario_raw)
    label = f"{scenario} — {protection}"
    return label, scenario, protection

# ---------------------------------------------------------------------------
# Loading semua file log dalam folder
# ---------------------------------------------------------------------------

def load_logs_from_dir(input_dir: Path):
    input_dir = Path(input_dir)
    log_paths = sorted(input_dir.glob("*.log"))
    if not log_paths:
        raise FileNotFoundError(f"Tidak menemukan file .log di: {input_dir.resolve()}")
    cpu_frames = []
    mem_frames = []
    for p in log_paths:
        try:
            content = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            # Coba binary then decode fallback
            content = p.read_bytes().decode('utf-8', errors='ignore')
        label, scenario, protection = extract_meta_from_filename(p.name)
        cpu_df, mem_df = parse_sar_log(content, label)
        if not cpu_df.empty:
            cpu_df['Scenario'] = scenario
            cpu_df['Protection'] = protection
            cpu_df['Source_File'] = p.name
            cpu_frames.append(cpu_df)
        if not mem_df.empty:
            mem_df['Scenario'] = scenario
            mem_df['Protection'] = protection
            mem_df['Source_File'] = p.name
            mem_frames.append(mem_df)
    if not cpu_frames:
        raise RuntimeError("Tidak ada data CPU terbaca. Pastikan format log sesuai `sar -u ALL`.")
    if not mem_frames:
        raise RuntimeError("Tidak ada data Memori terbaca. Pastikan format log sesuai `sar -r`.")
    cpu_combined = pd.concat(cpu_frames, ignore_index=True)
    mem_combined = pd.concat(mem_frames, ignore_index=True)
    return cpu_combined, mem_combined

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def try_set_style():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        pass

def make_plots(cpu_combined: pd.DataFrame, mem_combined: pd.DataFrame, outpath: Path, title: str):
    try_set_style()
    plt.figure(figsize=(16, 12))

    # CPU Usage = 100 - %idle
    ax1 = plt.subplot(2, 2, 1)
    for config in cpu_combined['Config'].unique():
        d = cpu_combined[cpu_combined['Config'] == config]
        ax1.plot(d['Time_Index'], 100 - d['%idle'], label=config, linewidth=2, alpha=0.9)
    ax1.set_title('CPU Utilization Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('CPU Usage (%)')
    ax1.legend(); ax1.grid(True, linestyle='--', linewidth=0.5)

    # Memory %used
    ax2 = plt.subplot(2, 2, 2)
    for config in mem_combined['Config'].unique():
        d = mem_combined[mem_combined['Config'] == config]
        ax2.plot(d['Time_Index'], d['%memused'], label=config, linewidth=2, alpha=0.9)
    ax2.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Memory Used (%)')
    ax2.legend(); ax2.grid(True, linestyle='--', linewidth=0.5)

    # Memory %commit
    ax3 = plt.subplot(2, 2, 3)
    for config in mem_combined['Config'].unique():
        d = mem_combined[mem_combined['Config'] == config]
        ax3.plot(d['Time_Index'], d['%commit'], label=config, linewidth=2, alpha=0.9)
    ax3.set_title('Memory Commit Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Commit Percentage (%)')
    ax3.legend(); ax3.grid(True, linestyle='--', linewidth=0.5)

    # CPU %iowait
    ax4 = plt.subplot(2, 2, 4)
    for config in cpu_combined['Config'].unique():
        d = cpu_combined[cpu_combined['Config'] == config]
        ax4.plot(d['Time_Index'], d['%iowait'], label=config, linewidth=2, alpha=0.9)
    ax4.set_title('I/O Wait Comparison', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time (minutes)')
    ax4.set_ylabel('I/O Wait (%)')
    ax4.legend(); ax4.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout(pad=3.0)
    plt.suptitle(title, fontsize=18, fontweight='bold')
    plt.subplots_adjust(top=0.92)
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    # Jangan plt.show() saat headless
    print(f"[OK] Grafik tersimpan: {outpath}")

# ---------------------------------------------------------------------------
# Ringkasan & perbandingan
# ---------------------------------------------------------------------------

def print_summaries(cpu_combined: pd.DataFrame, mem_combined: pd.DataFrame):
    print("CPU Statistics Summary (per Config):")
    cpu_summary = cpu_combined.groupby('Config').agg({
        '%user': ['mean', 'max'],
        '%system': ['mean', 'max'],
        '%iowait': ['mean', 'max'],
        '%idle': ['mean', 'min']
    }).round(2)
    print(cpu_summary)

    print("\nMemory Statistics Summary (per Config):")
    mem_summary = mem_combined.groupby('Config').agg({
        '%memused': ['mean', 'max'],
        '%commit': ['mean', 'max'],
        'kbcached': 'mean'
    }).round(2)
    print(mem_summary)
    return cpu_summary, mem_summary

def print_pairwise_differences(cpu_combined: pd.DataFrame, mem_combined: pd.DataFrame):
    # Hitung rata-rata per (Scenario, Protection)
    cpu_means = cpu_combined.groupby(['Scenario', 'Protection']).agg({
        '%iowait': 'mean',
        '%user': 'mean',
        '%system': 'mean',
        '%idle': 'mean'
    })
    mem_means = mem_combined.groupby(['Scenario', 'Protection']).agg({
        '%memused': 'mean',
        '%commit': 'mean'
    })
    scenarios = sorted(set(cpu_combined['Scenario']) | set(mem_combined['Scenario']))
    print("\nPerformance Differences (Mean Values) — Pairwise per Scenario:")
    for sc in scenarios:
        protos = ['SELinux', 'AppArmor', 'No Protection']
        available = [p for p in protos if (sc, p) in cpu_means.index and (sc, p) in mem_means.index]
        # Prioritas banding SELinux vs AppArmor, lalu SELinux vs No Protection
        pairs = []
        if 'SELinux' in available and 'AppArmor' in available:
            pairs.append(('SELinux', 'AppArmor'))
        if 'SELinux' in available and 'No Protection' in available:
            pairs.append(('SELinux', 'No Protection'))
        if not pairs and len(available) >= 2:
            # Ambil pasangan pertama tersedia
            pairs.append((available[0], available[1]))
        if not pairs:
            continue
        print(f"\n[Skenario] {sc}:")
        for a, b in pairs:
            iowait_diff = cpu_means.loc[(sc, a), '%iowait'] - cpu_means.loc[(sc, b), '%iowait']
            mem_diff = mem_means.loc[(sc, a), '%memused'] - mem_means.loc[(sc, b), '%memused']
            commit_diff = mem_means.loc[(sc, a), '%commit'] - mem_means.loc[(sc, b), '%commit']
            print(f"  {a} vs {b}: ΔIOwait={iowait_diff:.2f} pp, ΔMemUsed={mem_diff:.2f} pp, ΔCommit={commit_diff:.2f} pp")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Analisis log sar CPU/Mem untuk beberapa konfigurasi proteksi.")
    ap.add_argument("-i", "--input", default="log_cpu_mem",
                    help="Folder berisi file *.log (default: log_cpu_mem)")
    ap.add_argument("-o", "--output", default="selinux_apparmor_comparison.png",
                    help="Path file PNG output grafik")
    ap.add_argument("--title", default="Performance Comparison: CPU & Memory",
                    help="Judul utama grafik")
    args = ap.parse_args()

    try:
        cpu_combined, mem_combined = load_logs_from_dir(Path(args.input))
        make_plots(cpu_combined, mem_combined, Path(args.output), args.title)
        cpu_summary, mem_summary = print_summaries(cpu_combined, mem_combined)
        print_pairwise_differences(cpu_combined, mem_combined)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Terjadi kesalahan: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()