import argparse
from pathlib import Path
import re
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- PARSER SAR -----------------------------------

def parse_sar_log(log_content: str, label: str, base_dt: Optional[datetime] = None):
    """
    Parse log SAR dan kembalikan (cpu_df, mem_df) dengan kolom:
      Time (time-of-day), AbsTime (datetime absolut), Config (label).
    """
    lines = log_content.strip().split('\n')
    cpu_data, mem_data = [], []

    for line in lines:
        # Baris data diawali "HH:MM:SS AM/PM"
        if re.match(r'^\d{2}:\d{2}:\d{2} [AP]M', line):
            parts = line.split()
            timestamp = parts[0] + ' ' + parts[1]

            # CPU: baris "all"
            if len(parts) > 2 and parts[2] == 'all':
                if len(parts) >= 9:
                    cpu_data.append([
                        timestamp,
                        float(parts[3]),  # %user
                        float(parts[4]),  # %nice
                        float(parts[5]),  # %system
                        float(parts[6]),  # %iowait
                        float(parts[7]),  # %steal
                        float(parts[8])   # %idle
                    ])
            # Memori: baris dengan angka di kolom ke-3 (index 2)
            elif len(parts) > 2 and parts[2].isdigit():
                if len(parts) >= 13:
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

    cpu_cols = ['Time', '%user', '%nice', '%system', '%iowait', '%steal', '%idle']
    mem_cols = ['Time', 'kbmemfree', 'kbavail', 'kbmemused', '%memused', 'kbbuffers',
                'kbcached', 'kbcommit', '%commit', 'kbactive', 'kbinact', 'kbdirty']

    cpu_df = pd.DataFrame(cpu_data, columns=cpu_cols)
    mem_df = pd.DataFrame(mem_data, columns=mem_cols)

    # Label konfigurasi
    cpu_df['Config'] = label
    mem_df['Config'] = label

    # Konversi ke time-of-day
    cpu_df['Time'] = pd.to_datetime(cpu_df['Time'], format='%I:%M:%S %p', errors='coerce')
    mem_df['Time'] = pd.to_datetime(mem_df['Time'], format='%I:%M:%S %p', errors='coerce')
    cpu_df.dropna(subset=['Time'], inplace=True)
    mem_df.dropna(subset=['Time'], inplace=True)

    # Buat AbsTime (datetime absolut) dari base_dt + time-of-day; handle wrap day
    def attach_abs(df: pd.DataFrame, base: Optional[datetime]):
        if df.empty:
            df['AbsTime'] = pd.NaT
            return df
        if base is None:
            base = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        abs_times, last_dt = [], None
        for t in df['Time']:
            dt = datetime.combine(base.date(), t.time())
            if last_dt is not None and dt < last_dt:  # melewati tengah malam
                dt += timedelta(days=1)
            abs_times.append(dt)
            last_dt = dt
        df['AbsTime'] = pd.to_datetime(abs_times)
        return df

    cpu_df = attach_abs(cpu_df, base_dt)
    mem_df = attach_abs(mem_df, base_dt)
    return cpu_df, mem_df


# ------------------------- META DARI NAMA FILE ------------------------------

def _normalize_scenario(raw: str) -> str:
    s = raw.replace('shuwdown', 'shutdown')  # toleransi typo
    s = s.replace('-', ' ').replace('_', ' ').strip()
    s = re.sub(r'\s+', ' ', s)
    return s.title() if s else 'Unknown'

def extract_meta_from_filename(fname: str):
    """
    Kembalikan (label, scenario, protection) dari nama file.
    """
    base = Path(fname).name
    name = base.rsplit('.', 1)[0]
    parts = name.split('_')

    # Proteksi
    lower = name.lower()
    if 'selinux' in lower:
        protection = 'SELinux'
    elif 'apparmor' in lower:
        protection = 'AppArmor'
    elif 'no-protection' in lower or 'noprotection' in lower or 'no_protection' in lower:
        protection = 'No Protection'
    else:
        protection = 'No Protection'

    # Skenario setelah dua token timestamp
    stop = {'selinux','apparmor','no-protection','noprotection','no','protection','cpu','mem','cpu_mem','cpumem'}
    scenario_tokens = []
    if len(parts) > 2:
        for tok in parts[2:]:
            if tok.lower() in stop:
                break
            scenario_tokens.append(tok)
    scenario = _normalize_scenario(' '.join(scenario_tokens) if scenario_tokens else 'unknown')

    label = f"{scenario} — {protection}"
    return label, scenario, protection

def base_dt_from_filename(fname: str) -> Optional[datetime]:
    m = re.match(r'^(\d{4}-\d{2}-\d{2})_(\d{6})_', Path(fname).name)
    if not m:
        return None
    return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H%M%S")


# ---------------------- LOAD + INDEKS WAKTU GANDA ---------------------------

def load_logs_from_dir(input_dir: Path):
    paths = sorted(Path(input_dir).glob("*.log"))
    if not paths:
        raise FileNotFoundError(f"Tidak menemukan file .log di: {Path(input_dir).resolve()}")

    cpu_frames, mem_frames = [], []
    for p in paths:
        try:
            content = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            content = p.read_bytes().decode('utf-8', errors='ignore')

        label, scenario, protection = extract_meta_from_filename(p.name)
        base_dt = base_dt_from_filename(p.name)

        cpu_df, mem_df = parse_sar_log(content, label, base_dt)
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
        raise RuntimeError("Tidak ada data CPU terbaca. Pastikan log hasil `sar -u ALL` valid.")
    if not mem_frames:
        raise RuntimeError("Tidak ada data Memori terbaca. Pastikan log hasil `sar -r` valid.")

    cpu = pd.concat(cpu_frames, ignore_index=True)
    mem = pd.concat(mem_frames, ignore_index=True)

    # Global index (menit) dari waktu terkecil seluruh file
    t0 = min(cpu['AbsTime'].min(), mem['AbsTime'].min())
    cpu['Global_Index'] = (cpu['AbsTime'] - t0).dt.total_seconds() / 60.0
    mem['Global_Index'] = (mem['AbsTime'] - t0).dt.total_seconds() / 60.0

    # Relative index (menit) dari awal tiap file → tetap transparent (gap tidak dipampatkan)
    cpu['Rel_Index'] = cpu.groupby('Source_File')['AbsTime'].transform(lambda s: (s - s.min()).dt.total_seconds() / 60.0)
    mem['Rel_Index'] = mem.groupby('Source_File')['AbsTime'].transform(lambda s: (s - s.min()).dt.total_seconds() / 60.0)

    cpu.sort_values(['AbsTime','Config'], inplace=True, ignore_index=True)
    mem.sort_values(['AbsTime','Config'], inplace=True, ignore_index=True)
    return cpu, mem


# --------------------------- PLOTTING & GAPS --------------------------------

def try_set_style():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        pass

def median_gap_seconds(df: pd.DataFrame) -> float:
    diffs = df.sort_values('AbsTime')['AbsTime'].diff().dropna()
    return float(diffs.dt.total_seconds().median()) if not diffs.empty else 0.0

def break_on_gaps(df: pd.DataFrame, y_col: str, gap_mult: float = 3.0) -> pd.DataFrame:
    """
    Sisipkan NaN pada y_col jika gap waktu antar sampel > gap_mult × median interval.
    """
    d = df.sort_values('AbsTime').copy()
    med = median_gap_seconds(d)
    thr = med * gap_mult if med > 0 else float('inf')
    gaps = d['AbsTime'].diff().dt.total_seconds().fillna(0.0)
    d.loc[gaps > thr, y_col] = np.nan
    return d

def make_plots(cpu: pd.DataFrame, mem: pd.DataFrame, outpath: Path, title: str, time_mode: str):
    try_set_style()
    xcol = 'Rel_Index' if time_mode == 'relative' else 'Global_Index'
    subtitle = " (Relative time)" if time_mode == 'relative' else " (Global time)"

    plt.figure(figsize=(16, 12))

    # ---- Panel 1: CPU Usage (100 - %idle)
    ax1 = plt.subplot(2, 2, 1)
    for config in cpu['Config'].unique():
        d = cpu[cpu['Config'] == config].copy()
        d['CPU_Usage'] = 100.0 - d['%idle']
        d = break_on_gaps(d, 'CPU_Usage')
        ax1.plot(d[xcol], d['CPU_Usage'], label=config, linewidth=2, alpha=0.9)
    ax1.set_title('CPU Utilization Comparison' + subtitle, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (minutes)'); ax1.set_ylabel('CPU Usage (%)')
    ax1.legend(); ax1.grid(True, linestyle='--', linewidth=0.5)

    # ---- Panel 2: Memory %used
    ax2 = plt.subplot(2, 2, 2)
    for config in mem['Config'].unique():
        d = mem[mem['Config'] == config].copy()
        d = break_on_gaps(d, '%memused')
        ax2.plot(d[xcol], d['%memused'], label=config, linewidth=2, alpha=0.9)
    ax2.set_title('Memory Usage Comparison' + subtitle, fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (minutes)'); ax2.set_ylabel('Memory Used (%)')
    ax2.legend(); ax2.grid(True, linestyle='--', linewidth=0.5)

    # ---- Panel 3: Memory %commit
    ax3 = plt.subplot(2, 2, 3)
    for config in mem['Config'].unique():
        d = mem[mem['Config'] == config].copy()
        d = break_on_gaps(d, '%commit')
        ax3.plot(d[xcol], d['%commit'], label=config, linewidth=2, alpha=0.9)
    ax3.set_title('Memory Commit Comparison' + subtitle, fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time (minutes)'); ax3.set_ylabel('Commit Percentage (%)')
    ax3.legend(); ax3.grid(True, linestyle='--', linewidth=0.5)

    # ---- Panel 4: CPU %iowait
    ax4 = plt.subplot(2, 2, 4)
    for config in cpu['Config'].unique():
        d = cpu[cpu['Config'] == config].copy()
        d = break_on_gaps(d, '%iowait')
        ax4.plot(d[xcol], d['%iowait'], label=config, linewidth=2, alpha=0.9)
    ax4.set_title('I/O Wait Comparison' + subtitle, fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time (minutes)'); ax4.set_ylabel('I/O Wait (%)')
    ax4.legend(); ax4.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout(pad=3.0)
    plt.suptitle(title, fontsize=18, fontweight='bold')
    plt.subplots_adjust(top=0.92)

    outpath = Path(outpath); outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"[OK] Grafik tersimpan: {outpath}")


# ---------------------- PER-SKENARIO & RINGKASAN ----------------------------

def _safe(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_')

def make_plots_per_scenario(cpu: pd.DataFrame, mem: pd.DataFrame,
                            out_dir: Path, title: str, time_mode: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = sorted(set(cpu['Scenario']) | set(mem['Scenario']))
    for sc in scenarios:
        cpu_sc = cpu[cpu['Scenario'] == sc]
        mem_sc = mem[mem['Scenario'] == sc]
        if cpu_sc.empty and mem_sc.empty:
            continue
        outpath = out_dir / f"{_safe(sc)}_{time_mode}.png"
        make_plots(cpu_sc, mem_sc, outpath, f"{title} — {sc}", time_mode)

def print_summaries(cpu: pd.DataFrame, mem: pd.DataFrame):
    print("CPU Statistics Summary (per Config):")
    cpu_summary = cpu.groupby('Config').agg({
        '%user': ['mean', 'max'],
        '%system': ['mean', 'max'],
        '%iowait': ['mean', 'max'],
        '%idle': ['mean', 'min']
    }).round(2)
    print(cpu_summary)

    print("\nMemory Statistics Summary (per Config):")
    mem_summary = mem.groupby('Config').agg({
        '%memused': ['mean', 'max'],
        '%commit': ['mean', 'max'],
        'kbcached': 'mean'
    }).round(2)
    print(mem_summary)
    return cpu_summary, mem_summary


# --------------------------------- MAIN -------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Analisis log sar CPU/Mem dengan opsi per-skenario & mode waktu.")
    ap.add_argument("-i","--input", default="log_cpu_mem", help="Folder berisi *.log")
    ap.add_argument("-o","--output", default="selinux_apparmor_all.png",
                    help="File PNG gabungan (semua skenario)")
    ap.add_argument("--title", default="Performance Comparison: CPU & Memory", help="Judul grafik")
    ap.add_argument("--time-mode", choices=["relative","global"], default="relative",
                    help="relative=overlay per file, global=linimasa gabungan")
    ap.add_argument("--per-scenario", action="store_true",
                    help="Simpan satu PNG per skenario ke --out-dir")
    ap.add_argument("--out-dir", default="figures",
                    help="Folder output untuk --per-scenario")
    ap.add_argument("--no-combined", action="store_true",
                    help="Jangan buat grafik gabungan (hanya per-skenario)")
    args = ap.parse_args()

    try:
        cpu, mem = load_logs_from_dir(Path(args.input))

        if args.per_scenario:
            make_plots_per_scenario(cpu, mem, Path(args.out_dir), args.title, args.time_mode)

        if not args.no_combined:
            make_plots(cpu, mem, Path(args.output), args.title, args.time_mode)

        # Ringkasan di terminal (opsional untuk BAB IV)
        print_summaries(cpu, mem)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(2)
    except Exception as e:
        print(f"Terjadi kesalahan: {e}", file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()