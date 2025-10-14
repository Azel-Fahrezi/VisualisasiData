import argparse
from pathlib import Path
import re
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ========================== WARNA & URUTAN TETAP ============================

# Urutan dan warna tetap per proteksi (Tableau10 default)
PROTECTION_ORDER = ["SELinux", "AppArmor", "No Protection"]
COLOR_MAP = {
    "SELinux": "#1f77b4",       # biru
    "AppArmor": "#ff7f0e",      # oranye
    "No Protection": "#2ca02c", # hijau
}


# ============================== PARSER SAR ==================================

def parse_sar_log(log_content: str, label: str, base_dt: Optional[datetime] = None):
    """Parse log SAR dan kembalikan (cpu_df, mem_df) dengan kolom Time & AbsTime."""
    lines = log_content.strip().split('\n')
    cpu_data, mem_data = [], []

    for line in lines:
        if re.match(r'^\d{2}:\d{2}:\d{2} [AP]M', line):
            parts = line.split()
            timestamp = parts[0] + ' ' + parts[1]

            if len(parts) > 2 and parts[2] == 'all':  # CPU
                if len(parts) >= 9:
                    cpu_data.append([
                        timestamp, float(parts[3]), float(parts[4]),
                        float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
                    ])
            elif len(parts) > 2 and parts[2].isdigit():  # Memory
                if len(parts) >= 13:
                    mem_data.append([
                        timestamp, int(parts[2]), int(parts[3]), int(parts[4]), float(parts[5]),
                        int(parts[6]), int(parts[7]), int(parts[8]), float(parts[9]),
                        int(parts[10]), int(parts[11]), int(parts[12])
                    ])

    cpu_cols = ['Time', '%user', '%nice', '%system', '%iowait', '%steal', '%idle']
    mem_cols = ['Time', 'kbmemfree', 'kbavail', 'kbmemused', '%memused', 'kbbuffers',
                'kbcached', 'kbcommit', '%commit', 'kbactive', 'kbinact', 'kbdirty']

    cpu_df = pd.DataFrame(cpu_data, columns=cpu_cols)
    mem_df = pd.DataFrame(mem_data, columns=mem_cols)
    cpu_df['Config'] = label; mem_df['Config'] = label

    cpu_df['Time'] = pd.to_datetime(cpu_df['Time'], format='%I:%M:%S %p', errors='coerce')
    mem_df['Time'] = pd.to_datetime(mem_df['Time'], format='%I:%M:%S %p', errors='coerce')
    cpu_df.dropna(subset=['Time'], inplace=True); mem_df.dropna(subset=['Time'], inplace=True)

    def attach_abs(df: pd.DataFrame, base: Optional[datetime]):
        if df.empty:
            df['AbsTime'] = pd.NaT; return df
        if base is None:
            base = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        abs_times, last_dt = [], None
        for t in df['Time']:
            dt = datetime.combine(base.date(), t.time())
            if last_dt is not None and dt < last_dt:  # wrap day
                dt += timedelta(days=1)
            abs_times.append(dt); last_dt = dt
        df['AbsTime'] = pd.to_datetime(abs_times); return df

    cpu_df = attach_abs(cpu_df, base_dt)
    mem_df = attach_abs(mem_df, base_dt)
    return cpu_df, mem_df


# ========================= META DARI NAMA FILE ==============================

def _normalize_scenario(raw: str) -> str:
    s = s.replace('-', ' ').replace('_', ' ').strip()
    s = re.sub(r'\s+', ' ', s)
    return s.title() if s else 'Unknown'

def extract_meta_from_filename(fname: str):
    base = Path(fname).name
    name = base.rsplit('.', 1)[0]
    parts = name.split('_')

    lower = name.lower()
    if 'selinux' in lower:
        protection = 'SELinux'
    elif 'apparmor' in lower:
        protection = 'AppArmor'
    elif 'no-protection' in lower or 'noprotection' in lower or 'no_protection' in lower:
        protection = 'No Protection'
    else:
        protection = 'No Protection'

    stop = {'selinux','apparmor','no-protection','noprotection','no',
            'protection','cpu','mem','cpu_mem','cpumem'}
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
    if not m: return None
    return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H%M%S")


# =================== LOAD + INDEKS WAKTU (GLOBAL & REL) =====================

def load_logs_from_dir(input_dir: Path):
    paths = sorted(Path(input_dir).glob("*.log"))
    if not paths:
        raise FileNotFoundError(f"Tidak ada .log di: {Path(input_dir).resolve()}")

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
            cpu_df['Scenario'] = scenario; cpu_df['Protection'] = protection; cpu_df['Source_File'] = p.name
            cpu_frames.append(cpu_df)
        if not mem_df.empty:
            mem_df['Scenario'] = scenario; mem_df['Protection'] = protection; mem_df['Source_File'] = p.name
            mem_frames.append(mem_df)

    if not cpu_frames: raise RuntimeError("Tidak ada data CPU terbaca.")
    if not mem_frames: raise RuntimeError("Tidak ada data Memori terbaca.")

    cpu = pd.concat(cpu_frames, ignore_index=True)
    mem = pd.concat(mem_frames, ignore_index=True)

    t0 = min(cpu['AbsTime'].min(), mem['AbsTime'].min())
    cpu['Global_Index'] = (cpu['AbsTime'] - t0).dt.total_seconds() / 60.0
    mem['Global_Index'] = (mem['AbsTime'] - t0).dt.total_seconds() / 60.0
    cpu['Rel_Index'] = cpu.groupby('Source_File')['AbsTime'].transform(lambda s: (s - s.min()).dt.total_seconds() / 60.0)
    mem['Rel_Index'] = mem.groupby('Source_File')['AbsTime'].transform(lambda s: (s - s.min()).dt.total_seconds() / 60.0)

    cpu.sort_values(['AbsTime','Protection'], inplace=True, ignore_index=True)
    mem.sort_values(['AbsTime','Protection'], inplace=True, ignore_index=True)
    return cpu, mem


# ======================== PLOTTING & PENANGANAN GAP =========================

def try_set_style():
    try: plt.style.use('seaborn-v0_8-whitegrid')
    except Exception: pass

def median_gap_seconds(df: pd.DataFrame) -> float:
    diffs = df.sort_values('AbsTime')['AbsTime'].diff().dropna()
    return float(diffs.dt.total_seconds().median()) if not diffs.empty else 0.0

def break_on_gaps(df: pd.DataFrame, y_col: str, gap_mult: float = 3.0) -> pd.DataFrame:
    d = df.sort_values('AbsTime').copy()
    med = median_gap_seconds(d)
    thr = med * gap_mult if med > 0 else float('inf')
    gaps = d['AbsTime'].diff().dt.total_seconds().fillna(0.0)
    d.loc[gaps > thr, y_col] = np.nan
    return d


# ======================= GENERIC SINGLE-METRIC CHART ========================

def _plot_single_metric_by_protection(data: pd.DataFrame, metric_col: str, y_label: str,
                                      outpath: Path, title: str, time_mode: str,
                                      mean_name: str = 'μ'):
    """
    Plot satu metrik (overlay berdasarkan Protection) dengan warna terkunci.
    """
    try_set_style()
    xcol = 'Rel_Index' if time_mode == 'relative' else 'Global_Index'
    subtitle = " (Relative time)" if time_mode == 'relative' else " (Global time)"

    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    # Mean per Protection untuk legenda stabil
    means = data.groupby('Protection')[metric_col].mean().round(3)

    for prot in PROTECTION_ORDER:
        d = data[data['Protection'] == prot]
        if d.empty:
            continue
        d = break_on_gaps(d, metric_col)
        mu = means.loc[prot] if prot in means.index else np.nan
        ax.plot(
            d[xcol], d[metric_col],
            linewidth=2, alpha=0.95,
            color=COLOR_MAP.get(prot),
            label=f"{prot} — {mean_name}={mu}"
        )

    ax.set_title(f"{title}{subtitle}", fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (minutes)'); ax.set_ylabel(y_label)
    ax.legend(); ax.grid(True, linestyle='--', linewidth=0.5)
    outpath = Path(outpath); outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {outpath}")


# ================== PER-SCENARIO: 4 GRAFIK (CPU/MEM TERPISAH) ===============

def _safe(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_')

def make_four_charts_per_scenario(cpu: pd.DataFrame, mem: pd.DataFrame,
                                  out_dir: Path, time_mode: str):
    """
    Untuk setiap skenario -> 4 file PNG:
      <sc>/CPUUsage_<mode>.png, <sc>/IOWait_<mode>.png,
      <sc>/MemUsed_<mode>.png, <sc>/Commit_<mode>.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = sorted(set(cpu['Scenario']) | set(mem['Scenario']))
    for sc in scenarios:
        cpu_sc = cpu[cpu['Scenario'] == sc].copy()
        mem_sc = mem[mem['Scenario'] == sc].copy()
        if cpu_sc.empty and mem_sc.empty: continue

        sub = out_dir / _safe(sc)
        sub.mkdir(parents=True, exist_ok=True)

        # 1) CPU Usage
        cpu_sc['CPU_Usage'] = 100.0 - cpu_sc['%idle']
        _plot_single_metric_by_protection(
            data=cpu_sc, metric_col='CPU_Usage', y_label='CPU Usage (%)',
            outpath=sub / f"{_safe(sc)}_CPUUsage_{time_mode}.png",
            title=f"CPU Usage — {sc}", time_mode=time_mode
        )

        # 2) I/O Wait
        _plot_single_metric_by_protection(
            data=cpu_sc, metric_col='%iowait', y_label='I/O Wait (%)',
            outpath=sub / f"{_safe(sc)}_IOWait_{time_mode}.png",
            title=f"I/O Wait — {sc}", time_mode=time_mode
        )

        # 3) MemUsed
        _plot_single_metric_by_protection(
            data=mem_sc, metric_col='%memused', y_label='Memory Used (%)',
            outpath=sub / f"{_safe(sc)}_MemUsed_{time_mode}.png",
            title=f"Memory Usage — {sc}", time_mode=time_mode
        )

        # 4) Commit
        _plot_single_metric_by_protection(
            data=mem_sc, metric_col='%commit', y_label='Commit Percentage (%)',
            outpath=sub / f"{_safe(sc)}_Commit_{time_mode}.png",
            title=f"Memory Commit — {sc}", time_mode=time_mode
        )


# =========================== RINGKASAN & CSV ================================

def compute_summary(cpu: pd.DataFrame, mem: pd.DataFrame) -> pd.DataFrame:
    """Rata-rata per (Scenario, Protection) untuk 4 metrik utama."""
    cpu_tmp = cpu.copy(); cpu_tmp['CPU_Usage'] = 100.0 - cpu_tmp['%idle']
    g1 = cpu_tmp.groupby(['Scenario','Protection']).agg(
        CPU_Usage_mean=('CPU_Usage','mean'),
        IOwait_mean=('%iowait','mean')
    )
    g2 = mem.groupby(['Scenario','Protection']).agg(
        MemUsed_mean=('%memused','mean'),
        Commit_mean=('%commit','mean')
    )
    return g1.join(g2, how='inner').round(3).reset_index()

def save_summary_csv(df: pd.DataFrame, out_csv: Path):
    out_csv = Path(out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Summary CSV saved: {out_csv}")


# ================================= MAIN =====================================

def main():
    ap = argparse.ArgumentParser(description="4 grafik per skenario (CPU Usage, I/O Wait, MemUsed, Commit) dengan warna proteksi konsisten.")
    ap.add_argument("-i","--input", default="log_cpu_mem", help="Folder *.log")
    ap.add_argument("--time-mode", choices=["relative","global"], default="relative",
                    help="relative=overlay per file, global=linimasa gabungan")
    ap.add_argument("--out-dir", default="figures",
                    help="Folder output per skenario")
    ap.add_argument("--summary-csv", default="metrics_summary.csv",
                    help="File CSV ringkasan mean per skenario & proteksi")
    args = ap.parse_args()

    try:
        cpu, mem = load_logs_from_dir(Path(args.input))
        make_four_charts_per_scenario(cpu, mem, Path(args.out_dir), args.time_mode)

        summary = compute_summary(cpu, mem)
        print("\n=== Mean Metrics per Scenario & Protection ===")
        print(summary.to_string(index=False))
        save_summary_csv(summary, Path(args.summary_csv))

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(2)
    except Exception as e:
        print(f"Terjadi kesalahan: {e}", file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()