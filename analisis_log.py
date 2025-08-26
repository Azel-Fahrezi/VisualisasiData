import pandas as pd
import matplotlib.pyplot as plt

def parse_sar_log(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith(('12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23')):  # Jam
                parts = line.split()
                if len(parts) >= 10:
                    timestamp = parts[0]
                    cpu_idle = float(parts[7])
                    mem_used = float(parts[9])  # %memused
                    data.append([timestamp, cpu_idle, mem_used])
    return pd.DataFrame(data, columns=['Time', 'CPU_Idle', 'Mem_Used'])

# Load data
df_no_selinux = parse_sar_log('no_selinux_cpu_mem.log')
df_with_selinux = parse_sar_log('with_selinux_cpu_mem.log')

# Plotting
plt.figure(figsize=(15, 10))

# CPU Usage
plt.subplot(2, 1, 1)
plt.plot(df_no_selinux.index, 100 - df_no_selinux['CPU_Idle'], 'b-', label='Tanpa SELinux')
plt.plot(df_with_selinux.index, 100 - df_with_selinux['CPU_Idle'], 'r-', label='Dengan SELinux')
plt.ylabel('CPU Usage (%)')
plt.title('Perbandingan Penggunaan CPU')
plt.legend()
plt.grid(True)

# Memory Usage
plt.subplot(2, 1, 2)
plt.plot(df_no_selinux.index, df_no_selinux['Mem_Used'], 'b-', label='Tanpa SELinux')
plt.plot(df_with_selinux.index, df_with_selinux['Mem_Used'], 'r-', label='Dengan SELinux')
plt.ylabel('Memory Used (%)')
plt.xlabel('Sample Index (1 detik per sampel)')
plt.title('Perbandingan Penggunaan Memori')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('selinux_performance_comparison.png')
plt.show()