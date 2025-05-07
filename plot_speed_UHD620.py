import matplotlib.pyplot as plt

# 資料長度（原始資料值，用於標示）
lengths = [100, 1000, 10000, 100000]

# 各版本的執行時間（單位：秒）
times_cpu = [0.0101459, 0.93595, 95.513, 10851.1]
times_webgpu_optimized = [0.136, 0.234, 1.524, 48.794]

# 計算 speedup（相對於 CPU 版本，即 CPU 時間 除以 其他版本時間）
speedup_cpu = [1 for _ in times_cpu]  # CPU 本身作為基準，speedup 為 1
speedup_webgpu_optimized = [cpu / opt for cpu, opt in zip(times_cpu, times_webgpu_optimized)]

# 以等距 x 軸位置（0, 1, 2, 3）代表各資料點
x = list(range(len(lengths)))

plt.figure(figsize=(10, 6))

# 繪製各版本的 Speedup 折線圖
# plt.plot(x, speedup_cuda, marker='o', linestyle='-', label='CUDA')
plt.plot(x, speedup_cpu, marker='s', linestyle='-', label='CPU')
plt.plot(x, speedup_webgpu_optimized, marker='d', linestyle='-', label='WebGPU Optimized Version')

# 設定 x 軸刻度，標示原始資料長度
plt.xticks(x, [str(l) for l in lengths])
plt.xlabel('Sequence length')
plt.ylabel('Speedup (Relative to CPU)')
plt.title('Speedup Relative to the CPU Version')
plt.legend()
plt.grid(True, linestyle='--', color='gray', alpha=0.5)
plt.tight_layout()

# 定義一個小函式，在每個資料點上方標示加速倍率數值
def annotate_points(x_vals, y_vals):
    for i, s in enumerate(y_vals):
        # 設定一個 offset，用來讓文字標示稍微高於資料點，
        # 若 s 較小則至少 offset 為 0.05
        offset = max(0.05, s * 0.05)
        plt.text(x_vals[i], s + offset, f"{s:.2f}", ha='center', va='bottom', fontsize=9)

# 分別為各條折線圖的資料點標示加速倍率
annotate_points(x, speedup_cpu)
annotate_points(x, speedup_webgpu_optimized)

# 顯示圖形
plt.show()
