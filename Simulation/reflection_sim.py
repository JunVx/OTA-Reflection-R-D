import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Global plot style
# =========================================================
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['lines.linewidth'] = 2.8
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.linewidth'] = 1.0

# =========================================================
# Basic settings
# =========================================================
f_start = 2e9
f_stop = 3e9
f = np.linspace(f_start, f_stop, 20001)

# Figure 1 reflection levels: -5 dB to -40 dB, step 5 dB
refl_db_fig1 = np.arange(-5, -41, -5)
amp_fig1 = 10 ** (refl_db_fig1 / 20.0)

# Figures 2~4 reflection levels: -5 dB to -40 dB, step 1 dB
refl_db_figx = np.arange(-5, -41, -1)
amp_figx = 10 ** (refl_db_figx / 20.0)

# Bandwidth list
# CW is handled separately, not by averaging
bw_list = ["CW", 1e6, 5e6, 10e6, 20e6, 50e6, 100e6]
bw_labels = ["CW", "1M", "5M", "10M", "20M", "50M", "100M"]

# Limits
limit_02 = 0.2
limit_05 = 0.5
limit_10 = 1.0

# Fixed nominal delays for the last 3 figures
tau_plot_list = [10e-9, 20e-9, 30e-9]

# Delay fluctuation settings
tau_fluct_ns = 1.0
tau_step_ns = 0.1

# =========================================================
# Functions
# =========================================================
def channel_response(freq, a, tau):
    return 1 + a * np.exp(-1j * 2 * np.pi * freq * tau)

def channel_mag_db(freq, a, tau):
    h = channel_response(freq, a, tau)
    return 20 * np.log10(np.abs(h))

def cw_worst_case_uncertainty(a):
    return max(20 * np.log10(1 + a), -20 * np.log10(1 - a))

def wideband_worst_case_uncertainty_fixed_tau(a, bw, tau):
    s = abs(np.sinc(bw * tau))
    p_max = 1 + a**2 + 2 * a * s
    p_min = 1 + a**2 - 2 * a * s
    return max(10 * np.log10(p_max), -10 * np.log10(p_min))

def worst_case_with_tau_fluctuation(a, bw, tau0, tau_fluct_ns=1.0, tau_step_ns=0.1):
    if bw == "CW":
        return cw_worst_case_uncertainty(a)

    tau_scan = np.arange(
        tau0 - tau_fluct_ns * 1e-9,
        tau0 + tau_fluct_ns * 1e-9 + 0.5 * tau_step_ns * 1e-9,
        tau_step_ns * 1e-9
    )

    vals = [wideband_worst_case_uncertainty_fixed_tau(a, bw, tau) for tau in tau_scan]
    return np.max(vals)

def beautify_axes():
    plt.grid(True, which='major', linestyle='--', alpha=0.70)
    ax = plt.gca()
    ax.tick_params(width=1.4, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

# =========================================================
# Figure 1
# Keep unchanged
# =========================================================
tau_example = 10e-9

plt.figure(figsize=(11, 6.8))
for refl_db, a in zip(refl_db_fig1, amp_fig1):
    plt.plot(f / 1e9, channel_mag_db(f, a, tau_example), linestyle='-', label=f"{refl_db} dB")

plt.xlabel("Frequency (GHz)")
plt.ylabel("Channel Magnitude (dB)")
plt.title(f"Channel Variation under Different Reflection Levels\n(Delay = {tau_example * 1e9:.2f} ns)")
beautify_axes()
plt.legend(ncol=2, frameon=True)
plt.tight_layout()
plt.show()

# =========================================================
# Figures 2~4
# All bandwidth curves use solid lines
# =========================================================
for tau_fixed in tau_plot_list:
    unc_mat = np.zeros((len(bw_list), len(amp_figx)))

    for i, bw in enumerate(bw_list):
        for j, a in enumerate(amp_figx):
            unc_mat[i, j] = worst_case_with_tau_fluctuation(
                a=a,
                bw=bw,
                tau0=tau_fixed,
                tau_fluct_ns=tau_fluct_ns,
                tau_step_ns=tau_step_ns
            )

    plt.figure(figsize=(11.5, 7.0))
    plot_order = list(range(1, len(bw_list))) + [0]
    for i in plot_order:
        plt.plot(
            refl_db_figx,
            unc_mat[i],
            linestyle='-',
            linewidth=3.6 if bw_list[i] == "CW" else 2.8,
            color='black' if bw_list[i] == "CW" else None,
            zorder=5 if bw_list[i] == "CW" else 3,
            label=bw_labels[i]
        )

    plt.axhline(limit_02, color='green', linestyle='--', linewidth=3.0, label='±0.2 dB limit')
    plt.axhline(limit_05, color='gold', linestyle='--', linewidth=3.0, label='±0.5 dB limit')
    plt.axhline(limit_10, color='red', linestyle='--', linewidth=3.0, label='±1.0 dB limit')

    plt.xlabel("Reflection Level (dB)")
    plt.ylabel("Maximum Uncertainty (dB)")
    plt.title(
        f"Maximum Uncertainty vs Reflection Level\n"
        f"Nominal Delay = {tau_fixed * 1e9:.0f} ns, "
    )
    beautify_axes()
    plt.legend(ncol=2, frameon=True)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 90)
    print(
        f"Nominal delay = {tau_fixed * 1e9:.0f} ns, "
        f"delay scan = ±{tau_fluct_ns:.1f} ns, step = {tau_step_ns:.1f} ns"
    )
    print("=" * 90)
    for i, bw in enumerate(bw_list):
        print(f"\nBandwidth = {bw_labels[i]}")
        for refl_db, val in zip(refl_db_figx, unc_mat[i]):
            print(f"  Reflection {refl_db:>4} dB : {val:.4f} dB")
