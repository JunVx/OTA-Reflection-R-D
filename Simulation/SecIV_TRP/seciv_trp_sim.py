from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import welch

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.8,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


EPS = 1e-18
C_LIGHT = 299_792_458.0
FS_HZ = 122.88e6
CARRIER_HZ = 1.0e9
SNR_DB = 40.0
PSD_FLOOR_DB = -40.0
LOS_PATHLOSS_DB = 0.0
TARGET_BW_MHZ = (10, 20, 50, 100)
FIGURE_BW_MHZ = (20, 100)
REPRESENTATIVE_REFLECTION_DB = -10.0
REPRESENTATIVE_DELAY_NS = 10.0
REFLECTION_LEVELS_DB = (-5.0, -10.0, -20.0, -30.0)
REFLECTION_DELAYS_NS = (6.0, 10.0, 20.0)
WELCH_SEGMENT = 32768
WELCH_OVERLAP = WELCH_SEGMENT // 2
MAX_FIT_POINTS = 2048
PHASE_JITTER_STD_DEG = 12.0
DEFAULT_SHOW_FIGURES = os.environ.get("SISO_TRP_SHOW", "1") != "0"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "datafiles"
RESULTS_DIR = ROOT_DIR / "results" / "SecIV_TRP"


@dataclass(frozen=True)
class SpectrumDataset:
    bandwidth_mhz: int
    filename: str
    freqs_hz: np.ndarray
    tx_psd: np.ndarray
    inband_mask: np.ndarray
    noise_mask: np.ndarray


@dataclass(frozen=True)
class ScenarioConfig:
    bandwidth_mhz: int
    reflection_db: float
    nominal_delay_ns: float


@dataclass(frozen=True)
class ScenarioResult:
    bandwidth_mhz: int
    reflection_db_true: float
    delay_ns_true: float
    reflection_db_est: float
    delay_ns_est: float
    reflection_db_error: float
    delay_ns_error: float
    measured_power_db: float
    restored_power_db: float
    ideal_los_power_db: float
    measured_error_db: float
    restored_error_db: float
    fit_rmse_db: float


def finalize_figure(fig: plt.Figure, output_path: Path, show_figures: bool) -> None:
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    if not show_figures:
        plt.close(fig)


def limit_plot_region(ax: plt.Axes, bandwidth_mhz: float, margin_mhz: float = 10.0) -> None:
    limit_mhz = min(FS_HZ / 2.0 / 1e6, bandwidth_mhz / 2.0 + margin_mhz)
    ax.set_xlim(-limit_mhz, limit_mhz)


def as_axis_array(axes: plt.Axes | np.ndarray) -> np.ndarray:
    return np.atleast_1d(axes)


def wrap_phase_rad(values: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def db10(values: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(values, EPS))


def db20(values: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(values, EPS))


def linear_from_db(value_db: float) -> float:
    return 10.0 ** (value_db / 10.0)


def amplitude_from_db(value_db: float) -> float:
    return 10.0 ** (value_db / 20.0)


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def clean_previous_figures(output_dir: Path) -> None:
    for pattern in ("*.png", "*.pdf"):
        for figure_path in output_dir.glob(pattern):
            figure_path.unlink()


def carrier_wavelength_m(carrier_hz: float = CARRIER_HZ) -> float:
    return C_LIGHT / carrier_hz


def measurement_position_offsets_m(carrier_hz: float = CARRIER_HZ) -> np.ndarray:
    wavelength_m = carrier_wavelength_m(carrier_hz)
    return wavelength_m * np.array([-0.43, -0.21, 0.0, 0.16, 0.34], dtype=float)


def measurement_delay_offsets_ns(carrier_hz: float = CARRIER_HZ) -> np.ndarray:
    # Assume the displacement is aligned with the dominant reflection axis,
    # so the reflected path difference changes by approximately 2 * delta_x.
    return 2.0 * measurement_position_offsets_m(carrier_hz) / C_LIGHT * 1e9


def nominal_phase_offsets_rad(carrier_hz: float = CARRIER_HZ) -> np.ndarray:
    return wrap_phase_rad(2.0 * np.pi * carrier_hz * measurement_delay_offsets_ns(carrier_hz) * 1e-9)


def scenario_seed(bandwidth_mhz: int, reflection_db: float, nominal_delay_ns: float) -> int:
    return int(10_000 * bandwidth_mhz + 100 * round(abs(reflection_db)) + round(nominal_delay_ns * 10))


def generate_measurement_phases_rad(
    bandwidth_mhz: int,
    reflection_db: float,
    nominal_delay_ns: float,
    carrier_hz: float = CARRIER_HZ,
) -> np.ndarray:
    rng = np.random.default_rng(scenario_seed(bandwidth_mhz, reflection_db, nominal_delay_ns))
    base_phase = rng.uniform(-np.pi, np.pi)
    phase_jitter = rng.normal(0.0, np.deg2rad(PHASE_JITTER_STD_DEG), size=measurement_position_offsets_m(carrier_hz).size)
    return wrap_phase_rad(base_phase + nominal_phase_offsets_rad(carrier_hz) + phase_jitter)


def parse_bandwidth_mhz(path: Path) -> int:
    match = re.search(r"BW(\d+)M", path.stem, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse bandwidth from file name: {path.name}")
    return int(match.group(1))


def discover_datafiles(data_dir: Path) -> dict[int, Path]:
    available: dict[int, Path] = {}
    for file_path in sorted(data_dir.glob("*.txt")):
        bw_mhz = parse_bandwidth_mhz(file_path)
        available[bw_mhz] = file_path
    if not available:
        raise FileNotFoundError(f"No IQ files found in {data_dir}.")
    return available


def select_datafile_for_bandwidth(available_files: dict[int, Path], target_bw_mhz: int) -> tuple[Path, int]:
    if target_bw_mhz in available_files:
        return available_files[target_bw_mhz], target_bw_mhz

    wider_bandwidths = sorted(bw for bw in available_files if bw > target_bw_mhz)
    if wider_bandwidths:
        source_bw_mhz = wider_bandwidths[0]
        return available_files[source_bw_mhz], source_bw_mhz

    raise FileNotFoundError(
        f"No IQ file is available for {target_bw_mhz} MHz or a wider bandwidth."
    )


def load_hex_iq(file_path: Path) -> np.ndarray:
    words = np.fromiter(
        (int(line.strip(), 16) for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()),
        dtype=np.uint32,
    )
    i_data = ((words >> 16) & 0xFFFF).astype(np.uint16).view(np.int16).astype(np.float64)
    q_data = (words & 0xFFFF).astype(np.uint16).view(np.int16).astype(np.float64)
    iq = i_data + 1j * q_data
    iq /= np.sqrt(np.mean(np.abs(iq) ** 2))
    return iq


def compute_tx_psd(iq: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    freqs_hz, psd = welch(
        iq,
        fs=fs_hz,
        window="hann",
        nperseg=WELCH_SEGMENT,
        noverlap=WELCH_OVERLAP,
        detrend=False,
        return_onesided=False,
        scaling="density",
    )
    return np.fft.fftshift(freqs_hz), np.fft.fftshift(psd)


def build_noise_mask(freqs_hz: np.ndarray, bw_hz: float, fs_hz: float) -> np.ndarray:
    guard_hz = min(5e6, max(1e6, 0.08 * bw_hz))
    edge_start_hz = min(fs_hz / 2.0 - 1e6, bw_hz / 2.0 + guard_hz)
    noise_mask = np.abs(freqs_hz) >= edge_start_hz
    if np.count_nonzero(noise_mask) < 64:
        percentile = np.quantile(np.abs(freqs_hz), 0.85)
        noise_mask = np.abs(freqs_hz) >= percentile
    return noise_mask


def prepare_datasets() -> list[SpectrumDataset]:
    datasets: list[SpectrumDataset] = []
    available_files = discover_datafiles(DATA_DIR)
    for bandwidth_mhz in TARGET_BW_MHZ:
        file_path, source_bw_mhz = select_datafile_for_bandwidth(available_files, bandwidth_mhz)
        iq = load_hex_iq(file_path)
        freqs_hz, tx_psd = compute_tx_psd(iq, FS_HZ)
        inband_mask = np.abs(freqs_hz) <= bandwidth_mhz * 1e6 / 2.0
        noise_mask = build_noise_mask(freqs_hz, bandwidth_mhz * 1e6, FS_HZ)
        filename = file_path.name
        if source_bw_mhz != bandwidth_mhz:
            filename = f"{file_path.name} (central {bandwidth_mhz} MHz)"
        datasets.append(
            SpectrumDataset(
                bandwidth_mhz=bandwidth_mhz,
                filename=filename,
                freqs_hz=freqs_hz,
                tx_psd=tx_psd,
                inband_mask=inband_mask,
                noise_mask=noise_mask,
            )
        )
    return datasets


def power_integral(psd: np.ndarray, freqs_hz: np.ndarray, mask: np.ndarray) -> float:
    return float(np.trapezoid(psd[mask], freqs_hz[mask]))


def normalize_psd_to_tx_peak(psd: np.ndarray, tx_psd: np.ndarray) -> np.ndarray:
    return np.maximum(db10(psd / np.max(tx_psd)), PSD_FLOOR_DB)


def smoothed_normalized_psd_db(psd: np.ndarray, tx_psd: np.ndarray, window: int = 81) -> np.ndarray:
    return moving_average(normalize_psd_to_tx_peak(psd, tx_psd), window)


def psd_floor_from_tx_peak(tx_psd: np.ndarray) -> float:
    return float(np.max(tx_psd) * linear_from_db(PSD_FLOOR_DB))


def composite_power_response(freqs_hz: np.ndarray, reflection_amp: float, delay_ns: float, point_phase_rad: float) -> np.ndarray:
    phase = 2.0 * np.pi * freqs_hz * delay_ns * 1e-9 + point_phase_rad
    return 1.0 + reflection_amp**2 + 2.0 * reflection_amp * np.cos(phase)


def simulate_received_psd(
    dataset: SpectrumDataset,
    reflection_db: float,
    delay_ns: float,
    point_phase_rad: float,
    snr_db: float = SNR_DB,
    los_pathloss_db: float = LOS_PATHLOSS_DB,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    los_power_gain = linear_from_db(-los_pathloss_db)
    los_psd = dataset.tx_psd * los_power_gain
    reflection_amp = amplitude_from_db(reflection_db)
    channel_power = composite_power_response(dataset.freqs_hz, reflection_amp, delay_ns, point_phase_rad)
    reflected_psd = los_psd * channel_power

    los_inband_power = power_integral(los_psd, dataset.freqs_hz, dataset.inband_mask)
    total_span_hz = dataset.freqs_hz.max() - dataset.freqs_hz.min()
    noise_power = los_inband_power / linear_from_db(snr_db)
    noise_density = max(noise_power / total_span_hz, psd_floor_from_tx_peak(dataset.tx_psd))
    noise_psd = np.full_like(los_psd, noise_density)
    rx_psd = reflected_psd + noise_psd
    return rx_psd, los_psd, noise_psd, channel_power


def estimate_noise_floor(psd: np.ndarray, noise_mask: np.ndarray) -> float:
    return float(np.median(psd[noise_mask]))


def select_fit_indices(
    corrected_reference: np.ndarray,
    inband_mask: np.ndarray,
    max_points: int = MAX_FIT_POINTS,
) -> np.ndarray:
    indices = np.flatnonzero(inband_mask & (corrected_reference > 0.0))
    if indices.size <= max_points:
        return indices
    decimation = int(math.ceil(indices.size / max_points))
    return indices[::decimation]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def estimate_delay_candidates_ns(
    corrected_measurements: list[np.ndarray],
    fit_freqs_hz: np.ndarray,
    num_candidates: int = 4,
) -> np.ndarray:
    if fit_freqs_hz.size < 32:
        return np.array([6.0, 10.0, 20.0], dtype=float)

    delay_candidates: list[float] = []
    freq_step_hz = float(np.median(np.diff(fit_freqs_hz)))
    positive_delay_axis_s = np.fft.rfftfreq(fit_freqs_hz.size, d=freq_step_hz)
    valid_mask = (positive_delay_axis_s >= 1.0e-9) & (positive_delay_axis_s <= 35.0e-9)
    if not np.any(valid_mask):
        return np.array([6.0, 10.0, 20.0], dtype=float)

    for spectrum in corrected_measurements:
        spectrum_db = db10(spectrum)
        detrended = spectrum_db - moving_average(spectrum_db, max(51, spectrum_db.size // 24))
        ripple_fft = np.abs(np.fft.rfft(detrended * np.hanning(detrended.size)))
        valid_fft = ripple_fft[valid_mask]
        valid_delays_ns = positive_delay_axis_s[valid_mask] * 1e9
        if valid_fft.size == 0:
            continue
        peak_order = np.argsort(valid_fft)[-num_candidates:]
        for peak_idx in peak_order:
            delay_candidates.append(float(valid_delays_ns[peak_idx]))

    if not delay_candidates:
        return np.array([6.0, 10.0, 20.0], dtype=float)

    candidate_array = np.unique(np.round(delay_candidates, 1))
    defaults = np.array([6.0, 10.0, 20.0], dtype=float)
    merged = np.unique(np.concatenate((candidate_array, defaults)))
    merged.sort()
    if merged.size > num_candidates:
        center = np.median(merged)
        keep_order = np.argsort(np.abs(merged - center))
        merged = np.unique(np.concatenate((merged[keep_order[:num_candidates]], defaults)))
        merged.sort()
    return merged


def build_ratio_model(
    freqs_hz: np.ndarray,
    reflection_amp: float,
    delay_ns: float,
    phase_offsets_rad: np.ndarray,
    reference_index: int,
) -> list[np.ndarray]:
    ref_response = composite_power_response(freqs_hz, reflection_amp, delay_ns, phase_offsets_rad[reference_index])
    ratios: list[np.ndarray] = []
    for point_phase_rad in phase_offsets_rad:
        current_response = composite_power_response(freqs_hz, reflection_amp, delay_ns, point_phase_rad)
        ratios.append(current_response / np.maximum(ref_response, EPS))
    return ratios


def fit_reflection_parameters(
    measurement_psds: list[np.ndarray],
    dataset: SpectrumDataset,
    reference_index: int,
) -> tuple[float, float, np.ndarray, float]:
    num_points = len(measurement_psds)
    noise_floors = np.asarray([estimate_noise_floor(psd, dataset.noise_mask) for psd in measurement_psds])
    corrected_measurements = [np.maximum(psd - floor, EPS) for psd, floor in zip(measurement_psds, noise_floors)]
    fit_indices = select_fit_indices(corrected_measurements[reference_index], dataset.inband_mask)
    fit_freqs = dataset.freqs_hz[fit_indices]
    reference_psd = corrected_measurements[reference_index][fit_indices]

    observed_ratios = []
    for idx, current_psd in enumerate(corrected_measurements):
        if idx == reference_index:
            continue
        observed_ratios.append(current_psd[fit_indices] / np.maximum(reference_psd, EPS))

    def residual_vector(params: np.ndarray) -> np.ndarray:
        reflection_amp = params[0]
        delay_ns = params[1]
        point_phases_rad = wrap_phase_rad(params[2:])
        modeled_ratios = build_ratio_model(
            fit_freqs,
            reflection_amp,
            delay_ns,
            point_phases_rad,
            reference_index,
        )
        residual_blocks = []
        block_idx = 0
        for idx in range(len(modeled_ratios)):
            if idx == reference_index:
                continue
            modeled_db = db10(modeled_ratios[idx])
            observed_db = db10(observed_ratios[block_idx])
            residual_blocks.append(observed_db - modeled_db)
            block_idx += 1
        return np.concatenate(residual_blocks)

    nominal_phase_seed = nominal_phase_offsets_rad()
    if nominal_phase_seed.size != num_points:
        nominal_phase_seed = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
    phase_seed_bank = [
        nominal_phase_seed,
        wrap_phase_rad(nominal_phase_seed + np.pi / 8.0),
        np.zeros(num_points),
    ]
    reflection_candidates = amplitude_from_db(np.array([-30.0, -20.0, -10.0, -5.0]))
    delay_candidates_ns = estimate_delay_candidates_ns(
        corrected_measurements=[measurement[fit_indices] for measurement in corrected_measurements],
        fit_freqs_hz=fit_freqs,
        num_candidates=4,
    )
    best_guess: np.ndarray | None = None
    best_cost = float("inf")
    for reflection_amp in reflection_candidates:
        for delay_ns in delay_candidates_ns:
            for phase_seed in phase_seed_bank:
                initial_guess = np.concatenate(([reflection_amp, delay_ns], wrap_phase_rad(phase_seed)))
                try:
                    optimization = least_squares(
                        residual_vector,
                        x0=initial_guess,
                        bounds=(
                            np.concatenate(([amplitude_from_db(-35.0), 1.0], np.full(num_points, -np.pi))),
                            np.concatenate(([0.9, 35.0], np.full(num_points, np.pi))),
                        ),
                        method="trf",
                        ftol=1e-10,
                        xtol=1e-10,
                        gtol=1e-10,
                        max_nfev=90,
                    )
                except ValueError:
                    continue
                cost = float(np.mean(residual_vector(optimization.x) ** 2))
                if cost < best_cost:
                    best_cost = cost
                    best_guess = optimization.x.copy()

    if best_guess is None:
        raise RuntimeError("Phase-aware reflection fitting did not converge to a valid initial solution.")

    optimization = least_squares(
        residual_vector,
        x0=best_guess,
        bounds=(
            np.concatenate(([amplitude_from_db(-35.0), 1.0], np.full(num_points, -np.pi))),
            np.concatenate(([0.9, 35.0], np.full(num_points, np.pi))),
        ),
        method="trf",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=180,
    )
    fitted_amp = float(optimization.x[0])
    fitted_delay_ns = float(optimization.x[1])
    fitted_phases_rad = wrap_phase_rad(optimization.x[2:])
    rmse_db = float(np.sqrt(np.mean(residual_vector(optimization.x) ** 2)))
    return fitted_amp, fitted_delay_ns, fitted_phases_rad, rmse_db


def restore_los_psd(
    rx_psd: np.ndarray,
    dataset: SpectrumDataset,
    reflection_amp: float,
    delay_ns: float,
    point_phase_rad: float,
) -> tuple[np.ndarray, float]:
    noise_floor = estimate_noise_floor(rx_psd, dataset.noise_mask)
    corrected_rx_psd = np.maximum(rx_psd - noise_floor, EPS)
    composite_power = composite_power_response(dataset.freqs_hz, reflection_amp, delay_ns, point_phase_rad)
    restored_psd = corrected_rx_psd / np.maximum(composite_power, EPS)
    return restored_psd, noise_floor


def save_trp_academic_figure(
    dataset: SpectrumDataset,
    output_dir: Path,
    received_map: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]],
    measurement_map: dict[tuple[float, float], list[np.ndarray]],
    result_map: dict[tuple[float, float], ScenarioResult],
    restored_reference_map: dict[tuple[float, float], np.ndarray],
    restored_map: dict[tuple[float, float], np.ndarray],
    ideal_los_map: dict[tuple[float, float], np.ndarray],
    show_figures: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.6))
    fig.subplots_adjust(hspace=0.34, wspace=0.20, top=0.93, bottom=0.09)
    freq_mhz = dataset.freqs_hz / 1e6
    rep_key = (REPRESENTATIVE_REFLECTION_DB, REPRESENTATIVE_DELAY_NS)
    scenario_result = result_map[rep_key]
    position_offsets_m = measurement_position_offsets_m()
    line_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(position_offsets_m)))
    smoothed_tx = moving_average(normalize_psd_to_tx_peak(dataset.tx_psd, dataset.tx_psd), 81)

    ax = axes[0, 0]
    ax.plot(freq_mhz, smoothed_tx, color="#005f73", linewidth=1.8)
    ax.set_title("(a) Transmit spectrum")
    ax.set_ylabel("Norm. PSD (dB)")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_ylim(PSD_FLOOR_DB - 3.0, 5.0)
    limit_plot_region(ax, dataset.bandwidth_mhz)

    ax = axes[0, 1]
    for reflection_db in REFLECTION_LEVELS_DB:
        rx_psd, los_psd = received_map[(reflection_db, REPRESENTATIVE_DELAY_NS)]
        ax.plot(
            freq_mhz,
            smoothed_normalized_psd_db(rx_psd, dataset.tx_psd),
            linewidth=1.2,
            label=f"{reflection_db:.0f} dB",
        )
    ax.plot(
        freq_mhz,
        smoothed_normalized_psd_db(received_map[(REPRESENTATIVE_REFLECTION_DB, REPRESENTATIVE_DELAY_NS)][1], dataset.tx_psd),
        color="#94d2bd",
        linestyle="--",
        linewidth=1.2,
        label="LOS only",
    )
    ax.set_title(f"(b) Received ripple at {REPRESENTATIVE_DELAY_NS:.0f} ns")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_ylim(PSD_FLOOR_DB - 3.0, 5.0)
    limit_plot_region(ax, dataset.bandwidth_mhz)
    ax.legend(loc="lower left", frameon=True, ncol=2)

    ax = axes[1, 0]
    spectra = measurement_map[rep_key]
    for color, offset_m, spectrum in zip(line_colors, position_offsets_m, spectra):
        ax.plot(
            freq_mhz,
            moving_average(normalize_psd_to_tx_peak(spectrum, dataset.tx_psd), 81),
            color=color,
            linewidth=1.2,
            label=f"{offset_m * 1000:+.0f} mm",
        )
    ax.set_title(f"(c) Multi-point spectra, est. {scenario_result.reflection_db_est:.2f} dB / {scenario_result.delay_ns_est:.2f} ns")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Norm. PSD (dB)")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_ylim(PSD_FLOOR_DB - 3.0, 5.0)
    limit_plot_region(ax, dataset.bandwidth_mhz)
    ax.legend(loc="lower left", frameon=True, ncol=3)

    ax = axes[1, 1]
    measured_psd = received_map[rep_key][0]
    ax.plot(
        freq_mhz,
        smoothed_normalized_psd_db(measured_psd, dataset.tx_psd),
        color="#ca6702",
        alpha=0.75,
        label="Measured",
    )
    ax.plot(
        freq_mhz,
        smoothed_normalized_psd_db(restored_map[rep_key], dataset.tx_psd),
        color="#005f73",
        label="Restored LOS",
    )
    ax.plot(
        freq_mhz,
        smoothed_normalized_psd_db(ideal_los_map[rep_key], dataset.tx_psd),
        color="black",
        linestyle="--",
        linewidth=1.8,
        zorder=10,
        label="Ideal LOS",
    )
    ax.set_title("(d) LOS restoration")
    ax.set_xlabel("Frequency (MHz)")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_ylim(PSD_FLOOR_DB - 3.0, 5.0)
    limit_plot_region(ax, dataset.bandwidth_mhz)
    ax.legend(loc="lower left", frameon=True)

    fig.suptitle(f"TRP reflection mitigation at {dataset.bandwidth_mhz} MHz", fontsize=13)
    finalize_figure(fig, output_dir / f"trp_academic_bw{dataset.bandwidth_mhz}M.png", show_figures)


def save_original_spectrum_figure(datasets: list[SpectrumDataset], output_dir: Path, show_figures: bool) -> None:
    fig, axes = plt.subplots(len(datasets), 1, figsize=(7.2, 2.0 * len(datasets)), sharex=True)
    for ax, dataset in zip(as_axis_array(axes), datasets):
        normalized_psd_db = normalize_psd_to_tx_peak(dataset.tx_psd, dataset.tx_psd)
        ax.plot(dataset.freqs_hz / 1e6, normalized_psd_db, color="#005f73")
        ax.set_ylabel("Norm. PSD (dB)")
        ax.set_title(f"{dataset.bandwidth_mhz} MHz")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(PSD_FLOOR_DB - 3.0, 5.0)
        limit_plot_region(ax, dataset.bandwidth_mhz)
    as_axis_array(axes)[-1].set_xlabel("Baseband Frequency (MHz)")
    fig.tight_layout()
    finalize_figure(fig, output_dir / "01_original_tx_spectra.png", show_figures)


def save_received_spectra_figure(
    dataset: SpectrumDataset,
    output_dir: Path,
    received_map: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]],
    show_figures: bool,
) -> None:
    fig, axes = plt.subplots(
        len(REFLECTION_DELAYS_NS),
        len(REFLECTION_LEVELS_DB),
        figsize=(12.0, 7.2),
        sharex=True,
        sharey=True,
    )
    for row_idx, delay_ns in enumerate(REFLECTION_DELAYS_NS):
        for col_idx, reflection_db in enumerate(REFLECTION_LEVELS_DB):
            ax = axes[row_idx, col_idx]
            rx_psd, los_psd = received_map[(reflection_db, delay_ns)]
            ax.plot(
                dataset.freqs_hz / 1e6,
                normalize_psd_to_tx_peak(los_psd, dataset.tx_psd),
                color="#94d2bd",
                linestyle="--",
                label="LOS only",
            )
            ax.plot(
                dataset.freqs_hz / 1e6,
                normalize_psd_to_tx_peak(rx_psd, dataset.tx_psd),
                color="#bb3e03",
                label="LOS + reflection",
            )
            ax.set_title(f"{reflection_db:.0f} dB, {delay_ns:.0f} ns")
            ax.grid(True, linestyle="--", alpha=0.45)
            ax.set_ylim(PSD_FLOOR_DB - 3.0, 5.0)
            limit_plot_region(ax, dataset.bandwidth_mhz)
            if row_idx == len(REFLECTION_DELAYS_NS) - 1:
                ax.set_xlabel("Frequency (MHz)")
            if col_idx == 0:
                ax.set_ylabel("Norm. PSD (dB)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01), frameon=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.99))
    finalize_figure(fig, output_dir / f"02_received_spectra_bw{dataset.bandwidth_mhz}M.png", show_figures)


def save_measurement_points_figure(
    dataset: SpectrumDataset,
    output_dir: Path,
    measurement_map: dict[tuple[float, float], list[np.ndarray]],
    result_map: dict[tuple[float, float], ScenarioResult],
    show_figures: bool,
) -> None:
    fig, axes = plt.subplots(
        len(REFLECTION_DELAYS_NS),
        len(REFLECTION_LEVELS_DB),
        figsize=(12.0, 7.2),
        sharex=True,
        sharey=True,
    )
    position_offsets_m = measurement_position_offsets_m()
    line_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(position_offsets_m)))
    for row_idx, delay_ns in enumerate(REFLECTION_DELAYS_NS):
        for col_idx, reflection_db in enumerate(REFLECTION_LEVELS_DB):
            ax = axes[row_idx, col_idx]
            spectra = measurement_map[(reflection_db, delay_ns)]
            scenario_result = result_map[(reflection_db, delay_ns)]
            for color, offset_m, spectrum in zip(line_colors, position_offsets_m, spectra):
                ax.plot(
                    dataset.freqs_hz / 1e6,
                    normalize_psd_to_tx_peak(spectrum, dataset.tx_psd),
                    color=color,
                    linewidth=1.4,
                    label=f"{offset_m * 1000:+.0f} mm",
                )
            ax.set_title(
                f"{reflection_db:.0f} dB, {delay_ns:.0f} ns\n"
                f"Est: {scenario_result.reflection_db_est:.2f} dB, {scenario_result.delay_ns_est:.2f} ns",
                fontsize=10,
            )
            ax.grid(True, linestyle="--", alpha=0.45)
            ax.set_ylim(PSD_FLOOR_DB - 3.0, 5.0)
            limit_plot_region(ax, dataset.bandwidth_mhz)
            if row_idx == len(REFLECTION_DELAYS_NS) - 1:
                ax.set_xlabel("Frequency (MHz)")
            if col_idx == 0:
                ax.set_ylabel("Norm. PSD (dB)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.01), frameon=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.99))
    finalize_figure(fig, output_dir / f"03_measurement_points_bw{dataset.bandwidth_mhz}M.png", show_figures)


def save_restored_spectra_figure(
    dataset: SpectrumDataset,
    output_dir: Path,
    reflected_reference_map: dict[tuple[float, float], np.ndarray],
    restored_map: dict[tuple[float, float], np.ndarray],
    ideal_los_map: dict[tuple[float, float], np.ndarray],
    show_figures: bool,
) -> None:
    fig, axes = plt.subplots(
        len(REFLECTION_DELAYS_NS),
        len(REFLECTION_LEVELS_DB),
        figsize=(12.0, 7.2),
        sharex=True,
        sharey=True,
    )
    for row_idx, delay_ns in enumerate(REFLECTION_DELAYS_NS):
        for col_idx, reflection_db in enumerate(REFLECTION_LEVELS_DB):
            ax = axes[row_idx, col_idx]
            reflected_psd = reflected_reference_map[(reflection_db, delay_ns)]
            restored_psd = restored_map[(reflection_db, delay_ns)]
            ideal_los_psd = ideal_los_map[(reflection_db, delay_ns)]
            ax.plot(
                dataset.freqs_hz / 1e6,
                normalize_psd_to_tx_peak(reflected_psd, dataset.tx_psd),
                color="#ca6702",
                alpha=0.75,
                label="Measured",
            )
            ax.plot(
                dataset.freqs_hz / 1e6,
                normalize_psd_to_tx_peak(restored_psd, dataset.tx_psd),
                color="#005f73",
                label="Restored LOS",
            )
            ax.plot(
                dataset.freqs_hz / 1e6,
                normalize_psd_to_tx_peak(ideal_los_psd, dataset.tx_psd),
                color="#0a9396",
                linestyle="--",
                label="Ideal LOS",
            )
            ax.set_title(f"{reflection_db:.0f} dB, {delay_ns:.0f} ns")
            ax.grid(True, linestyle="--", alpha=0.45)
            ax.set_ylim(PSD_FLOOR_DB - 3.0, 5.0)
            limit_plot_region(ax, dataset.bandwidth_mhz)
            if row_idx == len(REFLECTION_DELAYS_NS) - 1:
                ax.set_xlabel("Frequency (MHz)")
            if col_idx == 0:
                ax.set_ylabel("Norm. PSD (dB)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01), frameon=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.99))
    finalize_figure(fig, output_dir / f"04_restored_spectra_bw{dataset.bandwidth_mhz}M.png", show_figures)


def save_power_error_figure(results: list[ScenarioResult], output_dir: Path, show_figures: bool) -> None:
    fig, axes = plt.subplots(len(TARGET_BW_MHZ), 1, figsize=(12.0, 2.4 * len(TARGET_BW_MHZ)), sharex=False)
    for ax, bandwidth_mhz in zip(as_axis_array(axes), TARGET_BW_MHZ):
        subset = [result for result in results if result.bandwidth_mhz == bandwidth_mhz]
        labels = [f"{result.reflection_db_true:.0f}dB\n{result.delay_ns_true:.0f}ns" for result in subset]
        x_axis = np.arange(len(subset))
        measured_errors = np.array([result.measured_error_db for result in subset])
        restored_errors = np.array([result.restored_error_db for result in subset])
        width = 0.38
        ax.bar(x_axis - width / 2.0, measured_errors, width=width, color="#ee9b00", label="Before removal")
        ax.bar(x_axis + width / 2.0, restored_errors, width=width, color="#0a9396", label="After removal")
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_ylabel("Power Error (dB)")
        ax.set_title(f"{bandwidth_mhz} MHz")
        ax.set_xticks(x_axis)
        ax.set_xticklabels(labels)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    handles, labels = as_axis_array(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01), frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 0.99))
    finalize_figure(fig, output_dir / "05_power_error_summary.png", show_figures)


def save_parameter_summary_figure(results: list[ScenarioResult], output_dir: Path, show_figures: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))

    true_reflection = np.array([result.reflection_db_true for result in results])
    est_reflection = np.array([result.reflection_db_est for result in results])
    true_delay = np.array([result.delay_ns_true for result in results])
    est_delay = np.array([result.delay_ns_est for result in results])

    axes[0].scatter(true_reflection, est_reflection, c="#005f73", alpha=0.8)
    reflection_line = np.linspace(true_reflection.min() - 1, true_reflection.max() + 1, 100)
    axes[0].plot(reflection_line, reflection_line, color="#bb3e03", linestyle="--")
    axes[0].set_xlabel("True Reflection Level (dB)")
    axes[0].set_ylabel("Estimated Reflection Level (dB)")
    axes[0].set_title("Reflection Level Estimation")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].scatter(true_delay, est_delay, c="#0a9396", alpha=0.8)
    delay_line = np.linspace(true_delay.min() - 1, true_delay.max() + 1, 100)
    axes[1].plot(delay_line, delay_line, color="#bb3e03", linestyle="--")
    axes[1].set_xlabel("True Delay (ns)")
    axes[1].set_ylabel("Estimated Delay (ns)")
    axes[1].set_title("Delay Estimation")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    finalize_figure(fig, output_dir / "06_parameter_estimation_summary.png", show_figures)


def save_summary_files(results: list[ScenarioResult], output_dir: Path) -> None:
    summary_path = output_dir / "trp_reflection_summary.csv"
    header = (
        "bandwidth_mhz,reflection_db_true,delay_ns_true,reflection_db_est,delay_ns_est,"
        "reflection_db_error,delay_ns_error,measured_power_db,restored_power_db,"
        "ideal_los_power_db,measured_error_db,restored_error_db,fit_rmse_db\n"
    )
    rows = []
    for result in results:
        rows.append(
            ",".join(
                [
                    f"{result.bandwidth_mhz:d}",
                    f"{result.reflection_db_true:.6f}",
                    f"{result.delay_ns_true:.6f}",
                    f"{result.reflection_db_est:.6f}",
                    f"{result.delay_ns_est:.6f}",
                    f"{result.reflection_db_error:.6f}",
                    f"{result.delay_ns_error:.6f}",
                    f"{result.measured_power_db:.6f}",
                    f"{result.restored_power_db:.6f}",
                    f"{result.ideal_los_power_db:.6f}",
                    f"{result.measured_error_db:.6f}",
                    f"{result.restored_error_db:.6f}",
                    f"{result.fit_rmse_db:.6f}",
                ]
            )
        )
    summary_path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")

    config = {
        "sample_rate_hz": FS_HZ,
        "carrier_hz": CARRIER_HZ,
        "snr_db": SNR_DB,
        "psd_floor_db": PSD_FLOOR_DB,
        "los_pathloss_db": LOS_PATHLOSS_DB,
        "reflection_levels_db": list(REFLECTION_LEVELS_DB),
        "reflection_delays_ns": list(REFLECTION_DELAYS_NS),
        "phase_jitter_std_deg": PHASE_JITTER_STD_DEG,
        "measurement_position_offsets_m": list(measurement_position_offsets_m()),
        "measurement_delay_offsets_ns": list(measurement_delay_offsets_ns()),
        "nominal_phase_offsets_rad": list(nominal_phase_offsets_rad()),
        "target_bandwidth_mhz": list(TARGET_BW_MHZ),
        "phase_model": "Unknown per-point phases are jointly fitted during inversion.",
    }
    (output_dir / "simulation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def run_trp_simulation(show_figures: bool = DEFAULT_SHOW_FIGURES) -> list[ScenarioResult]:
    output_dir = ensure_results_dir()
    clean_previous_figures(output_dir)
    datasets = prepare_datasets()

    all_results: list[ScenarioResult] = []
    reference_index = int(np.where(np.isclose(measurement_position_offsets_m(), 0.0))[0][0])

    for dataset in datasets:
        received_map: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
        measurement_map: dict[tuple[float, float], list[np.ndarray]] = {}
        restored_reference_map: dict[tuple[float, float], np.ndarray] = {}
        restored_map: dict[tuple[float, float], np.ndarray] = {}
        ideal_los_map: dict[tuple[float, float], np.ndarray] = {}
        result_map: dict[tuple[float, float], ScenarioResult] = {}

        for nominal_delay_ns in REFLECTION_DELAYS_NS:
            for reflection_db in REFLECTION_LEVELS_DB:
                measurement_psds: list[np.ndarray] = []
                reference_rx_psd = None
                reference_los_psd = None
                point_phases_rad = generate_measurement_phases_rad(
                    bandwidth_mhz=dataset.bandwidth_mhz,
                    reflection_db=reflection_db,
                    nominal_delay_ns=nominal_delay_ns,
                )
                for point_phase_rad in point_phases_rad:
                    rx_psd, los_psd, _noise_psd, _channel_power = simulate_received_psd(
                        dataset=dataset,
                        reflection_db=reflection_db,
                        delay_ns=nominal_delay_ns,
                        point_phase_rad=point_phase_rad,
                    )
                    measurement_psds.append(rx_psd)
                reference_rx_psd = measurement_psds[reference_index]
                reference_los_psd = los_psd
                if reference_rx_psd is None or reference_los_psd is None:
                    raise RuntimeError("Reference spectrum was not captured correctly.")

                estimated_amp, estimated_delay_ns, estimated_phases_rad, fit_rmse_db = fit_reflection_parameters(
                    measurement_psds=measurement_psds,
                    dataset=dataset,
                    reference_index=reference_index,
                )

                restored_los_psd, noise_floor = restore_los_psd(
                    rx_psd=measurement_psds[reference_index],
                    dataset=dataset,
                    reflection_amp=estimated_amp,
                    delay_ns=estimated_delay_ns,
                    point_phase_rad=float(estimated_phases_rad[reference_index]),
                )
                corrected_reference_psd = np.maximum(measurement_psds[reference_index] - noise_floor, EPS)

                measured_power = power_integral(corrected_reference_psd, dataset.freqs_hz, dataset.inband_mask)
                restored_power = power_integral(restored_los_psd, dataset.freqs_hz, dataset.inband_mask)
                ideal_los_power = power_integral(reference_los_psd, dataset.freqs_hz, dataset.inband_mask)

                result = ScenarioResult(
                    bandwidth_mhz=dataset.bandwidth_mhz,
                    reflection_db_true=reflection_db,
                    delay_ns_true=nominal_delay_ns,
                    reflection_db_est=db20(estimated_amp),
                    delay_ns_est=estimated_delay_ns,
                    reflection_db_error=db20(estimated_amp) - reflection_db,
                    delay_ns_error=estimated_delay_ns - nominal_delay_ns,
                    measured_power_db=db10(measured_power),
                    restored_power_db=db10(restored_power),
                    ideal_los_power_db=db10(ideal_los_power),
                    measured_error_db=db10(measured_power / ideal_los_power),
                    restored_error_db=db10(restored_power / ideal_los_power),
                    fit_rmse_db=fit_rmse_db,
                )
                all_results.append(result)

                scenario_key = (reflection_db, nominal_delay_ns)
                received_map[scenario_key] = (reference_rx_psd, reference_los_psd)
                measurement_map[scenario_key] = measurement_psds
                restored_reference_map[scenario_key] = corrected_reference_psd
                restored_map[scenario_key] = restored_los_psd
                ideal_los_map[scenario_key] = reference_los_psd
                result_map[scenario_key] = result

        if dataset.bandwidth_mhz in FIGURE_BW_MHZ:
            save_trp_academic_figure(
                dataset=dataset,
                output_dir=output_dir,
                received_map=received_map,
                measurement_map=measurement_map,
                result_map=result_map,
                restored_reference_map=restored_reference_map,
                restored_map=restored_map,
                ideal_los_map=ideal_los_map,
                show_figures=show_figures,
            )

    save_summary_files(all_results, output_dir)
    if show_figures:
        plt.show()
        plt.close("all")
    return all_results


def main(show_figures: bool = DEFAULT_SHOW_FIGURES) -> None:
    results = run_trp_simulation(show_figures=show_figures)
    mean_abs_before = float(np.mean(np.abs([result.measured_error_db for result in results])))
    mean_abs_after = float(np.mean(np.abs([result.restored_error_db for result in results])))
    max_abs_before = float(np.max(np.abs([result.measured_error_db for result in results])))
    max_abs_after = float(np.max(np.abs([result.restored_error_db for result in results])))

    print("TRP reflection simulation finished.")
    print(f"Result directory: {RESULTS_DIR}")
    print(f"Carrier frequency: {CARRIER_HZ / 1e9:.3f} GHz")
    print(f"Measurement position offsets: {measurement_position_offsets_m() * 1000.0} mm")
    print(f"Equivalent delay offsets: {measurement_delay_offsets_ns()} ns")
    print(f"Nominal carrier-phase offsets: {np.rad2deg(nominal_phase_offsets_rad())} deg")
    print(f"Injected phase jitter std: {PHASE_JITTER_STD_DEG:.1f} deg")
    print(f"Mean |power error| before removal: {mean_abs_before:.4f} dB")
    print(f"Mean |power error| after removal : {mean_abs_after:.4f} dB")
    print(f"Max  |power error| before removal: {max_abs_before:.4f} dB")
    print(f"Max  |power error| after removal : {max_abs_after:.4f} dB")


if __name__ == "__main__":
    main()
