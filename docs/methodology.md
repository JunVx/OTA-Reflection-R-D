# Methodology Summary

This document summarizes the research framing for residual chamber-reflection effects in wideband SISO OTA measurements.

## 1) Wideband reflection-induced uncertainty

Under a normalized line-of-sight (LOS) path and one dominant reflected path, with approximately uniform power spectral density over bandwidth \(B\), the in-band averaged normalized received power is:

\[
P_{avg} = 1 + a^2 + 2a\,\mathrm{sinc}(B\tau)\cos(\phi)
\]

with envelope bounds:

\[
P_{max/min} = 1 + a^2 \pm 2a\,|\mathrm{sinc}(B\tau)|
\]

where:
- \(a\): reflected-path amplitude relative to LOS,
- \(\tau\): relative delay,
- \(\phi\): effective spatial phase,
- \(\mathrm{sinc}(x)=\sin(\pi x)/(\pi x)\), \(\mathrm{sinc}(0)=1\).

Interpretation:
- The fluctuation term scales with \(|\mathrm{sinc}(B\tau)|\), so larger effective delay-bandwidth products suppress first-order fluctuation.
- A residual reflection-energy component \(a^2\) remains in the averaged power even when phase-dependent fluctuation is reduced.

### Closed-form assumptions

- Dominant two-path model (one LOS + one dominant reflection).
- Narrow spatial region per local estimate.
- Uniform (or approximately uniform) PSD over integrated bandwidth.
- Reflection parameters treated as locally stable over analyzed snapshots.

## 2) Sparse frequency-spatial de-embedding for EIRP/TRP

For radiated-power-oriented analysis, the documented workflow is:

1. Measure received spectra at a small set of nearby positions with sufficient phase diversity.
2. Apply noise-floor correction.
3. Form spectral ratios against a reference position.
4. Jointly estimate reflection amplitude, delay, and spatial phases via nonlinear least squares.
5. Reconstruct an LOS-equivalent spectrum.
6. Integrate in-band power for EIRP-level quantities and related TRP-oriented processing.

Notes:
- Spatial displacements and reflection parameters are **estimated**, not assumed known inputs.
- This documentation describes method logic; it does not claim completed repository reproduction.

## 3) Sparse spatial averaging for EIS/TIS

For sensitivity-oriented analysis, DUT internal received spectra are not directly observable in the same way as external receive spectra used in TRP de-embedding. Therefore, the above spectral de-embedding approach is not directly transferable.

The manuscript-motivated alternative is localized multi-position averaging in the **linear power domain**. With approximately uniform local phase coverage, first-order fluctuation tends to cancel, while a residual factor \(1+a^2\) remains. This mitigates but does not fully remove reflection effects.

## Validation scope clarification

This companion distinguishes:

- **EIRP/EIS-level method validation**: validating the uncertainty behavior and local mitigation concepts at equivalent isotropic metrics.
- **Complete TRP/TIS workflows**: full chamber procedures, sampling plans, instrument integration, and compliance-grade processing.

The latter requires additional implementation and measurement infrastructure not yet included in this initial documentation-only setup.
