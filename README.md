# OTA Reflection R&D

Maintainer: Jun Li ([JunVx](https://github.com/JunVx))

This repository is the research-code companion for the manuscript:

**"Uncertainty Quantification and Mitigation of Residual Chamber-Reflection Effects in Wideband SISO OTA Measurements"**

## Project overview

The project supports research into residual chamber-reflection effects in wideband single-input single-output (SISO) over-the-air (OTA) measurements. It is intended for antenna researchers, RF measurement engineers, and developers of wireless testing tools.

The goal is to make methods easier to inspect, understand, and reproduce once the maintainer uploads the original research code and authorized supporting data.

## Research motivation and scope

This repository documents three focused research topics:

1. **Wideband reflection-induced uncertainty**
   - Closed-form relationship for normalized integrated received power with one dominant reflected path:

     \[
     P_{avg} = 1 + a^2 + 2a\,\mathrm{sinc}(B\tau)\cos(\phi)
     \]

     \[
     P_{max/min} = 1 + a^2 \pm 2a\,|\mathrm{sinc}(B\tau)|
     \]

   - Definitions: reflected-path amplitude ratio \(a\), bandwidth \(B\) [Hz], relative delay \(\tau\) [s], and effective spatial phase \(\phi\).
   - Uses normalized \(\mathrm{sinc}(x)=\sin(\pi x)/(\pi x)\), with \(\mathrm{sinc}(0)=1\).

2. **Sparse frequency-spatial de-embedding for EIRP/TRP**
   - Multi-position received spectra with sufficient phase diversity.
   - Workflow summary: noise-floor correction, spectral ratios to a reference position, joint nonlinear least-squares estimation of reflection parameters, LOS-equivalent spectrum reconstruction, and in-band power integration.

3. **Sparse spatial averaging for EIS/TIS**
   - Sensitivity measurements cannot directly access the DUT internal received spectrum.
   - Therefore, TRP spectral de-embedding is not directly transferable.
   - Localized multi-position linear-power averaging can reduce first-order fluctuation under approximately uniform phase coverage, while a residual factor \(1+a^2\) remains.

## Terminology

- **OTA**: Over-the-air measurement in a controlled RF environment.
- **SISO**: Single-input single-output wireless link model.
- **TRP**: Total radiated power.
- **TIS**: Total isotropic sensitivity.
- **EIRP**: Equivalent isotropically radiated power.
- **EIS**: Equivalent isotropic sensitivity.

## Repository structure

- `docs/` — methodology and reproducibility guidance.
- `src/` — placeholder for original research code upload.
- `examples/` — placeholder for minimal, clearly labeled examples.
- `tests/` — placeholder for verification assets.
- `data/` — placeholder for documented datasets and metadata.
- `results/` — placeholder for generated outputs and figure artifacts.
- `.github/ISSUE_TEMPLATE/` — issue templates for bugs and reproduction questions.

## Project status

> Initial documentation-only setup. The original research code and approved supporting data will be uploaded by the maintainer.

This repository currently does **not** claim reproduced manuscript results.

## Installation (pending code upload)

Pending maintainer upload of the original codebase and dependency definitions.

## Usage (pending code upload)

Pending maintainer upload of executable scripts/modules, input formats, and run commands.

## Reproducibility guide

See [docs/reproducibility.md](docs/reproducibility.md).

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening issues or pull requests.
