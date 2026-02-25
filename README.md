# vircampype

`vircampype` is a Python data reduction pipeline for infrared imaging data obtained with the [VIRCAM](https://www.eso.org/sci/facilities/paranal/instruments/vircam.html) instrument on ESO's VISTA telescope. It was originally developed for the [VISIONS public survey](http://visions.univie.ac.at) and covers the full reduction chain from raw calibration frames through science-ready co-added images and photometric source catalogs.

An overview of the survey is given in [Meingast et al. 2023a](https://arxiv.org/abs/2303.08831). The algorithms and data processing steps implemented in this pipeline are described in [Meingast et al. 2023b](https://arxiv.org/abs/2303.08840).

---

## Features

- **Calibration pipeline**: bad pixel masks, linearity correction, dark subtraction, flat-fielding, gain tables
- **Science pipeline**: sky subtraction, source masking (via noisechisel or built-in methods), destriping, background subtraction, NaN interpolation
- **Astrometry**: SCAMP-based astrometric calibration against Gaia, with proper motion propagation
- **Photometry**: 2MASS-based photometric calibration with illumination correction
- **Coaddition**: SWarp-based resampling, stacking, and tile construction
- **Catalog building**: per-pawprint and per-tile source catalogs, public ESO Phase 3-compliant output
- **QC plots**: astrometric and photometric quality control diagnostic plots
- **Parallel processing**: multi-threaded execution via joblib
- **Checkpoint system**: interrupted runs resume from the last completed step

---

## Requirements

### Python
Python 3.13 or later is required. Python dependencies are listed in `requirements.txt` and include numpy, scipy, astropy, scikit-learn, scikit-image, matplotlib, astroquery, pyyaml, joblib, and regions.

### External tools
The following tools must be installed and available in `PATH`:

| Tool | Purpose |
|---|---|
| [SExtractor](https://github.com/astromatic/sextractor) | Source extraction |
| [SCAMP](https://github.com/astromatic/scamp) | Astrometric calibration |
| [SWarp](https://github.com/astromatic/swarp) | Image resampling and coaddition |
| [GNU Astro / noisechisel](https://www.gnu.org/software/gnuastro/) | Source mask generation |

---

## Installation

### From source
```bash
git clone https://github.com/smeingast/vircampype.git
cd vircampype
pip install -r requirements.txt
pip install -e .          # development install
# or
pip install .             # regular install
```

### Docker
A pre-built Docker image is available on [Docker Hub](https://hub.docker.com/r/smeingast/vircampype) and includes all external tools (SExtractor, SCAMP, SWarp, GNU Astro). This is the recommended way to run the pipeline without manually installing dependencies.

```bash
docker pull smeingast/vircampype
```

To build the image locally:
```bash
docker build -t vircampype .
```

---

## Quick Start

### 1. Sort raw files
Before running the pipeline, raw FITS files need to be sorted into calibration and science sub-folders:

```bash
vircampype --sort /path/to/raw/files/*.fits
```

### 2. Create a setup file
The pipeline is configured via a YAML file. A minimal science setup looks like this:

```yaml
name: my_field
path_data: /path/to/sorted/science/data
path_pype: /path/to/pipeline/output

n_jobs: 8
overwrite: false
qc_plots: true

build_tile: true
build_stacks: false
build_phase3: false
build_public_catalog: false
```

A separate setup file is needed for calibration data (the pipeline detects calibration runs when `name` contains `calibration`):

```yaml
name: calibration_2024
path_data: /path/to/sorted/calibration/data
path_pype: /path/to/pipeline/output
```

### 3. Run the pipeline
```bash
# Run calibration
vircampype --setup /path/to/calibration_setup.yml

# Run science reduction
vircampype --setup /path/to/science_setup.yml

# Reset progress (re-run from the start)
vircampype --reset-progress --setup /path/to/setup.yml

# Remove all generated object and phase3 folders
vircampype --clean --setup /path/to/setup.yml
```

When installed as a Python package, the `vircampype` command is available directly. Alternatively, invoke the worker script:

```bash
python vircampype/pipeline/worker.py --setup /path/to/setup.yml
```

---

## Pipeline Overview

### Calibration (`process_calibration`)
Processes a set of raw calibration frames and produces master calibration files:

1. **Master bad pixel mask** — from lamp flat frames
2. **Master linearity** — non-linearity correction table from lamp flats
3. **Master dark** — median-combined dark current frames
4. **Master gain table** — per-detector gain and read noise
5. **Master twilight flat** — normalised twilight flat fields
6. **Master weight map** — per-detector global weight images

### Science (`process_science`)
Processes raw science frames through to final co-added products:

1. **Basic raw processing** — linearity correction, dark subtraction, flat-fielding
2. **Source masking** — builds per-exposure source masks (noisechisel or built-in)
3. **Master sky** — constructs sky frames from a sliding window of exposures
4. **Final raw processing** — sky subtraction, destriping, background subtraction, NaN interpolation
5. **Astrometry (SCAMP)** — astrometric calibration against Gaia DR3
6. **Photometry (2MASS)** — photometric zero-point calibration
7. **Illumination correction** — variable or constant illumination correction map
8. **Resampling (SWarp)** — resamples exposures to a common grid
9. **Stacks / Tile** — co-adds resampled images into per-offset stacks and a final tile
10. **Statistics images** — per-pixel exposure time, image count, astrometric RMS, and MJD maps
11. **Source catalogs** — SExtractor source extraction on stacks and tile
12. **QC plots** — astrometric and photometric diagnostic plots
13. **Phase 3 / Public catalog** — ESO Phase 3-compliant output and public source catalog

---

## Key Configuration Parameters

All parameters below are set in the YAML setup file. Default values are used when a parameter is omitted.

| Parameter | Default | Description |
|---|---|---|
| `name` | — | Pipeline run name (required) |
| `path_data` | — | Path to input FITS files (required) |
| `path_pype` | — | Path for pipeline output (required) |
| `n_jobs` | `8` | Number of parallel threads |
| `overwrite` | `false` | Overwrite existing output files |
| `qc_plots` | `true` | Generate QC diagnostic plots |
| `build_stacks` | `false` | Build per-offset stacks |
| `build_tile` | `true` | Build final co-added tile |
| `build_phase3` | `false` | Build ESO Phase 3 products |
| `build_public_catalog` | `false` | Build public source catalog |
| `destripe` | `true` | Apply destriping correction |
| `subtract_background` | `true` | Subtract 2D background model |
| `flat_type` | `twilight` | Flat field type: `twilight` or `sky` |
| `scamp_mode` | `loose` | SCAMP mode: `loose` or `fix_focalplane` |
| `illumination_correction_mode` | `variable` | IC mode: `variable` or `constant` |
| `source_mask_method` | `noisechisel` | Source masking: `noisechisel` or `built-in` |
| `resampling_kernel` | `lanczos3` | SWarp resampling kernel |
| `mask_bright_galaxies` | `true` | Mask bright galaxies from de Vaucouleurs (1991) |

---

## Output Structure

The pipeline creates the following folder structure under `path_pype`:

```
path_pype/
├── master_common/      # Master calibration files (shared across runs)
├── master_object/      # Master sky, source masks, illumination corrections
├── headers/            # SCAMP astrometric headers
├── processed/          # Calibrated pawprint images
├── resampled/          # Resampled pawprint images
├── stacks/             # Per-offset stack images and catalogs
├── tile/               # Final co-added tile image and catalog
├── statistics/         # Statistics images (exptime, nimg, astrms, mjd)
├── phase3/             # ESO Phase 3-compliant products
├── qc/                 # Quality control plots
└── temp/               # Temporary files and pipeline status
```

---

## Testing

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## Issues

If you encounter a bug or unexpected behaviour, please [open an issue](https://github.com/smeingast/vircampype/issues) on GitHub. Include the pipeline log file (found in `path_pype/temp/`) and a description of the setup if possible.

---

## Citation

If you use `vircampype` in your research, please cite:

- Meingast et al. 2023a — [VISIONS survey overview](https://arxiv.org/abs/2303.08831)
- Meingast et al. 2023b — [Pipeline description](https://arxiv.org/abs/2303.08840)
