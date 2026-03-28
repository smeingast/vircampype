# vircampype

`vircampype` is a Python data reduction pipeline for infrared imaging data obtained with the [VIRCAM](https://www.eso.org/sci/facilities/paranal/instruments/vircam.html) instrument on ESO's VISTA telescope. It was originally developed for the [VISIONS public survey](http://visions.univie.ac.at) and covers the full reduction chain from raw calibration frames through science-ready co-added images and photometric source catalogs.

An overview of the survey is given in [Meingast et al. 2023a](https://arxiv.org/abs/2303.08831). The algorithms and data processing steps implemented in this pipeline are described in [Meingast et al. 2023b](https://arxiv.org/abs/2303.08840).

---

## Features

- **Calibration pipeline**: bad pixel masks, per-channel linearity correction, dark subtraction, flat-fielding, gain tables
- **Science pipeline**: sky subtraction, source masking (via noisechisel or built-in methods), destriping, background subtraction, NaN interpolation
- **Astrometry**: SCAMP-based astrometric calibration against Gaia, with proper motion propagation and optional header caching
- **Photometry**: 2MASS-based photometric calibration with illumination correction
- **Local reference catalogs**: optional pre-downloaded Gaia and 2MASS catalogs to avoid online queries
- **Coaddition**: SWarp-based resampling, stacking, and tile construction
- **Catalog building**: per-pawprint and per-tile source catalogs, public ESO Phase 3-compliant output
- **Completeness testing**: artificial star injection/recovery via SkyMaker and PSFEx to measure detection completeness as a function of magnitude
- **QC plots**: astrometric and photometric quality control diagnostic plots
- **Parallel processing**: parallel execution via joblib (threads or processes depending on the operation)
- **Checkpoint system**: interrupted runs resume from the last completed step
- **Cluster batch processing**: distribute jobs across multiple machines via SSH + Docker using a shared NAS (see [`cluster/README.md`](cluster/README.md))

---

## Requirements

### Python
Python 3.13 or later is required. Python dependencies are declared in `pyproject.toml` and include numpy, scipy, astropy, scikit-learn, scikit-image, matplotlib, astroquery, pyyaml, joblib, regions, tqdm, and pillow.

### External tools
The following tools must be installed and available in `PATH`:

| Tool | Purpose |
|---|---|
| [SExtractor](https://github.com/astromatic/sextractor) | Source extraction |
| [SCAMP](https://github.com/astromatic/scamp) | Astrometric calibration |
| [SWarp](https://github.com/astromatic/swarp) | Image resampling and coaddition |
| [PSFEx](https://github.com/astromatic/psfex) | PSF modelling (completeness testing) |
| [SkyMaker](https://github.com/astromatic/skymaker) | Artificial image generation (completeness testing) |
| [GNU Astro / noisechisel](https://www.gnu.org/software/gnuastro/) | Source mask generation |

---

## Installation

### From source
```bash
git clone https://github.com/smeingast/vircampype.git
cd vircampype
pip install -e .          # development install (includes all dependencies)
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
python vircampype/pipeline/worker.py --sort /path/to/raw/files/*.fits
```

### 2. Create a setup file
The pipeline is configured via a YAML file. A minimal science setup looks like this:

```yaml
name: my_field
path_data: /path/to/sorted/science/data
path_pype: /path/to/pipeline/output
path_master_common: /path/to/pipeline/output/master/

n_jobs: -2
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
path_master_common: /path/to/pipeline/output/master/
```

### 3. Run the pipeline
```bash
# Run calibration
python vircampype/pipeline/worker.py --setup /path/to/calibration_setup.yml

# Run science reduction
python vircampype/pipeline/worker.py --setup /path/to/science_setup.yml

# Reset progress (re-run from the start)
python vircampype/pipeline/worker.py --reset-progress --setup /path/to/setup.yml

# Remove all generated object and phase3 folders
python vircampype/pipeline/worker.py --clean --setup /path/to/setup.yml

# Clear cached header databases for this setup
python vircampype/pipeline/worker.py --clean-cache --setup /path/to/setup.yml

# Validate setup paths without processing
python vircampype/pipeline/worker.py --dry-run --setup /path/to/setup.yml
```

### 4. Cluster batch processing (optional)

To process many configs across multiple machines, use the cluster batch system:

```bash
vircampype --cluster cluster.yml              # queue + dispatch to all nodes
vircampype --cluster cluster.yml --status     # monitor progress
vircampype --cluster cluster.yml --abort      # stop all workers and reset
```

Each node only needs Docker and SSH access — no Python or vircampype install required. See [`cluster/README.md`](cluster/README.md) for full setup instructions.

---

## Pipeline Overview

### Calibration (`process_calibration`)
Processes a set of raw calibration frames and produces master calibration files:

1. **Master bad pixel mask** — from lamp flat frames
2. **Master linearity** — per-channel non-linearity correction table from lamp flats (16 readout channels per detector)
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
13. **QC summary table** — aggregates key metrics from stacks and tile into `qc/qc_summary.ecsv`
14. **Completeness** — artificial star injection/recovery to measure detection completeness per sub-tile (optional, via `build_completeness`)
15. **Phase 3 / Public catalog** — ESO Phase 3-compliant output and public source catalog

---

## Key Configuration Parameters

All parameters below are set in the YAML setup file. Default values are used when a parameter is omitted.

| Parameter | Default | Description |
|---|---|---|
| `name` | — | Pipeline run name (required) |
| `path_data` | — | Path to input FITS files (required) |
| `path_pype` | — | Path for pipeline output (required) |
| `path_master_common` | — | Path to shared master calibration files (required) |
| `path_master_object` | `null` | Path to object-specific calibration files (default: `<path_pype>/<name>/calibration/`) |
| `n_jobs` | `-2` | Parallel workers: `0` = all physical cores, `-1` = all minus 1, `-2` = all minus 2, `>0` = exact count (capped at physical cores) |
| `overwrite` | `false` | Overwrite existing output files |
| `qc_plots` | `true` | Generate QC diagnostic plots |
| `build_stacks` | `false` | Build per-offset stacks |
| `build_tile` | `true` | Build final co-added tile |
| `build_phase3` | `false` | Build ESO Phase 3 products |
| `build_public_catalog` | `false` | Build public source catalog |
| `build_completeness` | `false` | Run completeness analysis on tile |
| `destripe` | `true` | Apply destriping correction |
| `subtract_background` | `true` | Subtract 2D background model |
| `flat_type` | `twilight` | Flat field type: `twilight` or `sky` |
| `scamp_mode` | `loose` | SCAMP mode: `loose` or `fix_focalplane` |
| `illumination_correction_mode` | `variable` | IC mode: `variable` or `constant` |
| `source_mask_method` | `noisechisel` | Source masking: `noisechisel` or `built-in` |
| `resampling_kernel` | `lanczos3` | SWarp resampling kernel |
| `local_gaia_catalog` | `null` | Path to local Gaia FITS catalog (skip Vizier download) |
| `local_2mass_catalog` | `null` | Path to local 2MASS FITS catalog (skip Vizier download) |
| `scamp_cache_dir` | `null` | Directory for caching SCAMP `.ahead` header files |
| `local_cache_dir` | `null` | Local directory for temp files, header cache, and SWarp swap (default: system temp) |
| `mask_bright_galaxies` | `true` | Mask bright galaxies from de Vaucouleurs (1991) |

---

## Output Structure

The pipeline creates the following folder structure under `path_pype`:

```
path_pype/
└── <name>/                         # Per-target output folder
    ├── calibration/                # Object-specific masters (sky, source masks; configurable via path_master_object)
    ├── processing/
    │   ├── basic/                  # Basic-calibrated pawprint images
    │   ├── final/                  # Final-calibrated pawprint images
    │   ├── illumcorr/              # Illumination correction maps
    │   ├── resampled/              # Resampled pawprint images
    │   └── statistics/             # Statistics images (exptime, nimg, astrms, mjd)
    ├── products/
    │   ├── stacks/                 # Per-offset stack images and catalogs
    │   ├── tile/                   # Final co-added tile image and catalog
    │   └── phase3/                 # ESO Phase 3-compliant products
    ├── qc/                         # Quality control plots
    │   ├── astrometry/
    │   ├── photometry/
    │   ├── sky/
    │   ├── illumcorr/
    │   └── completeness/           # Completeness QC plots and FITS image
    └── temp/                       # Temporary files and pipeline status
        └── completeness/           # Completeness sub-tiles and PSF models
            ├── tiles/
            └── psf/
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
