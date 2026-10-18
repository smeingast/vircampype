<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/icon/vircampype-icon-dark.png">
    <img src="assets/icon/vircampype-icon-light.png" alt="vircampype icon" width="160">
  </picture>
</p>

<h1 align="center">vircampype</h1>

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
- **Catalog building**: per-pawprint and per-tile source catalogs, public ESO Phase 3-compliant output with full RA/Dec position-error covariance (including the SCAMP cross-axis correlation)
- **Completeness testing**: artificial star injection/recovery via SkyMaker and PSFEx to measure detection completeness as a function of magnitude
- **QC plots**: astrometric and photometric quality control diagnostic plots
- **Parallel processing**: parallel execution via joblib (threads or processes depending on the operation)
- **Checkpoint system**: interrupted runs resume from the last completed step
- **Cluster batch processing**: run many configs on several machines via SSH + Docker using a shared NAS (`vircampype-cluster`) (see [`cluster/README.md`](cluster/README.md))

---

## Requirements

### Python
Python 3.13 or later is required. Python dependencies are declared in `pyproject.toml` and include numpy, scipy, astropy, scikit-learn, scikit-image, matplotlib, astroquery, pyyaml, joblib, regions, rich, and pillow.

### External tools
The following tools must be installed and available in `PATH`:

| Tool | Purpose |
|---|---|
| [SExtractor](https://github.com/smeingast/sextractor/tree/feature/multi-seeing-class-star) | Source extraction (requires the vircampype fork: multi-seeing CLASS_STAR, `WINPOS_SIGMAP` window cap) |
| [SCAMP](https://github.com/astromatic/scamp) | Astrometric calibration |
| [SWarp](https://github.com/astromatic/swarp) | Image resampling and coaddition |
| [PSFEx](https://github.com/astromatic/psfex) | PSF modelling (completeness testing) |
| [SkyMaker](https://github.com/astromatic/skymaker) | Artificial image generation (completeness testing) |
| [GNU Astro / noisechisel](https://www.gnu.org/software/gnuastro/) | Source mask generation |

**Important:** This pipeline requires a [modified SExtractor build](https://github.com/smeingast/sextractor/tree/feature/multi-seeing-class-star) that supports multi-seeing CLASS_STAR output. The upstream astromatic SExtractor will **not** work when star/galaxy classification is enabled (`source_classification: true`). The pipeline verifies this at startup and will raise an error if the installed binary lacks multi-seeing support. The same fork (commit `5cb35ce`, pinned in the Dockerfile) adds the `WINPOS_SIGMAP` parameter used by `sex_winpos_cap`; with that option on, the pipeline checks for it at startup as well. The Docker image includes the correct build.

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
python vircampype/pipeline/worker.py --reset progress --setup /path/to/setup.yml

# Clear cached header databases for this setup
python vircampype/pipeline/worker.py --reset cache --setup /path/to/setup.yml

# Remove all generated output folders
python vircampype/pipeline/worker.py --reset all --setup /path/to/setup.yml

# Validate setup paths without processing
python vircampype/pipeline/worker.py --dry-run --setup /path/to/setup.yml
```

### 4. Cluster batch processing (optional)

To process many configs on several machines (SSH + Docker + a shared NAS):

```bash
vircampype-cluster check  cluster.yml    # test every node first
vircampype-cluster start  cluster.yml    # queue the configs and start the workers
vircampype-cluster status cluster.yml    # monitor progress
```

Worker machines only need Docker and SSH. See [`cluster/README.md`](cluster/README.md) for full setup instructions.

---

## Pipeline Overview

### Calibration (`process_calibration`)
Processes a set of raw calibration frames and produces master calibration files:

1. **Master bad pixel mask** — unstable pixels from lamp flat frames, plus pixels with excess noise in pairs of linearity darks
2. **Master linearity** — per-channel non-linearity correction table from lamp flats (16 readout channels per detector), fitted on robust per-channel levels up to a fraction of the saturation level, with one 5-sigma outlier-rejection pass (sequences must have NDIT = 1)
3. **Master dark** — min/max-rejected mean of dark frames
4. **Master gain table** — per-detector gain and read noise from robust (MAD-based) per-channel variances
5. **Master twilight flat** — normalised twilight flat fields
6. **Master weight map** — per-detector global weight images

### Science (`process_science`)
Processes raw science frames through to final co-added products:

1. **Basic raw processing** — linearity correction, dark subtraction, flat-fielding
2. **Source masking** — builds per-exposure source masks (noisechisel or built-in)
3. **Master sky** — constructs sky frames from a sliding window of exposures
4. **Final raw processing** — sky subtraction, destriping, background subtraction, NaN interpolation
5. **Astrometry (SCAMP)** — astrometric calibration against Gaia DR3
6. **Photometry (2MASS)** — photometric zero-point calibration, with a data-driven bright limit for the reference stars
7. **Illumination correction** — variable or constant illumination correction map
8. **Resampling (SWarp)** — resamples exposures to a common grid
9. **Stacks / Tile** — co-adds resampled images into per-offset stacks and a final tile
10. **Statistics images** — per-pixel exposure time, image count, astrometric RMS, RA/Dec astrometric correlation, and MJD maps
11. **Source catalogs** — SExtractor source extraction on stacks and tile (optionally with the WIN window cap, `sex_winpos_cap`)
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
| `bpm_noise_mask` | `true` | Add pixels with excess temporal noise (from pairs of linearity darks) to the master bad pixel mask |
| `bpm_noise_threshold` | `5` | Noise threshold for the bad pixel mask, in units of the readout-channel median noise |
| `linearity_fit_fraction` | `0.85` | Fit the per-channel linearity only on levels below this fraction of the detector saturation level |
| `saturation_override` | `null` | Per-detector saturation levels in ADU that replace the preset (`set_saturation_levels`), e.g. `{10: 22000, 12: 20000}`. Used by the linearity fit cap, twilight masks, gain and the science `SATURATE` keyword (capped below the linearity turnover at short DIT) |
| `destripe` | `true` | Apply destriping correction |
| `subtract_background` | `true` | Subtract 2D background model |
| `flat_type` | `twilight` | Flat field type: `twilight` or `sky` |
| `scamp_mode` | `loose` | SCAMP mode: `loose` or `fix_focalplane` |
| `astrms_self_build_map` | `true` | The bright-star astrometric position-error floor is self-measured on the co-added tile vs epoch-warped Gaia (replacing the over-estimated flat SCAMP `ASTRMSH` floor, ~2–2.5× too large at the bright end); sparse/cloud fields fall back to the SCAMP floor automatically. This knob toggles the spatial floor map — an edge-preserving fixed-size-cell (`astrms_self_cell_arcmin`, ~one detector) grid smoothed by a 3×3 median, which keeps the per-detector/per-tile floor steps a smooth surface would blur (`false` = a single scalar floor per tile). Advanced `astrms_self_*` knobs tune the clean sample, cell map, and fallback |
| `illumination_correction_mode` | `variable` | IC mode: `variable` or `constant` |
| `source_mask_method` | `noisechisel` | Source masking: `noisechisel` or `built-in` |
| `resampling_kernel` | `lanczos3` | SWarp resampling kernel |
| `statistics_coadd` | `numpy` | Tile statistics maps (nimg, exptime, MJD): `numpy` builds them without SWarp, bit-identical to SWarp 2.41.6 and with bounded RAM (`statistics_coadd_mem_max`, MB, default 2048); inputs it cannot reproduce fall back to SWarp. `swarp` always uses SWarp |
| `photometric_error_correlated_noise` | `true` | Inflate the public-catalog aperture-magnitude errors for correlated noise: `MAGERR = sqrt(kappa(D)^2 MAGERR_APER^2 + floor^2)`, `kappa(D)^2 = c^2 kappa_resamp(D)^2 + beta pi min(D, dmax)^2/4`, with `kappa_resamp` the noise correlation of the SWarp kernel (computed from `resampling_kernel` and the coadd/`native_pixel_scale` ratio). `false` reproduces the uninflated errors exactly |
| `photometric_error_noise_scale` | `1.0` | Per-pixel noise calibration `c` of the inflation (empirical, from repeat measurements) |
| `photometric_error_background_beta` | `0.0084` | Coherent-background term `beta` of the inflation (calibrated on VISIONS H repeats) |
| `photometric_error_background_dmax` | `8.0` | Aperture diameter (px) beyond which the `beta` term is held constant (largest calibrated aperture) |
| `native_pixel_scale` | `0.337` | Native detector pixel scale (arcsec) used for the resampling ratio |
| `local_gaia_catalog` | `null` | Path to local Gaia FITS catalog (skip Vizier download) |
| `local_2mass_catalog` | `null` | Path to local 2MASS FITS catalog (skip Vizier download) |
| `reference_mag_lo` / `reference_mag_hi` | `null` | Bright and faint 2MASS magnitude limits for the reference stars (band defaults if unset) |
| `reference_mag_lo_auto` | `true` | Move the bright limit brighter where the data show no saturation or non-linearity: the offset to 2MASS of point sources in the first calibrated products is checked in 0.25 mag bins and the limit stops 0.25 mag before a 20 mmag departure (at most 2.25 mag brighter, never fainter than `reference_mag_lo`). Used for the zero points, the public-catalogue 2MASS replacement (decided by magnitude alone; `FLAGS & 4`, the saturation bit, does not trigger it and is informational only: it stays in `SFLG` but does not make `CFLG`/`QFLG` X), `MAGSAT` and QC (the illumination correction runs earlier and keeps the setup limit); stored in `temp/reference_mag_lo.json` and the catalogue headers (`PYPE REFMAG LO`) |
| `scamp_cache_dir` | `null` | Directory for caching SCAMP `.ahead` header files |
| `local_cache_dir` | `null` | Local directory for disposable temp (header cache, SWarp swap, completeness sub-tiles). Default `null` uses the system temp dir for caches/swap and `path_pype/<name>/temp` for completeness |
| `source_classification` | `false` | Run star/galaxy classification (requires multi-seeing SExtractor fork) |
| `sex_winpos_cap` | `false` | Cap SExtractor's windowed-centroid (WIN) window at the local stellar size in the tile catalogue (requires the fork's `WINPOS_SIGMAP`). SExtractor's window has `sigma = 2 FLUX_RADIUS / 2.35`, so extended sources get wide windows that pick up epoch-dependent PSF asymmetry and a spurious proper motion (CrA galaxies: +0.6 mas/yr); with the cap each window is `min(own, local stellar sigma)`, which removes the galaxy PM while stars move by ≤ 0.6 mas. Products: `WINVAR` (stellar window sigma², arcsec², 0 = unknown) in the detector HDUs of the per-pawprint statistics weights (written at tile level from the SCAMP catalogue and `.ahead`; mosaics read it through their statistics symlinks), `<tile>.winsig.fits` (sigma map, arcsec) and `<tile>.winsig.map.fits` (the SExtractor map, pixels) next to the coadd, and `WINPSMAP` in the tile catalogue header. Enabling it on an existing reduction: reset the `photometry_tile` status and remove the tile `full` catalogue and everything derived from it (a catalogue made with the other setting raises); tiles need `WINVAR` before their mosaics |
| `classification_n_seeing` | `32` | Number of SEEING_FWHM grid points for classification |
| `classification_seeing_step_warn` | `0.05` | Warn if classification FWHM step exceeds this value (arcsec) |
| `mask_bright_galaxies` | `true` | Mask bright galaxies from de Vaucouleurs (1991) |
| `pushover_user_key` | `null` | Pushover user key for pipeline completion/failure notifications |
| `pushover_api_token` | `null` | Pushover API token for pipeline completion/failure notifications |

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
    │   └── statistics/             # Statistics images (exptime, nimg, astrms, astrms_corr, mjd; WINVAR headers with sex_winpos_cap)
    ├── products/
    │   ├── stacks/                 # Per-offset stack images and catalogs
    │   ├── tile/                   # Final co-added tile image and catalog
    │   └── phase3/                 # ESO Phase 3-compliant products
    ├── qc/                         # Quality control plots
    │   ├── astrometry/
    │   ├── photometry/
    │   ├── sky/
    │   ├── illumcorr/
    │   ├── psf/                    # PSF diagnostic plots and PSFEx checkplots
    │   └── completeness/           # Completeness QC plots and FITS image
    └── temp/                       # Temporary files and pipeline status
        └── completeness/           # Completeness sub-tiles and PSF models
            ├── tiles/
            └── psf/
```

With `local_cache_dir` set, the disposable temporaries (header cache, SWarp
swap, and the completeness sub-tiles/PSF models) are routed to
`<local_cache_dir>/<name>/...`, keeping that churn off network storage. Without
it, caches and swap fall back to the system temp dir and the completeness
temporaries stay under `path_pype/<name>/temp`. These files are re-derived as
needed and are safe to delete between runs.

---

## Public catalogue flags

`*.full.fits.ptab` (built when `build_public_catalog` is set) carries three flag columns:

- `SFLG`: SExtractor `FLAGS` unchanged (0 for 2MASS rows). Bits 1 (neighbours), 2 (deblended) and 4 (saturated
  pixel) are informational; bits 8-128 (truncated, corrupted data, memory overflow) are fatal (`CFLG_REASON` 16).
  For the most precise photometry/astrometry use `SFLG == 0` (or `SFLG & 1 == 0` if deblended sources are fine).
- `CFLG` (bool) is True if any `CFLG_REASON` bit is set; `CFLG_REASON` (sum of bits):

  | bit | meaning |
  |---|---|
  | 1 | brightest aperture magnitude brighter than the bright limit (`PYPE REFMAG LO`) |
  | 2 | `FLUX_AUTO` not >= 0.01 (incl. non-positive / NaN) |
  | 4 | `SNR_WIN` not > 0 (incl. NaN) |
  | 8 | `NIMG` < 1 |
  | 16 | fatal SExtractor flag (bits 8-128) |
  | 32 | `FLAGS_WEIGHT` > 0 (zero-weight pixels in the footprint) |
  | 64 | `FWHM_WORLD` <= 0.2 arcsec |
  | 128 | `MJDEFF` not > 0 (such rows are removed by the final row cut, so it never appears) |
  | 256 | no source classification (`CLASS_STAR_INTERP` NaN) |
  | 512 | growth-curve outlier: more concentrated than point sources of the same magnitude (3 px vs 4 px aperture-matched magnitude) |
  | 1024 | nebula: high local background relative to the 3 px flux, growth 3->4 px >= 0.2 mag, non-stellar |
  | 2048 | 2MASS row of a bright star whose 2MASS entry fails the quality purge (MAGERR may be NaN) |

- `QFLG`: A/B/C/D for `MAGERR` < 0.10857 / 0.15510 / 0.21714 / otherwise (S/N 10/7/5, as 2MASS) when `CFLG` is
  False; X when `CFLG` is True. 2MASS rows (`SURVEY` = "2MASS") replace survey detections around stars brighter
  than the bright limit; they carry the 2MASS `ph_qual` letter (A-D) of the band, or X (bit 2048).

## Logging

Each run writes a verbose, timestamped log to `path_pype/<name>/temp/pipeline_<timestamp>.log` at DEBUG level, including captured external-tool output, so a run is reconstructable from the file alone. The console stays minimal: a live progress bar on a terminal plus real warnings and errors (no progress codes are written when output is redirected). Set `log_retention: <N>` to keep only the newest N per-run logs, or `file_log: false` to skip the per-run file and rely on stdout (for example in a container that redirects to a node log).

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
