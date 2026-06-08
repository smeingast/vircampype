# vircampype pipeline: ultra code review

_Generated 2026-06-08. Assessment from a literature-grounded multi-agent review of the science/calibration pipeline, with external (codex) double-checking of the high-severity findings. Fixed items are marked and listed at the bottom._

## Scope and method

Two passes: (1) reference gathering from the local Papers library plus bounded web search across six topics; (2) one reviewer per pipeline dimension reading the real code with that reference digest. Every finding was adversarially verified (correctness + data-product impact); the highs were re-checked by codex and by hand. Cluster batch subsystem and the recently-validated astrometric floor / RADECCORR work are out of scope.

**Counts:** 44 raw, 35 confirmed (no criticals). **5 fixed so far** (2 high + 3 medium, see bottom); **open:** 2 high, 7 medium, 19 low.

Severity: **critical** = silently wrong science survey-wide; **high** = significant error in a common regime; **medium** = wrong in edge cases; **low** = minor / hardening.

## Open - High severity

### H1. SCAMP solutions accepted on internal RMS only (no external-RMS / match-count gate)

- **Where:** vircampype/fits/tables/sextractor.py:213-216 (sole gate); messaging.py:281-326 (QC prints, never raises); tabletools.py:613 (NDeg_Reference read but unused as a gate)
- **What:** The only hard post-SCAMP check is `if np.max(xml["AstromSigma_Internal"]*1000) > 100: raise`. It uses only the internal (source-to-source) scatter, never the external/reference RMS vs Gaia, the matched-reference count, or NDeg_Reference. A degenerate / near-identity solution from too few matches has a SMALL internal sigma and passes. `np.max` is not nan-aware, so a no-overlap exposure (NaN) or all-zero sigma also passes. It reads the GROUP table (table_id=1), one aggregate over all exposures, so a single bad pawprint barely moves it.
- **Impact:** A sparse-match or internally-consistent-but-absolutely-wrong WCS passes the checkpoint, is cached as .ahead, and propagates into resampling / coadds / public catalog with no error. Bites sparse fields, single-pass regions, and edge exposures.
- **Reference:** Bertin 2006 (SCAMP): internal vs external RMS are distinct; external RMS catches bad absolute solutions.
- **Verification:** Workflow: confirmed. Codex: PARTIAL. Nuance: the fully-unconstrained case (missing ASTRMSH1/2) is now caught later by the build_statistics fail-loud (sky.py:2919-2932), so the gap is narrowed to sparse / degenerate-but-plausible / high-finite-external-RMS solutions.
- **Status:** OPEN. Suggested: add an external-RMS (AstromSigma_Reference) threshold + a minimum matched-reference / NDeg_Reference gate, per exposure, and make np.max nan-aware.

### H3. Published flux/mag errors have no correlated-noise inflation after LANCZOS3 resampling

- **Where:** vircampype/tools/tabletools.py:845-859 (magerr path); coadd.yml:57 (RESAMPLING_TYPE LANCZOS3) + RESCALE_WEIGHTS Y; sky.py:2129 (resample kernel)
- **What:** Tiles/stacks are LANCZOS3-resampled, which correlates neighbouring output pixels, so the per-pixel variance from the SWarp weight map and SExtractor's white-noise FLUXERR underestimates the true noise. The published error is `magerr_aper = sqrt(MAGERR_APER**2 + photerr_internal**2)`, with MAGERR_APER straight from SExtractor on the resampled coadd. No correlated-noise / RMS inflation factor exists anywhere in the flux-error path. RESCALE_WEIGHTS=Y compounds it in crowded fields.
- **Impact:** All MAG_APER/MAG_AUTO *_ERR (and Qflg thresholds keyed on magerr_best) are systematic lower limits, largest for aperture photometry and faint sources. Biases completeness/quality flags and any chi2 weighting. Survey-wide.
- **Reference:** resampling-coadd / extraction-catalog digest; SWarp & SExtractor manuals; correlated noise after Lanczos resampling under-estimates per-pixel errors.
- **Verification:** Workflow: confirmed. Codex: CONFIRM (only the fixed floor is added, no inflation).
- **Status:** OPEN. Suggested: derive and apply a correlated-noise inflation factor for the resampling kernel/pixel-scale (or measure empty-aperture RMS) before publishing errors; reconsider RESCALE_WEIGHTS in crowded fields.

## Open - Medium severity

### M1. Twilight flat masks saturation AFTER NDIT normalization, so the raw-ADU saturation threshold under-masks saturated/non-linear pixels when NDIT>1

- **Where:** vircampype/fits/images/flat.py:121-138
- **What:** In build_master_twilight_flat the cube is first divided to NDIT=1 (cube.normalize(files.ndit), line 126) and linearized (line 132), and only then is the saturation mask applied via cube.apply_masks(bpm=bpm, mask_above=sat) (line 138) where sat = setup.saturation_levels[d-1]. saturation_levels are absolute raw ADU (full NDIT-summed counts, ~33000-36000). After dividing by NDIT the pixel values are at the per-DIT level, so for NDIT>1 the comparison value > sat is never reached for pixels that were saturated in the raw frame (raw_value/NDIT < sat), and saturated/non-linear pixels leak into the master flat. This is inconsistent with build_master_linearity (flat.py:466) and build_master_gain (flat.py:912-918), which both compare saturation against the RAW (non-normalized) flux, the correct convention.
- **Impact:** For any twilight-flat sequence acquired with NDIT>1, saturated and strongly non-linear pixels are not rejected from the master flat, biasing the per-pixel flat response (and thus the inter-detector/illumination flat normalization) in the affected regions. No effect when twilight flats are NDIT=1 (the usual VIRCAM case), so it is an edge-case defect rather than survey-wide.

### M2. '--reset progress' does not actually re-run from scratch because overwrite defaults to False

- **Where:** vircampype/pipeline/worker.py:56-67,185-187; vircampype/pipeline/setup.py:38 (overwrite=False)
- **What:** The --reset progress help string promises it 'clears checkpoint state so the pipeline re-runs from scratch' (worker.py:59-61). Implementation only does clean_directory(folders['temp']) (worker.py:186), which deletes pipeline_status.p. With the cleared status, process_science walks every step again, but each step's per-file guard is `check_file_exists(...) and not self.setup.overwrite` (e.g. apply_illumination_correction sky.py:619-623, resample sky.py:2162-2166, build_stacks sky.py:2286-2290, process_one_basic sky.py:744-748). Since overwrite defaults to False (setup.py:38), every already-existing product is SKIPPED. So the prior, possibly-buggy products are silently kept and merely re-validated; the headers are re-read (matching the existing files), and only genuinely-missing files get produced. The user-visible contract ('re-runs from scratch') is therefore false unless overwrite=True is also passed.
- **Impact:** An operator trying to recover from a bad reduction by resetting progress gets the OLD products back, believing they were regenerated. Corrupt intermediate products survive the reset across the survey. Compounds the stale-header-cache finding.

### M3. SCAMP cache-skip marks astrometry complete without verifying the cached .ahead carry the high-S/N RMS floor

- **Where:** vircampype/fits/tables/sextractor.py:124-146 (cache restore + early return); vircampype/pipeline/main.py:25-56 (pipeline_step sets flag on return), 709-723 (calibrate_astrometry)
- **What:** scamp() restores cached .ahead files from scamp_cache_dir (sextractor.py:126-137) and, if all .ahead headers are then present, returns immediately (sextractor.py:143-146) WITHOUT running the XML QC, without injecting ASTCORR, and crucially without checking that the cached aheaders contain ASTRMSH1/2. The wrapping pipeline_step decorator (main.py:48-54) sets status.astrometry=True on any non-exception return. If the cache predates the high-S/N floor work (the very 'stale .ahead cache' the build_statistics comment at sky.py:2914-2915 warns about), astrometry is marked done, illumination_correction/resample copy those aheaders forward, and the run proceeds until build_statistics raises PipelineValueError on the missing ASTRMSH1/2 (sky.py:2919-2932). That is fail-loud (good), but the failure surfaces many expensive steps later (after IC, resampling) rather than at the astrometry checkpoint, and the astrometry checkpoint is left set True, so a naive re-run skips SCAMP again and re-hits the same late failure.
- **Impact:** A stale astrometric cache wastes the IC+resample compute and then dead-ends at statistics, with the astrometry checkpoint stuck True so the loop does not self-heal. Bites only on reductions reusing an old scamp_cache_dir.

### M4. Zero-point uncertainty is never propagated into the published per-source MAGERR

- **Where:** vircampype/tools/tabletools.py:847; vircampype/fits/tables/sextractor.py:902-918
- **What:** calibrate_photometry derives zeropoint_err for each magnitude column (sextractor.py:890-902) but writes it ONLY to FITS header keywords (HIERARCH PYPE ZP ERR ...; sextractor.py:915,918). It is never carried as a per-source column and never combined into the calibrated magnitude error. In convert2public the published catalog error is magerr_aper = sqrt(data_magerr_aper**2 + photerr_internal**2) (tabletools.py:847) and then magerr_best is selected from that; the ZP uncertainty term is absent. Note the ZP error itself is also computed loosely (photometry.py:132 uses scipy.stats.sem over the *NaN-masked* diff array, but the sigma-clip-outlier mask from photometry.py:119 was reassigned at line 125 and is not applied to the SEM input, so even the header ZP error includes clipped outliers).
- **Impact:** Published MAGERR is a lower bound: it omits the systematic ZP-transfer uncertainty (2MASS calibration error, per-tile/per-detector ZP scatter). For bright, high-S/N sources whose Poisson+aperture error is small, the catalog error is dominated by, and missing, the ZP floor, so reported errors understate the true magnitude uncertainty. The internal-photometric floor only partially compensates and is a single survey-wide constant.

### M5. Per-tile ZP fit runs on the full (uncleaned) source table with no FLAG=0 / own-error cut

- **Where:** vircampype/fits/tables/sextractor.py:884-899
- **What:** The definitive zero points (MAG_APER, MAG_APER_MATCHED, MAG_AUTO) are fit with get_zeropoint using mag1=table[cmag].data on the FULL table, not table_clean. clean_source_table was already run (sextractor.py:737) and produces clean_idx / flux limits (lines 727-745), but those cuts (FLAGS in {0,2}, SNR>=10, FWHM/ellipticity/edge, flux_auto_min/max) are applied only to the apcor-interpolation reference (table_clean), not to the ZP-fitting sample. The only protection is the reference-side magnitude window (mag_limits_ref, e.g. J [12,15.5]) plus the internal 2.5-sigma clip (photometry.py:119). Saturated/blended/poorly-shaped science sources whose 2MASS counterpart happens to fall in the band-limited window can still enter the solution; there is no explicit SExtractor FLAG=0 nor own-catalog MAGERR<0.1 cut on the fitting list.
- **Impact:** ZP can be pulled by contaminated bright/blended sources in crowded VISIONS star-forming fields (exactly where 2MASS bright stars cluster), biasing the per-tile zero point and hence every calibrated magnitude on that tile. Mitigated, but not eliminated, by the reference magnitude window and sigma-clip; magnitude of bias depends on contamination fraction inside the window.
- **Decision (2026-06-08):** DEFERRED to a later version per Stefan. A fix (fit on the existing clean_idx sample, fall back to the full table) was drafted and reverted; the existing safeguards (bright reference window + 1/err^2 weighting + 2.5-sigma clip) are considered sufficient for now. Quick check before reviving: re-derive the ZP on a crowded tile using only FLAGS == 0 and compare to the shipped ZP.

### M6. Catalog MAGERR omits the aperture-correction interpolation uncertainty and correlated-noise inflation

- **Where:** vircampype/tools/tabletools.py:847,859; vircampype/fits/tables/sextractor.py:868,854-857
- **What:** The published aperture magnitude is `MAG_APER_MATCHED_CAL` = MAG_APER + interpolated growth correction MAG_APER_COR_INTERP (sextractor.py:868). The published error is `magerr_aper = sqrt(MAGERR_APER**2 + photerr_internal**2)` (tabletools.py:847) and `magerr_best` is selected at the same aperture index. This propagates only the raw SExtractor aperture error plus the flat internal-error floor; it never adds the spatial-interpolation scatter of the aperture correction, which the pipeline already computes and stores as `MAG_APER_COR_INTERP_STD` (sextractor.py:855) and then discards for error purposes. It also applies no correlated-noise inflation factor for the LANCZOS-resampled tiles. The SExtractor MAGERR_APER itself is the white-noise estimate that the manual states underestimates errors on resampled images.
- **Impact:** Published MAGERR is a lower bound on the true magnitude uncertainty, most noticeably where the aperture-correction field varies spatially (FWHM gradients across the focal plane) and in resampled coadds where pixel-to-pixel covariance is ignored. Downstream weighting, variability detection, and QFLG assignment (which is keyed directly on magerr_best thresholds, lines 947-949) inherit the optimistic errors.

### M7. Per-plane sky weights 1/bkg_std are not guarded against mmm failure values (sigma = -1.0 or NaN), silently corrupting the weighted master sky

- **Where:** vircampype/fits/images/sky.py:91-102 (weights), with mmm failure returns at vircampype/external/mmm.py:203 (sigma=-1.0), 305 (sigma=np.nan)
- **What:** In _build_sky_detector the weighted-combine path builds weights = (1/bkg_std)[:,None,None] where bkg_std is the per-plane mmm sigma returned by ImageCube.background_planes(). mmm can return sigma = -1.0 (mmm.py:202-211, 'too few valid sky elements') or sigma = NaN (mmm.py:304-309, 'outlier rejection left too few elements') while still returning a finite skymod, so bkg is finite and the plane is kept. The only weight sanitisation is weights[~np.isfinite(cube.cube)] = 0.0 (line 96), which zeroes weights at pixels masked in the CUBE, not where the WEIGHT itself is bad. A plane with sigma=-1.0 therefore enters np.ma.average (cube.py:858) with a uniform NEGATIVE weight, biasing or even sign-flipping the combined sky on every pixel that frame contributes to; a plane with sigma=NaN propagates NaN into every unmasked pixel of that detector's master sky (np.ma.average does not treat NaN weights as masked). The negative-weight case is fully silent (no NaN, no exception); the NaN case eventually trips the min_valid_pixel_fraction guard in process_raw_final and aborts.
- **Impact:** Corrupted master-sky flat/shape for affected detector-groups. Negative-weight planes silently bias the sky used as a flat (sky mode, cube /= sky_norm) or as the subtracted sky shape (twilight mode, cube -= sky_norm*sky_level), producing per-detector additive/multiplicative sky errors. Bites whenever an input sky frame has a heavily masked plane (deep source masks in crowded/nebular fields, offset frames over a bright region, or a dead/low-signal detector) so that mmm's acceptance band drops below minsky. Most damaging in the default sky_combine_metric='weighted' configuration.

## Open - Low severity / hardening

1. **astrometry** `vircampype/tools/fitstools.py:776-809, vircampype/fits/images/sky.py:1421` - Reference catalog is pre-propagated to the mean epoch but its position errors are not inflated for the propagation baseline  
   Reference position uncertainties fed to SCAMP are slightly under-stated for high-PM stars over multi-year baselines (a few mas for typical EDR3 PM errors, larger for the highest-PM calibrators), giving them marginally too much weight in the astrometric fit and a small bias in the per-exposure distortion solution. Second-order vs the ~70 mas external RMS, but systematic.

2. **astrometry** `vircampype/tools/astromatic.py:82` - read_aheaders splits multi-extension headers on the substring 'END', not the END card  
   No effect on current canonical SCAMP output. Latent corruption of per-detector WCS assignment if any 'END'-containing card is ever added to the aheaders (e.g. a provenance comment), which would silently shift distortion/CRVAL to the wrong detector.

3. **astrometry** `vircampype/tools/wcstools.py:334-338 (CRPIX1/2 = CRPIX * tfactor); used by build_statistics (sky.py:2884-2892)` - resize_header scales CRPIX by the bin factor without the half-pixel binning offset, shifting statistics maps ~0.5 arcsec vs the data  
   Per-pixel statistics (MJD, NIMG, EXPTIME, astrometric RMS floor) are registered ~0.5 arcsec off from the actual science flux. Because these maps are nearly piecewise-constant per detector, the effect is confined to coverage/detector boundaries, so impact on most sources is negligible; it can mis-assign NIMG/EXPTIME/ASTRMS to sources right at edges.

4. **calibration** `vircampype/fits/images/flat.py:709-744` - MASTER-BPM subtracts a time-matched dark with no DIT/NDIT (or linearity) matching  
   If a wrong-DIT/NDIT dark is the temporally-closest one, the subtracted offset shifts the normalization baseline and can slightly over/under-flag pixels near the bpm_rel_threshold boundary. Low because the median normalization absorbs most of the constant offset and BPM is a coarse mask.

5. **calibration** `vircampype/fits/images/flat.py:141-215` - Low-flux twilight planes are NaN-masked in the cube but still counted in the inter-detector global scale factors  
   Mild bias in the per-detector global flat scale / flat_error when a discarded low-flux plane is retained in the flux statistics. Usually negligible because flat_min_flux removes genuinely bad exposures and several good planes dominate the mean, but it is a non-robust averaging path feeding the inter-detector ZP equalization.

6. **catalogs** `vircampype/tools/tabletools.py:1242-1255` - 2MASS replacement weight check indexes weight map with unclipped pixel coordinates inside a bare try/except  
   A small number of 2MASS bright-star replacements near image edges can be accepted or rejected based on a wrapped-around weight value from the opposite side of the tile, occasionally inserting a 2MASS source into a zero-coverage edge region or dropping a valid one. Edge-only, low source count.

7. **extraction-catalog** `vircampype/fits/images/sky.py:337 (seeing_fwhm=2.5 in 'full' branch); full.param:11 emits CLASS_STAR; full.yml:66 PIXEL_SCALE 0 (uses WCS)` - Science 'full' SExtractor preset hardcodes SEEING_FWHM=2.5 arcsec, biasing its CLASS_STAR on sub-arcsecond data  
   If any downstream consumer uses the 'full' catalog CLASS_STAR (rather than the multi-seeing library columns), star-galaxy separation is biased toward the wrong class, particularly for compact sources. Severity depends on which CLASS_STAR is published.

8. **numerical** `vircampype/external/mmm.py:134-138; callers cube.py:1425/1430-1440 (highbad omitted/False) and imagetools.py:335,706 (mmm(t), mmm(v,minsky=10))` - mmm highbad branch indexes a scalar and would crash, but is dead because the pipeline never passes highbad  
   No current data impact (dead code). Becomes an immediate hard crash if anyone enables highbad to exclude saturated/CR pixels from sky/mode estimation. Also documents that saturated pixels are currently NOT rejected from mmm sky estimates, relying entirely on the upstream source/BPM masks.

9. **numerical** `vircampype/data/cube.py:1040-1051 (destripe)` - Destripe bad-row/bad-plane thresholds use the wrong array axis length (n_rows where n_cols is intended), correct only because VIRCAM detectors are square  
   No effect on current square-detector data. Latent: silently wrong bad-row/bad-plane gating (hence wrong destripe additive correction) for any non-2048-square geometry. Same square-detector fragility class noted elsewhere in the codebase.

10. **numerical** `vircampype/tools/photometry.py:120-129` - get_zeropoint weighted ZP does not guard against NaN reference (2MASS) errors in the weights  
   A single NaN 2MASS error among the matched calibrators silently NaNs the zero point (and thus all calibrated magnitudes) for an aperture/extension. In practice rare because Qflg='A' sources carry valid errors, but there is no defensive masking, so any future widening of allowed Qflg/Cflg or a masked-error edge case poisons the calibration without an error being raised.

11. **numerical** `vircampype/tools/mathtools.py:433-436` - Linearity inversion reconstructs value as |root-data|+data, which silently folds any downward correction  
   Dormant in the unsaturated regime where the non-linearity correction is always positive (true counts > measured). Bites only if a fitted root ever falls below the measured ADU (pathological coefficients, very low counts near the intercept, or fit noise), where it would apply a correction of the wrong sign to those pixels. Latent rather than survey-wide because real linearity corrections are upward.

12. **numerical** `vircampype/tools/imagetools.py:176-205, 246-281` - chop_image/merge_chopped break when a chop segment is smaller than the overlap (zero/over-width pieces)  
   Hard crash (not silent corruption) of interpolate_nan for a science frame when n_jobs is high enough that 2048/(2*n_jobs) <= overlap, aborting that frame's processing. Edge-case robustness bug, no wrong-science risk because it fails loudly.

13. **orchestration** `vircampype/fits/images/common.py:797-808 (no else branch); vircampype/fits/images/sky.py:2147-2148 (resample), 2792-2793 area uses get_master_weight_global which DOES raise` - get_master_weight_image returns None when not all weights are present, causing a late AttributeError instead of a clear diagnostic  
   Missing per-image weight maps surface as a confusing NoneType crash in resampling instead of a clear 'not all images have weights' message at the matching step; harder to diagnose, and the inconsistency with get_master_weight_global is a latent footgun.

14. **photometry** `vircampype/tools/tabletools.py:847` - Catalog magnitude errors carry no correlated-noise inflation after LANCZOS resampling _(same root as H3)_  
   Published per-source flux/magnitude errors are systematically too small (typically tens of percent for aperture photometry on Lanczos-resampled coadds), most relevant for faint sources near the detection limit and for color/variability work. Partly absorbed by the empirical internal-photometric floor, which is why this is low rather than higher.

15. **photometry** `vircampype/tools/esotools.py:1107-1112` - Phase3 ESO catalog writer takes sqrt of MAGERR (and reads a nonexistent ZPC column)  
   If ESO Phase3 export is enabled, published Phase3 magnitude errors would be the square root of the real errors (grossly wrong) - but the missing _ZPC_INTERP columns mean the function currently raises before that, so it is effectively dead/broken rather than silently corrupting. Flagged so it is fixed before Phase3 is turned on.

16. **photometry** `vircampype/tools/imagetools.py:519-531` - grid_value_2d weighted branch averages the sigma-clip REJECTED points instead of the kept ones  
   If grid_value_2d's weighted mode were ever wired into the illumination-correction / ZP-grid path it would build the grid from rejected outliers, producing grossly wrong per-pixel zero points and flux scales. As shipped it is a latent landmine (unused), not an active corruption.

17. **resampling** `vircampype/fits/tables/sextractor.py:1708 (bad &= weight_sources < 0.0001), in context of 1677-1712` - Per-source statistics (MJDEFF/EXPTIME/NIMG/ASTRMS) are NOT masked on zero-coverage pixels due to an AND-instead-of-OR bug  
   Sources in low/zero-coverage regions (chip gaps, edges, masked stripes) get corrupted MJDEFF, EXPTIME, NIMG=0/garbage, and corrupted ASTRMS1/2 position-error floors instead of nan/0. ASTRMS feeds the published RA/Dec position errors and NIMG/EXPTIME feed depth bookkeeping, so the affected (minority) sources carry silently wrong metadata in the public catalog. Bites the survey wherever coverage holes overlap detections.

18. **sky** `vircampype/fits/images/sky.py:1484-1512 (split_window + mjd_mean naming); vircampype/fits/common.py:526-576 (split_window); vircampype/fits/images/common.py:740-756 + common.py:620-668 (get_master_sky / match_mjd)` - Master-sky time window is built per-frame then deduplicated, but each science frame is matched to the nearest master by mjd_mean, so a frame can receive a sky window not centered on (or not containing) it  
   Sub-optimal temporal sky matching at sequence boundaries and on sparse nights: residual additive sky or slightly stale sky shape on a minority of frames. Low impact under dense default observing; grows if sky_window is shortened or cadence is irregular.

19. **sky** `vircampype/tools/imagetools.py:361, 367-398` - upscale_image(new_size=image.shape) transposes the result for non-square inputs in background_image  
   No effect for square VIRCAM detectors (the only shape that reaches background_image). Would silently transpose the background/noise map (corrupting sky subtraction structure) the moment a non-square array is passed, e.g. a future change to mesh geometry or a non-square cutout.

## Fixed in this review cycle

- **[High] Public ELLIPTICITY computed from the position-error ellipse, not the source shape** (`e0a10024`)  
  vircampype/tools/tabletools.py:775, 988 - Now uses the windowed source shape 1 - BWIN_WORLD/AWIN_WORLD; keep-filter relaxed to >= 0 so round sources survive. Regenerate the .ptab only.

- **[High] Header shelve cache keyed by basename was never invalidated on regeneration** (`d17b42d0`)  
  vircampype/fits/common.py:165-195 - Entries now carry a (size, mtime_ns) signature validated on read; a regenerated file misses the cache and is re-read. Old bare-list entries upgrade transparently.

- **[Medium] MASTER-GAIN divided by a possibly non-positive variance difference** (`733c9cb7`)  
  vircampype/fits/images/flat.py:921-932 - Detectors with var(flat_diff) - var(dark_diff) <= 0 are set to NaN (with a logged count) instead of yielding a negative/inf GAIN fed to SExtractor.

- **[Medium] comp90/comp50 returned even when the completeness curve never crosses the level** (`5f974ef9`)  
  vircampype/tools/completeness.py:333-334 and 2 sibling sites - Return NaN when the level lies outside the fitted curve's range, instead of the closest-approach magnitude.

- **[Medium] Pipeline.stacks hard-coded exactly 6 offset positions** (`6c7cc232`)  
  vircampype/pipeline/main.py:383 - Compare against the configurable setup.n_offset_positions; fields with a non-default offset count no longer abort the stacks branch.

