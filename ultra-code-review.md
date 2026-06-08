# vircampype pipeline: ultra code review

_Generated 2026-06-08. Assessment only; no behavioural change except the H4 fix noted below._

## Scope and method

Literature-grounded multi-agent review of the **science/calibration pipeline** (not the cluster batch subsystem, reviewed separately). Two passes: (1) reference gathering from the local Papers library plus bounded web search across six topics (VISTA/VIRCAM reduction, flat/sky, astrometry, photometry, resampling/coaddition, extraction/catalogues); (2) one reviewer per pipeline dimension, given the reference digest and reading the real code. Every finding was adversarially verified inside the workflow (correctness + data-product impact). The four high-severity findings were then independently double-checked by an external reviewer (codex) and re-checked by hand against the code.

**Counts:** 44 raw findings, 35 confirmed. No criticals. 4 distinct high (one was found by two dimensions), 10 medium (one medium folded into H1), 19 low. Excluded by design (intentional, recently validated): the high-S/N astrometric position-error floor (ASTRMSH1/2) and the per-field SCAMP RADECCORR fix.

Severity: **critical** = silently wrong science across the survey; **high** = significant error in a common regime; **medium** = wrong in edge cases; **low** = minor / hardening.

## High severity

### H1. SCAMP solutions accepted on internal RMS only (no external-RMS / match-count gate)

- **Where:** vircampype/fits/tables/sextractor.py:213-216 (sole gate); messaging.py:281-326 (QC prints, never raises); tabletools.py:613 (NDeg_Reference read but unused as a gate)
- **What:** The only hard post-SCAMP check is `if np.max(xml["AstromSigma_Internal"]*1000) > 100: raise`. It uses only the internal (source-to-source) scatter, never the external/reference RMS vs Gaia, the matched-reference count, or NDeg_Reference. A degenerate / near-identity solution from too few matches has a SMALL internal sigma and passes. `np.max` is not nan-aware, so a no-overlap exposure (NaN) or all-zero sigma also passes. It reads the GROUP table (table_id=1), i.e. one aggregate over all exposures, so a single bad pawprint barely moves it.
- **Impact:** A sparse-match or internally-consistent-but-absolutely-wrong WCS passes the checkpoint, is cached as .ahead, and propagates into resampling / coadds / public catalog with no error. Bites sparse fields, single-pass regions, and edge exposures.
- **Reference:** Bertin 2006 (SCAMP): internal vs external RMS are distinct; external RMS catches bad absolute solutions. Astrometry digest pitfall: 'SCAMP succeeds with a degenerate solution.'
- **Verification:** Workflow: confirmed. Codex: PARTIAL. Codex's correct nuance: the FULLY-unconstrained case (missing ASTRMSH1/2) is now caught later by the build_statistics fail-loud (sky.py:2919-2932), so the gap is narrowed to sparse / degenerate-but-plausible / high-finite-external-RMS solutions, not the empty case. Also folds in the medium duplicate at sextractor.py:213-216 (too-few-matches / NDeg gate).
- **Status:** OPEN. Suggested: add an external-RMS (AstromSigma_Reference) threshold + a minimum matched-reference / NDeg_Reference gate, per exposure, and make np.max nan-aware (treat NaN as failure).

### H2. Public ELLIPTICITY is computed from the position-ERROR ellipse, not the source shape

- **Where:** vircampype/tools/tabletools.py:741-742, 775 (definition); :981 (survey keep-filter ellipticity > 0); :1028/1052 (published column)
- **What:** convert2public sets `data_erra = ERRAWIN_WORLD`, `data_errb = ERRBWIN_WORLD`, then `data_ellipticity = 1 - data_errb/data_erra`. ERRAWIN/ERRBWIN are the SExtractor WINDOWED POSITIONAL-ERROR ellipse semi-axes (astrometric centroid uncertainty), NOT the light-profile shape. The true shape (AWIN_WORLD/BWIN_WORLD, and a ready-made ELLIPTICITY = 1 - B_IMAGE/A_IMAGE) is emitted by the 'full' preset (full.param:26-34) and ignored. Because errb <= erra always, the value sits in [0,1] and looks plausible, so the error is silent.
- **Impact:** Every public-catalog row's ELLIPTICITY reflects centroid error-ellipse anisotropy (driven by S/N and PSF sampling), not morphology. Any morphological / extended-source selection or shape QC on the public catalog uses a meaningless quantity. It is also used in the survey keep-filter, so it affects which rows are published. Survey-wide.
- **Reference:** SExtractor manual: ELLIPTICITY = 1 - B_IMAGE/A_IMAGE (shape) vs ERRAWIN/ERRBWIN (positional-error ellipse).
- **Verification:** Workflow: confirmed (found independently by two dimensions). Codex: CONFIRM, with no later substitution of the shape column. Verified directly against the code and full.param.
- **Status:** OPEN. Fix is a science choice (which shape column to publish); see the plain-English breakdown provided separately. NOT changed in this pass per request.

### H3. Published flux/mag errors have no correlated-noise inflation after LANCZOS3 resampling

- **Where:** vircampype/tools/tabletools.py:845-859 (magerr path); coadd.yml:57 (RESAMPLING_TYPE LANCZOS3) + RESCALE_WEIGHTS Y; sky.py:2129 (resample kernel); full preset SExtractor run on the resampled coadd
- **What:** Tiles/stacks are LANCZOS3-resampled, which correlates neighbouring output pixels, so the per-pixel variance from the SWarp weight map and SExtractor's white-noise FLUXERR underestimates the true noise. The published error is `magerr_aper = sqrt(MAGERR_APER**2 + photerr_internal**2)`, where MAGERR_APER comes straight from SExtractor on the resampled coadd. No correlated-noise / RMS inflation factor exists anywhere in the flux-error path (the only covariance code is the astrometric position-error ellipse). RESCALE_WEIGHTS=Y compounds the bias in crowded NIR fields.
- **Impact:** All MAG_APER/MAG_AUTO *_ERR (and Qflg thresholds keyed on magerr_best, tabletools.py:947) are systematic lower limits, largest for aperture photometry and faint sources where resampling covariance dominates. Biases completeness/quality flags and any chi2 weighting downstream. Survey-wide.
- **Reference:** resampling-coadd / extraction-catalog digest; SWarp & SExtractor manuals; correlated-noise after Lanczos resampling is a known under-estimation of per-pixel errors.
- **Verification:** Workflow: confirmed. Codex: CONFIRM (only the fixed floor is added, no inflation). Folds in the low-tier duplicate at tabletools.py:847 and overlaps medium M6 (aperture-correction-interpolation scatter).
- **Status:** OPEN. Suggested: derive and apply a correlated-noise inflation factor for the resampling kernel/pixel-scale (or measure empty-aperture RMS) before publishing errors; reconsider RESCALE_WEIGHTS in crowded fields.

### H4. Header shelve cache keyed by basename was never invalidated when a product is regenerated

- **Where:** vircampype/fits/common.py:165-195 (headers property); dead delete_headers at :336; only commented call at sky.py:2577
- **What:** FitsFiles.headers cached every HDU header in a persistent shelve keyed PURELY on the file basename, in local_cache_dir/tempdir, with no mtime/size/content check, persisting across runs and instances. The only invalidation method (delete_headers) was dead code. Regenerating a product under the same basename (overwrite=True, a SCAMP re-solve changing .ahead keywords like GAIN/MJD/ASTRMSH/ASTCORR, or a replaced raw) returned the STALE header; '--reset progress' did not clear it (only '--reset cache' did).
- **Impact:** On any overwrite/regeneration path, stale per-detector GAIN/SATURATE/MJD/astrometric-RMS/WCS were used, silently corrupting Poisson error terms, statistics maps, and the position-error floor. Directly relevant to reprocessing campaigns (matches the 'must clear the shelve on alcyone' operational note).
- **Reference:** Header-cache staleness hazard: a same key must not map to changed bytes without invalidation.
- **Verification:** Workflow: confirmed. Codex: CONFIRM (basename-only key; delete_headers dead; only --reset cache clears it). Verified directly.
- **Status:** FIXED in this change. Cache entries now store a (size, mtime_ns) signature and are validated on read; a regenerated file misses the cache and is re-read. Old bare-list entries upgrade transparently. Verified: unchanged file stays cached, rewritten file is re-read, suite green (256 tests).

## Medium severity

### M1. MASTER-GAIN: no guard against (fvar - dvar) <= 0, producing negative/inf/NaN gain fed to SExtractor survey-wide

- **Where:** vircampype/fits/images/flat.py:921-932
- **What:** build_master_gain computes gain = ((mf0+mf1)-(md0+md1)) / (fvar - dvar) with no check that the denominator is positive. fvar=(f0-f1).var, dvar=(d0-d1).var. Whenever the flat-pair difference variance does not exceed the dark-pair difference variance (low-illumination or near-equal-level pairs, a noisy/striped detector channel, or a partially saturated flat whose variance is suppressed), fvar-dvar is <= 0, so gain goes negative or, at fvar==dvar, becomes +/-inf. The value is written verbatim into the MASTER-GAIN table (no clamp, no NaN/positivity filter), read back unvalidated in fits/tables/gain.py:18, and at sky.py:815 becomes the per-detector SExtractor GAIN (gain*NDIT) for every science frame matched to that table. rdnoise = gain*sqrt(...) inherits the same sign error. Additionally the gain uses only the first two MJD-sorted frames (file_index 0 and 1) with no check that f0 and f1 share the same illumination level; any lamp/twilight drift inflates var(f0-f1) and biases the gain. Janesick's method (digest: flat-sky gain best-practice) explicitly requires guarding var(F1-F2)-var(D1-D2) <= 0.
- **Impact:** A single detector/epoch with a degenerate variance difference silently propagates a negative/inf/NaN GAIN into SExtractor for all science frames using that gain table, corrupting the Poisson (FLUX/GAIN) term of FLUXERR/MAGERR for bright sources on that detector (errors NaN, inf, or nonsensically small/negative). Bites in any epoch with weak/poorly-matched gain-flat pairs or a striped channel; the error is silent because nothing validates the gain table before or after use.

### M2. Twilight flat masks saturation AFTER NDIT normalization, so the raw-ADU saturation threshold under-masks saturated/non-linear pixels when NDIT>1

- **Where:** vircampype/fits/images/flat.py:121-138
- **What:** In build_master_twilight_flat the cube is first divided to NDIT=1 (cube.normalize(files.ndit), line 126) and linearized (line 132), and only then is the saturation mask applied via cube.apply_masks(bpm=bpm, mask_above=sat) (line 138) where sat = setup.saturation_levels[d-1]. saturation_levels are absolute raw ADU (full NDIT-summed counts, ~33000-36000). After dividing by NDIT the pixel values are at the per-DIT level, so for NDIT>1 the comparison value > sat is never reached for pixels that were saturated in the raw frame (raw_value/NDIT < sat), and saturated/non-linear pixels leak into the master flat. This is inconsistent with build_master_linearity (flat.py:466) and build_master_gain (flat.py:912-918), which both compare saturation against the RAW (non-normalized) flux, the correct convention.
- **Impact:** For any twilight-flat sequence acquired with NDIT>1, saturated and strongly non-linear pixels are not rejected from the master flat, biasing the per-pixel flat response (and thus the inter-detector/illumination flat normalization) in the affected regions. No effect when twilight flats are NDIT=1 (the usual VIRCAM case), so it is an edge-case defect rather than survey-wide.

### M3. comp90/comp50 returned even when the completeness curve never crosses 90%/50%

- **Where:** vircampype/tools/completeness.py:333-334 (also 955-957, 1323-1324)
- **What:** After the logistic fit, the limiting magnitudes are taken as `comp90 = x_fine[np.argmin(np.abs(y_fine - 90))]` and likewise for 50%. argmin-of-absolute-difference always returns a magnitude even when the modelled completeness never actually reaches 90% (shallow/crowded sub-tile saturating below 90%) or stays above it across the whole range. In the never-reaches case it silently returns the magnitude of closest approach (typically the bright edge of the fit range); there is no NaN guard checking that the curve truly crosses the level. The same pattern is duplicated in plot_completeness_tile and save_completeness_results.
- **Impact:** Per-sub-tile and tile-summary 90%/50% completeness limits (and the FITS COMP90/COMP50 header keywords written to the tile and the completeness map image) can be silently nonsensical in crowded star-forming regions, exactly where VISIONS completeness varies most. The reported survey depth in those regions is unreliable rather than flagged as undefined.

### M4. '--reset progress' does not actually re-run from scratch because overwrite defaults to False

- **Where:** vircampype/pipeline/worker.py:56-67,185-187; vircampype/pipeline/setup.py:38 (overwrite=False)
- **What:** The --reset progress help string promises it 'clears checkpoint state so the pipeline re-runs from scratch' (worker.py:59-61). Implementation only does clean_directory(folders['temp']) (worker.py:186), which deletes pipeline_status.p. With the cleared status, process_science walks every step again, but each step's per-file guard is `check_file_exists(...) and not self.setup.overwrite` (e.g. apply_illumination_correction sky.py:619-623, resample sky.py:2162-2166, build_stacks sky.py:2286-2290, process_one_basic sky.py:744-748). Since overwrite defaults to False (setup.py:38), every already-existing product is SKIPPED. So the prior, possibly-buggy products are silently kept and merely re-validated; the headers are re-read (matching the existing files), and only genuinely-missing files get produced. The user-visible contract ('re-runs from scratch') is therefore false unless overwrite=True is also passed.
- **Impact:** An operator trying to recover from a bad reduction by resetting progress gets the OLD products back, believing they were regenerated. Corrupt intermediate products survive the reset across the survey. Compounds the stale-header-cache finding.

### M5. Pipeline.stacks hard-codes exactly 6 offset positions, contradicting the configurable n_offset_positions

- **Where:** vircampype/pipeline/main.py:376-386 (len(images) != 6); vircampype/pipeline/setup.py:293 (n_offset_positions: int = 6); vircampype/fits/images/sky.py:2260-2263
- **What:** build_stacks correctly validates the number of stacks against the configurable setup.n_offset_positions (sky.py:2260). But the Pipeline.stacks property (main.py:383) hard-codes `if len(images) != 6: raise PipelineValueError("Stacks incomplete")`. n_offset_positions is a real Setup parameter (setup.py:293) that defaults to 6 but can be set per field. For any survey field configured with a number of offset positions other than 6, build_stacks would produce that many stacks, but every subsequent access to pipeline.stacks (photometry_stacks, build_statistics_stacks via resampled_statistics, classification_stacks, QC steps, build_phase3, build_qc_summary) raises 'Stacks incomplete' and aborts.
- **Impact:** Silent config trap: any field with non-default offset count cannot complete the stack branch (photometry/QC/phase3) even though the stacks were built correctly. The consistency check is wrong, not the data.

### M6. SCAMP cache-skip marks astrometry complete without verifying the cached .ahead carry the high-S/N RMS floor

- **Where:** vircampype/fits/tables/sextractor.py:124-146 (cache restore + early return); vircampype/pipeline/main.py:25-56 (pipeline_step sets flag on return), 709-723 (calibrate_astrometry)
- **What:** scamp() restores cached .ahead files from scamp_cache_dir (sextractor.py:126-137) and, if all .ahead headers are then present, returns immediately (sextractor.py:143-146) WITHOUT running the XML QC, without injecting ASTCORR, and crucially without checking that the cached aheaders contain ASTRMSH1/2. The wrapping pipeline_step decorator (main.py:48-54) sets status.astrometry=True on any non-exception return. If the cache predates the high-S/N floor work (the very 'stale .ahead cache' the build_statistics comment at sky.py:2914-2915 warns about), astrometry is marked done, illumination_correction/resample copy those aheaders forward, and the run proceeds until build_statistics raises PipelineValueError on the missing ASTRMSH1/2 (sky.py:2919-2932). That is fail-loud (good), but the failure surfaces many expensive steps later (after IC, resampling) rather than at the astrometry checkpoint, and the astrometry checkpoint is left set True, so a naive re-run skips SCAMP again and re-hits the same late failure.
- **Impact:** A stale astrometric cache wastes the IC+resample compute and then dead-ends at statistics, with the astrometry checkpoint stuck True so the loop does not self-heal. Bites only on reductions reusing an old scamp_cache_dir.

### M7. Zero-point uncertainty is never propagated into the published per-source MAGERR

- **Where:** vircampype/tools/tabletools.py:847; vircampype/fits/tables/sextractor.py:902-918
- **What:** calibrate_photometry derives zeropoint_err for each magnitude column (sextractor.py:890-902) but writes it ONLY to FITS header keywords (HIERARCH PYPE ZP ERR ...; sextractor.py:915,918). It is never carried as a per-source column and never combined into the calibrated magnitude error. In convert2public the published catalog error is magerr_aper = sqrt(data_magerr_aper**2 + photerr_internal**2) (tabletools.py:847) and then magerr_best is selected from that; the ZP uncertainty term is absent. Note the ZP error itself is also computed loosely (photometry.py:132 uses scipy.stats.sem over the *NaN-masked* diff array, but the sigma-clip-outlier mask from photometry.py:119 was reassigned at line 125 and is not applied to the SEM input, so even the header ZP error includes clipped outliers).
- **Impact:** Published MAGERR is a lower bound: it omits the systematic ZP-transfer uncertainty (2MASS calibration error, per-tile/per-detector ZP scatter). For bright, high-S/N sources whose Poisson+aperture error is small, the catalog error is dominated by, and missing, the ZP floor, so reported errors understate the true magnitude uncertainty. The internal-photometric floor only partially compensates and is a single survey-wide constant.

### M8. Per-tile ZP fit runs on the full (uncleaned) source table with no FLAG=0 / own-error cut

- **Where:** vircampype/fits/tables/sextractor.py:884-899
- **What:** The definitive zero points (MAG_APER, MAG_APER_MATCHED, MAG_AUTO) are fit with get_zeropoint using mag1=table[cmag].data on the FULL table, not table_clean. clean_source_table was already run (sextractor.py:737) and produces clean_idx / flux limits (lines 727-745), but those cuts (FLAGS in {0,2}, SNR>=10, FWHM/ellipticity/edge, flux_auto_min/max) are applied only to the apcor-interpolation reference (table_clean), not to the ZP-fitting sample. The only protection is the reference-side magnitude window (mag_limits_ref, e.g. J [12,15.5]) plus the internal 2.5-sigma clip (photometry.py:119). Saturated/blended/poorly-shaped science sources whose 2MASS counterpart happens to fall in the band-limited window can still enter the solution; there is no explicit SExtractor FLAG=0 nor own-catalog MAGERR<0.1 cut on the fitting list.
- **Impact:** ZP can be pulled by contaminated bright/blended sources in crowded VISIONS star-forming fields (exactly where 2MASS bright stars cluster), biasing the per-tile zero point and hence every calibrated magnitude on that tile. Mitigated, but not eliminated, by the reference magnitude window and sigma-clip; magnitude of bias depends on contamination fraction inside the window.

### M9. Catalog MAGERR omits the aperture-correction interpolation uncertainty and correlated-noise inflation

- **Where:** vircampype/tools/tabletools.py:847,859; vircampype/fits/tables/sextractor.py:868,854-857
- **What:** The published aperture magnitude is `MAG_APER_MATCHED_CAL` = MAG_APER + interpolated growth correction MAG_APER_COR_INTERP (sextractor.py:868). The published error is `magerr_aper = sqrt(MAGERR_APER**2 + photerr_internal**2)` (tabletools.py:847) and `magerr_best` is selected at the same aperture index. This propagates only the raw SExtractor aperture error plus the flat internal-error floor; it never adds the spatial-interpolation scatter of the aperture correction, which the pipeline already computes and stores as `MAG_APER_COR_INTERP_STD` (sextractor.py:855) and then discards for error purposes. It also applies no correlated-noise inflation factor for the LANCZOS-resampled tiles. The SExtractor MAGERR_APER itself is the white-noise estimate that the manual states underestimates errors on resampled images.
- **Impact:** Published MAGERR is a lower bound on the true magnitude uncertainty, most noticeably where the aperture-correction field varies spatially (FWHM gradients across the focal plane) and in resampled coadds where pixel-to-pixel covariance is ignored. Downstream weighting, variability detection, and QFLG assignment (which is keyed directly on magerr_best thresholds, lines 947-949) inherit the optimistic errors.

### M10. Per-plane sky weights 1/bkg_std are not guarded against mmm failure values (sigma = -1.0 or NaN), silently corrupting the weighted master sky

- **Where:** vircampype/fits/images/sky.py:91-102 (weights), with mmm failure returns at vircampype/external/mmm.py:203 (sigma=-1.0), 305 (sigma=np.nan)
- **What:** In _build_sky_detector the weighted-combine path builds weights = (1/bkg_std)[:,None,None] where bkg_std is the per-plane mmm sigma returned by ImageCube.background_planes(). mmm can return sigma = -1.0 (mmm.py:202-211, 'too few valid sky elements') or sigma = NaN (mmm.py:304-309, 'outlier rejection left too few elements') while still returning a finite skymod, so bkg is finite and the plane is kept. The only weight sanitisation is weights[~np.isfinite(cube.cube)] = 0.0 (line 96), which zeroes weights at pixels masked in the CUBE, not where the WEIGHT itself is bad. A plane with sigma=-1.0 therefore enters np.ma.average (cube.py:858) with a uniform NEGATIVE weight, biasing or even sign-flipping the combined sky on every pixel that frame contributes to; a plane with sigma=NaN propagates NaN into every unmasked pixel of that detector's master sky (np.ma.average does not treat NaN weights as masked). The negative-weight case is fully silent (no NaN, no exception); the NaN case eventually trips the min_valid_pixel_fraction guard in process_raw_final and aborts.
- **Impact:** Corrupted master-sky flat/shape for affected detector-groups. Negative-weight planes silently bias the sky used as a flat (sky mode, cube /= sky_norm) or as the subtracted sky shape (twilight mode, cube -= sky_norm*sky_level), producing per-detector additive/multiplicative sky errors. Bites whenever an input sky frame has a heavily masked plane (deep source masks in crowded/nebular fields, offset frames over a bright region, or a dead/low-signal detector) so that mmm's acceptance band drops below minsky. Most damaging in the default sky_combine_metric='weighted' configuration.

## Low severity / hardening

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

## Notes

- Duplicates were merged: ELLIPTICITY (H2) was raised by two dimensions; the SCAMP internal-sigma medium is folded into H1; the LANCZOS correlated-noise low is folded into H3.

- This review made no code changes other than fixing H4 (header-cache invalidation in fits/common.py). H2 (ellipticity) is left for a science decision; a plain-English breakdown was provided separately.

