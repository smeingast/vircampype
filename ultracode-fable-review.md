# vircampype logging system: ultracode review (Fable)

_Generated 2026-06-10. Multi-agent review of the v2.4.0 logging/terminal-output overhaul: 5
parallel reviewers (skip-path tracing, terminal UX, 2 logging-gap sweeps, infrastructure) followed
by 4 adversarial verifiers, one per dimension. 64 findings raised, **64 confirmed, 0 refuted**
(several adjusted with corrected details, incorporated below), plus 8 verifier-added findings.
Cross-category duplicates are merged here. **No code was changed** — this is a suggestion list,
ranked by priority._

Priorities: **P1** = user-visible inconsistency or real work with no record anywhere;
**P2** = solid improvement with clear benefit; **P3** = polish/hardening.

**Status (2026-06-10, first fix round, codex-double-checked):** IMPLEMENTED — P1.1a (decorator
skip notice via `print_stage_skip`), P1.1b-lite (elapsed footer added to all 8 early-return skip
branches via new `print_elapsed`; full decorator-owned banner/elapsed NOT yet done), P1.4
(source-detection skip-logging inversion), P1.6 (`--sort`/`--reset`/`shallow_clean` logging;
`--reset progress` now keeps `pipeline_*.log*`; `clean_directory` returns counts), P2.6
(QC-summary skips demoted from WARNING, both sites). OPEN — P1.2, P1.3, P1.5 and the rest.

---

## Why the resumed-run output is inconsistent (inc.txt explained)

Every stage method hand-rolls its own console contract: banner (`print_header`), optional
column caption, optional per-file bar, and a copy-pasted trailing
`print_message("\n-> Elapsed time: …")`. The `@pipeline_step` decorator (main.py:25-62) knows the
stage name and wraps all 38 stages, but emits nothing to the console. Three skip levels then
produce three different appearances:

| Observed pattern | Mechanism |
|---|---|
| Stage missing entirely from the terminal | Decorator skip: `log.info("Skipping …")` at main.py:50-54 is file-only (console handler is WARNING+, logsetup.py:112-116) |
| Banner + `File` caption + elapsed, nothing between | Per-file `check_file_exists` skips inside the loop (DEBUG, file-only); method falls through to its elapsed print |
| Banner + caption, **no** elapsed (SCAMP, MASTER-SOURCEMASK, TILE-HEADER) | All-products-exist **early `return` between banner and elapsed print**: sextractor.py:145-147 (elapsed at 296), sky.py:1069-1070 (elapsed at 1269), sky.py:2042-2047 (elapsed at 2107) |
| Seven consecutive bare `TILE STATISTICS <MODE>` banners | One stage loops 7 modes (main.py:791-801); each `coadd_statistics_tile` call prints a captionless banner (sky.py:3166-3171, `left=None, right=None`) then early-returns (sky.py:3183-3188) before content and elapsed |
| `Building FWHM table...` glued to the elapsed line | `print_message(…, end="")` leaves the cursor mid-line (sky.py:457); the elapsed message's leading `\n` closes it |
| Persisted `1/1 20:17:56` bar line under PUBLIC CATALOG | By design (bars persist, `transient=False`) — but only stages that did work leave one, adding to the asymmetry |

Eight early-return sites in total lose the elapsed footer: `scamp` (sextractor.py:145),
`build_master_source_mask` (sky.py:1069), `build_coadd_header` (sky.py:2042),
`build_master_sky_static` (sky.py:1794), `coadd_statistics_tile` (sky.py:3183),
`plot_qc_photerr_internal` (sextractor.py:1813), `build_phase3_tile` (esotools.py:609),
`build_qc_summary` (fitstools.py:1316). Only `qc_completeness_tile` gets it right
(completeness.py:1506-1513 prints elapsed inside the skip branch). The calibration path
(flat.py/dark.py) uses loop-`continue` instead of early returns and is NOT affected.

---

## P1 — fix first

### P1.1 Make `pipeline_step` the single emitter of the per-stage console contract

- **Where:** pipeline/main.py:25-62 (decorator); ~44 `-> Elapsed time` call sites across 8 modules; 8 early-return sites listed above.
- **What:** Banner, skip notice, and elapsed footer are re-implemented per method, so: (a) decorator-skipped stages print nothing at all — a resumed run's terminal cannot distinguish "already complete" from "not configured"; (b) 8 stages lose their elapsed footer to early returns; (c) three elapsed formats coexist (`-> Elapsed time: 50.25s`, bars' `H:MM:SS`, `All done in Xs`); (d) decorator file-log labels don't match console banners in at least 9 stages (`MASTER-SOURCE-MASK` vs `MASTER-SOURCEMASK`, `TILE HEADER` vs `TILE-HEADER`, `PAWPRINT RESAMPLING` vs `RESAMPLING`, `TILE` vs `CREATING TILE`, …), breaking console↔file-log correlation.
- **Suggestion:** Move the whole contract into the decorator (or a stage context manager): on skip, one dim console line (e.g. `✓ MASTER-SKY — already complete`) + the existing INFO record; on run, print the banner from the decorator's `message` (single source of truth for console and file), time around `method()` and emit one standardized elapsed footer in a `finally` block so no early return can drop it. Then delete the ~44 per-method elapsed prints and per-method banners. A resumed run becomes a complete, ordered checklist of all 38 stages.

### P1.2 One banner per stage: TILE/STACKS STATISTICS print 7-8 banners each

- **Where:** pipeline/main.py:788-810 (tile), 772-786 (stacks); sky.py:3166-3188, 3029-3034; sextractor.py:1566.
- **What:** The single `TILE STATISTICS` checkpoint renders as 7 captionless mode banners + the differently-named `STATISTICS TABLES` banner (8 total); on a resumed run all are bare two-line banners conveying nothing — the noisiest block of the transcript. `combine_mjd_images` between them logs/prints nothing (and is re-run with `overwrite=True` on every resume). `TILE QC PHOTOMETRY` similarly emits 3 banners (sextractor.py:1923/1991/2181).
- **Suggestion:** Print the stage banner once (falls out of P1.1) and render sub-steps as rows beneath it: a 7-step `track()` bar when working, a single dim "all 7 products present" line when skipped.

### P1.3 Human output is torn across stdout and stderr

- **Where:** messaging.py:99-104, 148, 176-178, 202-204, 318 (raw `print()` → stdout); logsetup.py:48 (`Console(stderr=True)` carries bars + all WARNING+ records); progress.py:120.
- **What:** Banners/info/elapsed go to stdout; progress bars and every warning/error go to stderr. The streams only interleave correctly on a shared TTY. `pipeline > run.log` captures banners but **loses all warnings and bars**; `2>&1` merges nondeterministically (stdout block-buffered when redirected); `| tee` delays banners. Critically, in the documented `file_log: false` container mode (logsetup.py:179-181, setup.py:47) a stdout-only capture loses WARNINGs **with no record anywhere**. Nothing documents the split as intentional.
- **Suggestion:** Route all human output through the one cached rich Console (`get_console().print(...)`) so banners, bars, log records, and QC blocks share one stream, one width, and rich's Live-safe interleaving — which also makes the `_finalize_progress()`-before-print dance unnecessary as a correctness requirement. Decide the stream once and document it in logsetup.py.

### P1.4 SOURCE DETECTION skip-logging is inverted — dead code, and resumed runs are unreconstructable

- **Where:** fits/images/sky.py:291-299.
- **What:** A genuine logic bug: the loop selects files to process via `if self.setup.overwrite or not os.path.isfile(pt):` and calls `check_file_exists(pt)` only *inside* that branch — i.e. on paths already known not to exist, so the DEBUG "already exists, skipping" line can never fire. Catalogs that DO exist fall through with zero records, and there is no count summary. On a resumed run, the file log for this stage contains only the banner line and elapsed — and this method backs five pipeline steps (astrometry, illumcorr, pawprint/stacks/tile photometry).
- **Suggestion:** Call `check_file_exists` first (logging each skip), build `indices_to_process` from its result + `overwrite`, and add one INFO summary: `Source detection (preset X): creating M/N catalogs (N-M exist)`.

### P1.5 Loky worker processes lose all log records (completeness stage is unreconstructable)

- **Where:** tools/completeness.py:776 (`Parallel(prefer="processes", verbose=10)`), 522/560/585 (`run_command_shell(silent=True)` in workers); systemtools.py:372-378.
- **What:** `configure_logging` runs only in the parent. A loky child re-imports modules with an unconfigured, handler-less `vircampype` logger: every DEBUG/INFO record — the entire SExtractor/SkyMaker/detection command+output capture, thousands of commands in the longest QC stage — is dropped; WARNINGs surface only via `logging.lastResort` as raw stderr lines, never in the file. Meanwhile `verbose=10` makes joblib print its own `[Parallel] Done X out of Y` chatter raw to stderr, bypassing the logging policy, and the parallel branch has no per-sub-tile progress (the serial branch at 781-788 has both). Same structural gap for the `prefer="processes"` pools in data/cube.py:1033/1179/1468 (their helpers currently don't log, so completeness is the concrete loss today).
- **Suggestion:** Drop `verbose=10`. Drive progress from the parent (joblib `return_as="generator"` is available in the pinned joblib ≥ 1.5, calling `message_calibration` per completed sub-tile). For worker records: have `measure_completeness` return a compact per-tile summary the parent logs, or configure a per-child handler (`QueueHandler`/`QueueListener`).

### P1.6 Destructive operations leave no record anywhere

- **Where:** worker.py:177-197 (`--reset`), tools/datatools.py:101-104/152-153/184-190 (`--sort`), main.py:987-1029 (`shallow_clean`/`archive`/`deep_clean`), systemtools.py:481-497/528-546.
- **What:** Three clusters of silent destruction/relocation:
  - `--reset all` deletes the entire output tree and `--reset progress` wipes `temp/` with no print and no log record — and since `Pipeline()` is constructed first (worker.py:174), both delete the **just-created live log file** (the open handler keeps writing to an unlinked inode), so even the init records are lost. `--reset cache` and `--dry-run` report via bare `print()`.
  - `--sort` `shutil.move`s thousands of raw FITS files; datatools.py contains no logging or print whatsoever — the standalone log file `_run_sort` dutifully creates stays empty.
  - `shallow_clean`/`archive` delete whole intermediate-product directories via `clean_directory` → `remove_file`, neither of which logs anything (`remove_file` even swallows `OSError` with `pass`); `deep_clean` additionally wipes the active run log.
- **Suggestion:** Log a WARNING (reaches console + file) before each reset/clean describing exactly what is removed and how many files; in `--reset progress`, spare `pipeline_*.log` or reconfigure the handler after the wipe; add a module logger to datatools.py with an INFO summary per sort (counts + destinations) and DEBUG per move; have `clean_directory` return/log the per-directory file count.

---

## P2 — solid improvements

### Terminal output / modernization

- **P2.1 Replace BColors + hand-drawn rules with rich primitives.** messaging.py:52-65, 100-104, 142-146, 176-178, 202-204, 304-318. Raw ANSI escapes are emitted unconditionally — redirected/cluster/container captures get literal `\033[96m` bytes and `\r` characters, despite `_banner_width`'s docstring explicitly anticipating those environments. Delete `BColors`, use `console.rule(title)` and `console.print(…, style=…)`; rich then handles width, color detection, and ANSI stripping for free. (Slots into P1.3.)
- **P2.2 Retire the `\r` + 80-char-pad + `end=""` idiom in `print_message`.** messaging.py:113, 142-148. Every info line carries an invisible `\r` and trailing pad into captures; elapsed footers smuggle a leading `\n` inside the message; `end=""` leaves the cursor mid-line (causes the CLASSIFICATION glue, sky.py:457, and the dangling `Coadding <file>` line, sky.py:3235). Make it emit one complete line per call with an explicit blank-line policy.
- **P2.3 Remove the `File`/`Extension` caption machinery.** messaging.py:71-72, 102-104 and ~44 call sites (~20 pass `right=None`, 8 pass both `None` — three different banner shapes). The caption is a dead header for per-row prints that no longer exist; rows now render in the bar (on the other stream). Replace with a one-line stage status with real counts, e.g. `12 products: 0 to build, 12 exist` — which also fixes the information gap where a user can't see how much was skipped. (SOURCE DETECTION's caption even overstates work: "Running Sextractor … on 12 files" is printed from `len(self)` before the existence check, sky.py:282-299, even when 0 files need processing.)
- **P2.4 BASIC RAW PROCESSING (and any worker-thread loop) never shows a bar.** sky.py:919-932, 753-760. `message_calibration` is called inside `_process_one_basic_file`, which runs on joblib worker threads; `_can_drive_live` (progress.py:265-276) correctly refuses to drive the bar there, so the longest per-file stage shows banner → silence → elapsed even when doing real work (the observed 50.25s). Drive the bar from the main thread (joblib `return_as="generator_unordered"` or executor + `as_completed`, advancing per completed file — the same pattern `run_commands_shell_parallel` already uses, systemtools.py:312-319).
- **P2.5 Route `message_qc_astrometry` through the console/stage layout.** messaging.py:296-318. It bypasses rich, gates nothing on TTY, and is the only raw printer that does NOT call `_finalize_progress()` first — and its sole call site (sextractor.py:1353) runs right after a `message_calibration` loop, so a live bar is guaranteed to be active: the raw print corrupts/interacts with the Live display, and under rich's redirect machinery lands on the wrong stream. Render via the console with style by threshold; keep the logfmt file record (the good part).

### Skip/consistency

- **P2.6 QC SUMMARY is the only stage whose skip is loud — and it warns.** main.py:903-910 + fitstools.py:1306-1316. The one undecorated stage hand-rolls its checkpoint skip as `kind="warning"`, so a routine resume prints rich-formatted WARNINGs while all other skips are invisible — exactly inverted severity semantics. Decorate it with `@pipeline_step` and use the unified skip notice; reserve WARNING for anomalies. Same for completeness's routine status warnings (completeness.py:259-263, 300-304 — "PSF models available for N/M" warns even when N == M — and 733-737).

### Logging gaps (work invisible in console AND file)

- **P2.7 MASTER-SKY per-group loop has no progress and no per-group record.** sky.py:1504-1586. Minutes of per-group parallel collapse; the outpath is only ever logged on the *skip* path; the only per-group record ("Mean flat-field variation: X%", :1562) doesn't identify the group. Add `message_calibration` (main-thread loop, so bar + DEBUG both work) + a `Written: <outpath>` INFO.
- **P2.8 `tile_fits` writes hundreds of sub-tile cutouts invisibly.** fitstools.py:1106-1263. No logger, no progress, no print — minutes of multi-GB I/O at the start of TILE QC COMPLETENESS with zero trace until "Tiled into N sub-tiles" afterwards.
- **P2.9 NoiseChisel batch: commands and output never reach the file log.** Live path sky.py:1203-1205 (`run_commands_shell_parallel(silent=False)`): the non-silent branch (systemtools.py:300-302) inherits stdio and captures nothing; the `ran: {cmd}` DEBUG exists only in the silent branch (:295); commands are built with `-q` so the console shows nothing either; and `bar_label = label if silent else None` (:311) kills the bar too. Call with `silent=True` + a label like every other tool batch, and move the `ran:` record outside the silent branch. (The reviewer's originally-cited site cube.py:1287 is dead code — `build_source_masks_noisechisel` has no callers; consider deleting it.)
- **P2.10 `--cluster` never configures logging.** worker.py:205-236; cluster.py (no logging import at all, print-only, including destructive queue clearing and abort/kill paths). `configure_standalone_logging`'s docstring claims the cluster path uses it (logsetup.py:193-195) — only `--sort` does. A cluster failure's CRITICAL lands on an unconfigured logger → `lastResort` stderr only. Call `configure_standalone_logging` in `_run_cluster` and mirror cluster.py prints with log records. Also: the top-level handler catches `Exception` only — a Ctrl-C leaves no terminal record in the log on any path.
- **P2.11 Completeness science fallbacks are silent.** completeness.py:524-532 (catalog-read failure silently disables false-match filtering → biases completeness up), 588-597 (failed detection read recorded as 0% recovery and `continue`), 349-350 (logistic-fit failure → `pass`, NaN comp50/90). And if `run_completeness` returns an empty list, *everything* including the no-finite-values warning sits inside `if results:` (completeness.py:1529) — the stage prints only the elapsed footer, writes no products, and `@pipeline_step` still sets `qc_completeness_tile=True`, making the empty outcome both invisible and permanent. Add WARNINGs at each fallback + an `else` branch; consider not setting the flag on an empty run.
- **P2.12 `build_qc_summary` collection: bare `except Exception: pass` ×2.** main.py:913-924. Stacks/tile sections silently vanish from `qc_summary.ecsv` with the exception text discarded. Replace with `log.warning(f"QC summary: stacks unavailable ({e})")` — the pattern already used inside `build_qc_summary` itself (fitstools.py:1336-1337).
- **P2.13 Missing NDIT silently defaults to 1.** fits/images/common.py:57-64. The KeyError fallback feeds dark scaling (dark.py:98), flat normalization (flat.py:126), and read-noise (flat.py:942) — wrongly scaled masters with no record. One WARNING naming the keyword and file count.
- **P2.14 No provenance for external tools.** astromatic.py has no logging: resolved binary paths, versions (`-v`), and chosen presets (e.g. `scamp_mode` → scamp_loose/ffp.yml, astromatic.py:370-385) are never recorded — doubly relevant given the patched-SExtractor requirement. One INFO per tool at init.
- **P2.15 `rsync_file` bypasses command logging.** systemtools.py:421-457. Inherited stdio, no `ran:` record — used in the SCAMP cache path (sextractor.py:132/283/289) and write-then-publish (cube.py:495), so cache hits/misses and large-product copies leave no trace. Log + capture like the other runners.
- **P2.16 `message_calibration(silent=True)` also kills the file-log trace.** messaging.py:236-242 returns before `report_progress`, which owns the DEBUG record (progress.py:280-282) — so `setup.silent` (documented as suppressing *terminal* progress) silently degrades file-log reconstructability for nearly every per-file loop. Always emit the DEBUG line; let `silent` gate only the bar.

### Infrastructure

- **P2.17 `setup.log_level` is a no-op.** logsetup.py:174-175 validates the name and discards the result (logger pinned DEBUG, file handler DEBUG, console from `console_log_level`). A user setting `log_level: warning` sees zero change. Wire it to the file-handler level or remove/rename the knob.
- **P2.18 Dead console knobs; doc-behavior mismatches.** setup.py:42-46: `no_color` and `force_terminal` are read nowhere (`Console(stderr=True)` is built bare, logsetup.py:48), so cluster/container users cannot control color/TTY detection as documented; `console_log_level`'s "None → derived from silent" comment is false (`_console_level` never reads `silent`, hard WARNING); an invalid `console_log_level` raises a bare AttributeError unlike `log_level`'s validation. Honour the knobs in `get_console()`/`configure_logging` (note: the console is created lazily and cached, so it needs a configure step that can rebuild it) or delete them.
- **P2.19 `file_log: false` silently swallows captured warnings.** logsetup.py:179-182. With no file handler, `captureWarnings(True)` routes warnings to a handler-less `py.warnings` logger; verified on Python 3.13 that they are silently dropped (logging attaches a NullHandler — they do NOT even reach lastResort), in both fresh and reconfigured processes. And INFO/DEBUG records go nowhere despite the comment promising "a container relying on stdout redirection". Restore `py.warnings` propagation (or attach a stream handler) in the no-file branch; consider a plain stdout StreamHandler for the container mode.
- **P2.20 Root logger never configured: third-party log records bypass the file log.** logsetup.py:153-154. matplotlib/joblib/astroquery (and astropy's *logger* output) propagate to the handler-less root → lost (or lastResort). NOTE (verifier-corrected): AstropyWarning-family **warnings** DO reach the file log in real runs — astropy is imported before `configure_logging`, so `captureWarnings(True)` wins the `showwarning` hook — the gap is third-party *log records*, not astropy warnings. Attach the file handler (or a WARNING-level one) to the root logger — duplication is impossible since `vircampype` has `propagate=False` — tagged with `_own()` for idempotent reconfig.

---

## P3 — polish / hardening

- **P3.1** `photerr_internal` prints nothing on a cache hit — banner AND elapsed live inside the `except FileNotFoundError` compute branch (sextractor.py:1501-1556); inverse of every other stage. Log the cache hit; align with the unified contract.
- **P3.2** `check_file_exists`'s `silent` parameter is dead ("retained for signature compatibility", messaging.py:257-271) yet 38 call sites still pass it, some expecting loud behavior (sky.py:1069 passes `silent=False`). Remove it (or make it the per-file skip-notice hook).
- **P3.3** `silent=` plumbing is inconsistent overall: `print_header` is silent *by default* (messaging.py:70) so every call site must remember to pass `silent=self.setup.silent`. Centralize quiet mode in the console layer instead of threading a flag through every call.
- **P3.4** Dead code from the overhaul: `_ProgressDriver._task` (progress.py:197-204, no callers), `BColors.BOLD`/`UNDERLINE`, commented-out legacy prints (sky.py:2051-2054, :2517, :2589), uncalled `build_source_masks_noisechisel` (cube.py:1274-1288). Pure removal.
- **P3.5** The post-loop "finalizing" spinner is anonymous and timer-less (progress.py:177, 100-101) — long post-loop work (FITS assembly, plots) shows only a pulsing "finalizing". Label it with the stage and add `show_elapsed=True`.
- **P3.6** Custom progress columns re-implement stock rich idioms (pinned rich ≥ 13, installed 14.3.3): `_ElapsedOrClockColumn` duplicates `TimeElapsedColumn`'s H:MM:SS+freeze; `SpinnerColumn(finished_text="✓") + TextColumn` could replace `_LabelColumn` if the spinner-left layout is acceptable. Keep only the load-bearing custom behavior (wall-clock finish stamp).
- **P3.7** Per-run log plumbing edge cases (logsetup.py:79, 102-109, 184-185): redundant `touch()`; `_prune_old_logs` globs `stale_base + "*"` without `glob.escape`; 1-second filename resolution can collide across processes. Add `%f` + PID to the filename, escape the glob, drop the touch.
- **P3.8** `log_retention` defaults to None (unlimited): up to 300MB/run (50MB × 6) accumulates forever on repeatedly-reprocessed setups. Default to a small positive number.
- **P3.9** Setup dump logged as a multi-hundred-line indented JSON blob at INFO (main.py:105) — breaks the line-oriented greppability of the file log. DEBUG + compact (or `key=value` per line), keep a one-line INFO summary.
- **P3.10** Stale docstrings: logsetup.py:9-10 still says the console handler "is added in a later migration phase" (it's installed unconditionally, :155); progress.py:18-19 says the TTY check is on *stdout* (it's the stderr console). Document the stream split (see P1.3).
- **P3.11** Phase-3 tile skip records mislead: `check_file_exists` DEBUG "skipping" lines fire per file (esotools.py:609-612), but unless all 3 products exist everything is rebuilt with `overwrite=True` — the log says "skipping" for files that are then rebuilt. Test the group with `os.path.isfile`, log the group decision once.
- **P3.12** `combine_mjd_images` (fitstools.py:902-947): no logging, called once per stack/tile with unconditional `overwrite=True` — recreated on every resume, invisibly. One INFO per call; honor the skip convention.
- **P3.13** tabletools dormant bare prints: ~20 verbose-gated per-cut count prints in `clean_source_table` (tabletools.py:143-318) + `\r` iteration progress in `interpolate_classification` (:405, 448-452). Convert the cut counts to unconditional `log.debug` (valuable provenance!); delete the `\r` progress.
- **P3.14** MASTER-WEIGHT-IMAGE `except IndexError` fallback silently switches from masking detector 16 to masking ALL detectors (sky.py:1994-2004) — changes downstream coadd weights with no record. Log which branch ran.
- **P3.15** `read_yml` prints YAML parse errors raw to stdout and raises without `from exc` (systemtools.py:142-147) — parse details never reach the file log; traceback loses the cause.
- **P3.16** `remove_file` swallows ALL OSErrors (EACCES/EBUSY included), `remove_directory._on_error` passes silently (systemtools.py:481-525) — failed deletions indistinguishable from successful ones in reset/clean paths. Keep FileNotFoundError silent; WARN the rest.
- **P3.17** `header_reset_wcs` abandons the whole WCS re-fit on any KeyError with no record (wcstools.py:57-114) — silently disables the VIRCAM distortion cleanup for that extension. Narrow the try, log the missing key.
- **P3.18** Calibration QC plots written with no record (flat.py:257-260, 802-805, 986-990, 644-651; dark.py:193-195) while the FITS masters all get `Written:` lines. Mirror the convention.
- **P3.19** `PipelineError(logger=)` logs ERROR in `__init__` — used by exactly 1 of ~40 raise sites (tabletools.py:770), double-logs with the top-level handler, wrong type hint. Drop the parameter.
- **P3.20** Persisted bars freeze short of N/N on partially-resumed stages (per-file skips `continue` before `message_calibration`, e.g. sky.py:1623-1637) — the persisted bar suggests an unfinished loop and the finalizing/finish stamp never engages. Count skips into the bar (also makes resumed stages show a truthful N/N).
- **P3.21** `--reset progress` deletes the just-created active log (worker.py:174 constructs Pipeline → log file in temp/ → :195-197 wipes temp/). Subsumed by P1.6 but worth its own test once fixed.

---

## Suggested implementation order

1. **P1.1 + P1.2** (decorator-owned banner/skip/elapsed) — one structural change that eliminates the whole inc.txt inconsistency class and most of the skip-consistency P2/P3 items (P2.6, P3.1, P3.2 fall out).
2. **P1.3 + P2.1/P2.2/P2.3** (single rich console, drop BColors/`\r`/captions) — the modernization pass; mechanical once P1.1 owns the layout.
3. **P1.4, P1.5, P1.6** — the three independent "work with no record" bugs (source-detection inversion, loky workers, destructive ops).
4. P2 gaps/infra in any order; P3 opportunistically alongside.

## Verification notes

Every finding above was adversarially re-checked against the code by an independent verifier
agent (and the inc.txt mechanisms additionally hand-verified): all line references re-read, rich
API suggestions checked against the pinned rich (≥13, installed 14.3.3), joblib suggestions
against pinned joblib ≥1.5, and two empirical checks run on Python 3.13 (astropy warning routing,
`py.warnings` NullHandler swallowing). Notable verifier corrections already folded in: astropy
*warnings* do reach the file log in real runs (import-order effect) — the third-party gap is
logger records (P2.20); the NoiseChisel finding's original site is dead code, the live defect is
in sky.py (P2.9); joblib verbose chatter goes to stderr, not stdout (P1.5).
