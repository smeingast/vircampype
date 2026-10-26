"""Phase 3 header cards for the delivered products, and the one way to write them.

Every product HDU that is edited after its data are written carries one reserved
block of 36 blank cards. ``set_cards`` keeps the card count of the edited header
stable: new keys use up trailing blank cards (or the space of cards it removes), and
removed cards are replaced by blank cards, so the header never grows or shrinks across
a 2880-byte block boundary and the data offsets never move. ``set_cards`` removes
CHECKSUM/DATASUM, which the edit invalidates (astropy's ``checksum="remove"`` drops them
in every HDU of the file); the finalizer writes PRODCATG, ASSONi and the missing
checksums last.
"""

import datetime
import logging
import math
import numbers
import os

from astropy.io import fits

import vircampype
from vircampype.pipeline.errors import PipelineError, PipelineValueError
from vircampype.tools import sdpcheck

log = logging.getLogger(__name__)

Card = tuple[str, object, str]  # (keyword, value, comment)


def _unique_or_multi(
    key: str, index_key: str, values: list, comment: str
) -> list[Card]:
    """One card with the unique value, or MULTI plus one indexed card per value."""
    unique = sorted(set(values))
    if len(unique) == 1:
        return [(key, unique[0], comment)]
    indexed = [(f"{index_key}{i}", v, comment) for i, v in enumerate(unique, start=1)]
    return [(key, "MULTI", comment)] + indexed


def obs_cards(headers: list[fits.Header], setup) -> list[Card]:
    """Observation block of a product, derived from the raw primary headers.

    OBJECT is the pipeline run name (``setup.name``), not the raw OB target name.
    """
    if not headers:
        raise PipelineValueError("obs_cards needs at least one raw header")
    kw = setup.keywords

    def strings(key: str) -> list[str]:
        return [str(h[key]).strip() for h in headers]

    filters = strings(kw.filter_name)
    techs = sorted(set(strings("HIERARCH ESO DPR TECH")))
    if len(techs) > 1:
        raise PipelineValueError(f"Mixed observation techniques (DPR TECH): {techs}")
    mjd = [float(h[kw.date_mjd]) for h in headers]
    exptimes = [float(h[kw.dit]) * int(h[kw.ndit]) for h in headers]
    ob_ids = sorted({int(h["HIERARCH ESO OBS ID"]) for h in headers})

    cards = [("ORIGIN", "ESO-PARANAL", "European Southern Observatory")]
    cards += _unique_or_multi("TELESCOP", "TELESC", strings("TELESCOP"), "Telescope")
    cards += _unique_or_multi("INSTRUME", "INSTR", strings("INSTRUME"), "Instrument")
    cards += _unique_or_multi("FILTER", "FILTER", filters, "Filter name")
    if len(set(filters)) == 1:
        cards.append(("HIERARCH ESO INS FILT1 NAME", filters[0], "Filter name"))
    cards.append(("OBJECT", setup.name, "Target designation"))
    cards.append(("MJD-OBS", min(mjd), "Start of observations (days)"))
    mjd_end = max(m + t / 86400 for m, t in zip(mjd, exptimes))
    cards.append(("MJD-END", mjd_end, "End of observations (days)"))
    cards.append(("TEXPTIME", float(sum(exptimes)), "Total integration time (s)"))
    cards.append(("NCOMBINE", len(headers), "Number of raw science data files"))
    progs = strings("HIERARCH ESO OBS PROG ID")
    cards += _unique_or_multi("PROG_ID", "PROGID", progs, "ESO program identification")
    cards += [
        (f"OBID{i}", ob, "Observation block ID") for i, ob in enumerate(ob_ids, 1)
    ]
    cards.append(("OBSTECH", techs[0], "Technique of observation"))

    # Detector and jitter parameters only when identical in all inputs
    uniform = [
        ("DIT", kw.dit, "Detector integration time (s)"),
        ("NDIT", kw.ndit, "Number of detector integrations"),
        ("NJITTER", "NJITTER", "Number of jitter positions"),
        ("NOFFSETS", "NOFFSETS", "Number of offset positions"),
        ("NUSTEP", "NUSTEP", "Number of microstep positions"),
    ]
    for key, raw_key, comment in uniform:
        values = {h.get(raw_key) for h in headers}
        if len(values) == 1 and None not in values:
            cards.append((key, values.pop(), comment))

    cards.append(
        ("PROCSOFT", f"vircampype v{vircampype.__version__}", "Reduction software")
    )
    cards.append(("REFERENC", setup.reference or "", "Bibliographic reference"))
    return cards


def prov_cards(arcfiles: list[str]) -> tuple[list[Card], fits.BinTableHDU | None]:
    """PROVi cards for up to 9999 raw files, else PROVXTN and a provenance extension."""
    names = [a.removesuffix(".fits") for a in arcfiles]
    too_long = [n for n in names if len(n) > 68]
    if too_long:
        raise PipelineValueError(f"Provenance entries over 68 characters: {too_long}")
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise PipelineValueError(f"Duplicate provenance entries: {duplicates}")
    if len(names) <= 9999:
        cards = [
            (f"PROV{i}", n, "Originating raw science file")
            for i, n in enumerate(names, 1)
        ]
        return cards, None
    column = fits.Column(name="PROV", format="68A", array=names)
    hdu = fits.BinTableHDU.from_columns([column], name="PHASE3PROVENANCE")
    return [("PROVXTN", True, "Provenance in PHASE3PROVENANCE extension")], hdu


def _check_card(key: str, value, comment: str) -> None:
    """Raise PipelineValueError unless the card is valid and one 80-character record."""
    if key != key.upper():
        raise PipelineValueError(f"Keyword '{key}' is not upper case")
    if isinstance(value, numbers.Real) and not math.isfinite(value):
        raise PipelineValueError(f"Value of '{key}' is not finite: {value}")
    # Exactly one physical card (no CONTINUE records), and astropy can write it
    try:
        image = fits.Card(key, value, comment).image
    except Exception as e:
        raise PipelineValueError(f"Card '{key}' cannot be written: {e}") from e
    if len(image) != 80:
        raise PipelineValueError(f"Card '{key}' does not fit in one 80-char card")


def set_cards(
    path: str, cards: list[Card], hdu: int | str = 0, remove: tuple[str, ...] = ()
) -> None:
    """Write cards to one HDU in place without changing any card count.

    Raises PipelineValueError, leaving the file untouched, for an invalid card (lower-case
    keyword, non-finite number, or not writable as one 80-character record) or when the
    trailing blank cards plus the removed cards cannot hold the new keys. All checks run
    before the update open, because ``checksum="remove"`` rewrites the headers on close
    even when nothing else changed.
    """
    for card in cards:
        _check_card(*card)

    # Reserve: trailing blank cards plus the cards removed below
    header = fits.getheader(path, hdu)
    blanks = 0
    for card in reversed(header.cards):
        if not card.is_blank:
            break
        blanks += 1
    removed = [
        k for k in dict.fromkeys(("CHECKSUM", "DATASUM", *remove)) if k in header
    ]
    keys = dict.fromkeys(key for key, _, _ in cards)
    needed = sum(1 for k in keys if k in removed or k not in header)
    available = blanks + len(removed)
    if needed > available:
        raise PipelineValueError(
            f"Not enough header space in {path} [HDU {hdu}]: "
            f"{needed} new cards, {available} available"
        )

    with fits.open(path, mode="update", checksum="remove") as hdul:
        header = hdul[hdu].header
        n0 = len(header)
        for key in removed:
            header.remove(key)
        for key, value, comment in cards:
            header.set(key, value, comment)
        while len(header) < n0:
            header.append(fits.Card(), end=True)

        # checksum="remove" drops CHECKSUM/DATASUM in every HDU on close; blank cards
        # take their place so the other HDUs keep their card counts, too
        for other in hdul:
            n = len(other.header)
            for key in ("CHECKSUM", "DATASUM"):
                other.header.remove(key, ignore_missing=True)
            while len(other.header) < n:
                other.header.append(fits.Card(), end=True)
    log.debug(f"Wrote {len(cards)} cards to {path} [HDU {hdu}]")


def expected_products(pipeline) -> list[tuple[str, str, tuple[str, ...]]]:
    """Products the setup delivers: (path, PRODCATG, ASSONi basenames), batch order."""
    setup = pipeline.setup
    products = []
    if setup.build_tile:
        expmap = setup.path_coadd_expmap
        products += [
            (setup.path_coadd, "SCIENCE.IMAGE", (os.path.basename(expmap),)),
            (expmap, "ANCILLARY.EXPMAP", ()),
            (setup.path_coadd_sources, "SCIENCE.SRCTBL", ()),
        ]
    if setup.build_stacks:
        for path in pipeline.stacks.paths_full:
            weight = path.removesuffix(".fits") + ".weight.fits"
            products += [
                (path, "SCIENCE.MEFIMAGE", (os.path.basename(weight),)),
                (weight, "ANCILLARY.WEIGHTMAP", ()),
            ]
    return products


def finalize_products(pipeline) -> None:
    """Write identity and checksums to every product of the pipeline run."""
    finalize_files(expected_products(pipeline))


def finalize_files(products: list[tuple[str, str, tuple[str, ...]]]) -> None:
    """Check, then seal products with identity cards and checksums, then check again.

    ``products`` holds (path, PRODCATG, ASSONi basenames) in batch order (images before
    their source lists). Nothing is written unless every file passes ``sdpcheck`` with
    ``final=False`` and every HDU has header space for its checksums. Existing checksums
    are trusted; ``set_cards`` removes them in every HDU of the file it edits, so after
    the identity cards each HDU gets fresh ones. DATE (UTC, the time of sealing) goes
    into every primary: ESO's validator requires it for the science categories.
    Running it again is harmless (it refreshes DATE).

    Raises PipelineError with all findings when a check fails, and PipelineValueError
    when an HDU lacks the blank cards for its new cards.
    """
    # Pass 1: products as the recipes wrote them
    problems = []
    for path, prodcatg, _ in products:
        problems += sdpcheck.check_product(path, prodcatg, final=False)
    if problems:
        raise PipelineError("Products fail the Phase 3 check:\n" + "\n".join(problems))

    # Preflight, before anything is written: valid identity cards, complete files (the
    # checksums read all data), and header space. set_cards turns the CHECKSUM/DATASUM
    # cards of every HDU into blank cards and the new identity keys of the primary use
    # up blank cards; add_checksum then needs two blank cards per HDU, else the header
    # grows and the data move.
    identity = {}
    short = []
    date = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S")
    for path, prodcatg, asson in products:
        cards = [("PRODCATG", prodcatg, "Data product category")]
        cards += [
            (f"ASSON{i}", name, "Associated file") for i, name in enumerate(asson, 1)
        ]
        cards += [("DATE", date, "Date the file was written (UTC)")]
        for card in cards:
            _check_card(*card)
        identity[path] = cards
        with fits.open(path) as hdul:
            info = [hdul.fileinfo(i) for i in range(len(hdul))]
            end = max(i["datLoc"] + i["datSpan"] for i in info)
            if os.path.getsize(path) < end:
                raise PipelineValueError(
                    f"{path} is truncated: {os.path.getsize(path)} bytes, "
                    f"its HDUs end at {end}"
                )
            for index, hdu in enumerate(hdul):
                header = hdu.header
                space = sum(key in header for key in ("CHECKSUM", "DATASUM"))
                for card in reversed(header.cards):
                    if not card.is_blank:
                        break
                    space += 1
                if index == 0:
                    space -= sum(key not in header for key, _, _ in cards)
                if space < 2:
                    short.append(f"{path} [HDU {index}]: {space} blank cards left")
    if short:
        raise PipelineValueError(
            "Not enough header space for identity and checksums:\n" + "\n".join(short)
        )

    # Identity, then checksums for every HDU that lacks them
    for path, _, _ in products:
        set_cards(path, identity[path], hdu=0)
        with fits.open(path, mode="update") as hdul:
            for hdu in hdul:
                if "CHECKSUM" not in hdu.header or "DATASUM" not in hdu.header:
                    hdu.add_checksum()

    # Pass 2: sealed products and their batch
    paths = [path for path, _, _ in products]
    problems = []
    for path, prodcatg, _ in products:
        problems += sdpcheck.check_product(path, prodcatg)
    problems += sdpcheck.check_batch(paths)
    if problems:
        raise PipelineError(
            "Finalized products fail the check:\n" + "\n".join(problems)
        )
    for path, prodcatg, _ in products:
        log.info(f"Finalized {os.path.basename(path)} ({prodcatg})")
