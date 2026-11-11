"""Retrain Spec2Vec on GNPS AllPositive with the project's compounds held out.

Reproduces the training setup of the pretrained model in models/spec2vec_positive.model
(Zenodo 4173596: spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model), but removes
every spectrum whose compound (first 14 characters of the InChIKey) appears in the
pesticides or pharmaceuticals libraries, plus any spectrum with a matching spectrum_id.

Training data: the cleaned GNPS positive-mode dataset from Huber et al. (Zenodo 3978118).
It is downloaded automatically (~340 MB zip, ~1.3 GB JSON) if not found.

Usage (from the project folder, in the spec_benchmark environment):
    python scripts/retrain_spec2vec.py                 # held-out model (main experiment)
    python scripts/retrain_spec2vec.py --no-exclude    # control: full data, same pipeline
    python scripts/retrain_spec2vec.py --dry-run       # build corpus and report counts only
"""
import argparse
import json
import logging
import os
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.filtering import (
    normalize_intensities,
    reduce_to_number_of_peaks,
    require_minimum_number_of_peaks,
    select_by_mz,
    select_by_relative_intensity,
)
from rdkit import Chem, RDLogger
from spec2vec.model_building import train_new_word2vec_model

RDLogger.DisableLog("rdApp.*")

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
MODELS = PROJECT / "models"
ZENODO_URL = (
    "https://zenodo.org/api/records/3978118/files/"
    "gnps_positive_ionmode_cleaned_by_matchms_and_lookups.zip/content"
)
JSON_NAME = "gnps_positive_ionmode_cleaned_by_matchms_and_lookups.json"
LIBRARIES = ["pesticides", "pharmaceuticals"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def ensure_dataset(json_path: Path) -> Path:
    """Download and unzip the AllPositive dataset if it is not already present."""
    if json_path.exists():
        return json_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = json_path.with_suffix(".zip")
    if not zip_path.exists():
        print(f"Downloading training data from Zenodo to {zip_path} ...")
        urllib.request.urlretrieve(ZENODO_URL, zip_path)
    print(f"Unzipping {zip_path} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extract(JSON_NAME, json_path.parent)
    return json_path


def inchikey14_from_structure(smiles=None, inchi=None):
    """InChIKey14 from SMILES or InChI, or None if neither parses."""
    mol = None
    if isinstance(smiles, str) and smiles.strip() and smiles.strip() != "N/A":
        mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None and isinstance(inchi, str) and inchi.strip().startswith("InChI="):
        mol = Chem.MolFromInchi(inchi.strip())
    return Chem.MolToInchiKey(mol)[:14] if mol is not None else None


def record_inchikey14(rec):
    """InChIKey14 of a GNPS record: stored InChIKey first, then computed from structure."""
    for key in ("inchikey", "inchikey_smiles", "inchikey_inchi"):
        ik = rec.get(key)
        if isinstance(ik, str) and len(ik.strip()) >= 14:
            return ik.strip()[:14]
    return inchikey14_from_structure(rec.get("smiles"), rec.get("inchi"))


def load_exclusions():
    """Spectrum IDs and InChIKey14s of every spectrum in the project's two libraries."""
    ids, ik14s = set(), set()
    for name in LIBRARIES:
        md = pd.read_csv(DATA / f"{name}_metadata.csv")
        ids.update(md["spectrum_id"].dropna())
        ik14s.update(k for k in md["smiles"].map(inchikey14_from_structure) if k)
    return ids, ik14s


# ---------------------------------------------------------------------------
# Spectrum processing (matches the original Spec2Vec AllPositive training)
# ---------------------------------------------------------------------------
def record_to_spectrum(rec):
    peaks = rec.get("peaks_json")
    if isinstance(peaks, str):
        peaks = json.loads(peaks)
    if not peaks:
        return None
    arr = np.asarray(peaks, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    try:
        precursor_mz = float(rec.get("precursor_mz"))
    except (TypeError, ValueError):
        precursor_mz = None
    try:
        parent_mass = float(rec.get("parent_mass"))
    except (TypeError, ValueError):
        parent_mass = None
    if not parent_mass or parent_mass <= 0:
        return None  # needed by reduce_to_number_of_peaks(ratio_desired=...)
    metadata = {"spectrum_id": rec.get("spectrum_id"), "parent_mass": parent_mass}
    if precursor_mz and precursor_mz > 0:
        metadata["precursor_mz"] = precursor_mz
    try:
        return Spectrum(mz=arr[:, 0], intensities=arr[:, 1], metadata=metadata,
                        metadata_harmonization=False)
    except (AssertionError, ValueError):
        return None


def post_process(s):
    """Peak processing used for the pretrained AllPositive model ("ratio05" in its name)."""
    s = normalize_intensities(s)
    s = select_by_mz(s, mz_from=0, mz_to=1000)
    s = require_minimum_number_of_peaks(s, n_required=10)
    s = reduce_to_number_of_peaks(s, n_required=10, ratio_desired=0.5, n_max=500)
    if s is None:
        return None
    s_no_low = select_by_relative_intensity(s, intensity_from=0.001)
    if len(s_no_low.peaks) >= 10:
        s = s_no_low
    return s


def spectrum_to_words(s, n_decimals=2, loss_from=5.0, loss_to=200.0):
    """Peak words plus neutral-loss words, as in the original training corpus.

    spec2vec >= 0.8 no longer emits loss words, but the pretrained model was trained
    with them, so they are rebuilt here to keep the two models comparable.
    """
    words = [f"peak@{mz:.{n_decimals}f}" for mz in s.peaks.mz]
    losses = s.compute_losses(loss_mz_from=loss_from, loss_mz_to=loss_to)
    if losses is not None:
        words += [f"loss@{mz:.{n_decimals}f}" for mz in losses.mz]
    return words


def build_corpus(json_path, exclude):
    print(f"Loading {json_path} (this takes a few minutes and several GB of RAM) ...")
    with open(json_path) as f:
        records = json.load(f)

    excl_ids, excl_ik14 = load_exclusions() if exclude else (set(), set())
    stats = {"records": len(records), "excluded_by_id": 0, "excluded_by_compound": 0,
             "dropped_unparseable": 0, "dropped_processing": 0}
    corpus = []
    for rec in records:
        sid = rec.get("spectrum_id") or rec.get("spectrumid")
        if exclude:
            if sid in excl_ids:
                stats["excluded_by_id"] += 1
                continue
            if record_inchikey14(rec) in excl_ik14:
                stats["excluded_by_compound"] += 1
                continue
        s = record_to_spectrum(rec)
        if s is None:
            stats["dropped_unparseable"] += 1
            continue
        s = post_process(s)
        if s is None:
            stats["dropped_processing"] += 1
            continue
        corpus.append(spectrum_to_words(s))
    del records

    stats["training_documents"] = len(corpus)
    stats["excluded_compounds"] = len(excl_ik14)
    stats["vocabulary_size"] = len({w for doc in corpus for w in doc})
    return corpus, stats


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path, default=DATA / "external" / JSON_NAME,
                   help="Path to the AllPositive JSON (downloaded here if missing).")
    p.add_argument("--no-exclude", action="store_true",
                   help="Train on the full dataset (control run to validate the pipeline).")
    p.add_argument("--iterations", type=int, default=15,
                   help="Training epochs (default 15, matching the pretrained model).")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="Training threads.")
    p.add_argument("--out", type=Path, default=None, help="Output model path.")
    p.add_argument("--dry-run", action="store_true", help="Build the corpus, report counts, do not train.")
    args = p.parse_args()

    exclude = not args.no_exclude
    tag = "heldout" if exclude else "fullcontrol"
    out = args.out or MODELS / f"spec2vec_{tag}_iter{args.iterations}.model"

    corpus, stats = build_corpus(ensure_dataset(args.json), exclude)
    print(json.dumps(stats, indent=2))
    if not exclude:
        print("Pretrained model was trained on 95,320 documents with a 115,910-word vocabulary.")
    if args.dry_run:
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(f"Training {args.iterations} epochs on {len(corpus):,} documents with {args.workers} workers ...")
    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    train_new_word2vec_model(corpus, iterations=[args.iterations], filename=str(out),
                             workers=args.workers, progress_logger=True)
    stats.update(iterations=args.iterations, training_minutes=round((time.time() - t0) / 60, 1),
                 model_path=str(out))
    out.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    print(f"Done in {stats['training_minutes']} min. Model: {out}")


if __name__ == "__main__":
    main()
