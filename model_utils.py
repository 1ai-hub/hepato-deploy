from __future__ import annotations

import os
import tempfile
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors

MODEL_PATH = os.getenv("MODEL_PATH", "toxicity_model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "features.pkl")

model = joblib.load(MODEL_PATH)
feature_names: List[str] = joblib.load(FEATURES_PATH)


def mol_from_input(smiles: str | None = None, file_bytes: bytes | None = None, filename: str | None = None):
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string.")
        return mol

    if file_bytes is None or not filename:
        raise ValueError("Provide either a SMILES string or a MOL/SDF file.")

    suffix = os.path.splitext(filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if suffix == ".mol":
            mol = Chem.MolFromMolFile(tmp_path)
        elif suffix == ".sdf":
            supplier = Chem.SDMolSupplier(tmp_path)
            mol = next((m for m in supplier if m is not None), None)
        else:
            raise ValueError("Unsupported file type. Use .mol or .sdf")

        if mol is None:
            raise ValueError("Could not parse structure file.")
        return mol
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def featurize_mol(mol, feature_cols: List[str]) -> pd.DataFrame:
    feats: Dict[str, float] = {
        "mol_wt": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rot_bonds": Lipinski.NumRotatableBonds(mol),
        "rings": Lipinski.RingCount(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "frac_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
    }

    fp_cols = [c for c in feature_cols if c.startswith("mfp_")]
    n_bits = len(fp_cols)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.array(list(fp), dtype=int)

    for i, bit in enumerate(arr):
        feats[f"mfp_{i}"] = int(bit)

    row = pd.DataFrame([feats])

    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0

    row = row[feature_cols]
    return row


def predict_toxicity(smiles: str | None = None, file_bytes: bytes | None = None, filename: str | None = None):
    mol = mol_from_input(smiles=smiles, file_bytes=file_bytes, filename=filename)
    x = featurize_mol(mol, feature_names)

    prob = float(model.predict_proba(x)[0][1])
    pct = round(prob * 100, 2)

    if prob < 0.33:
        label = "Low risk"
    elif prob < 0.66:
        label = "Medium risk"
    else:
        label = "High risk"

    return {
        "toxicity_probability": round(prob, 4),
        "toxicity_percent": pct,
        "risk_label": label,
    }