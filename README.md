# Multi-Strategy Virtual Screening of CHEK1 Kinase Inhibitors

Computational discovery of novel **checkpoint kinase 1 (CHEK1)** inhibitors from the
ChemDiv million-compound library, combining ligand-based and structure-based virtual
screening followed by deep-learning rescoring.

> **Scope of this repository:** code, input files, and key results only. The final
> written report is **not** included (coursework deliverable). See
> [Reproducing the workflow](#reproducing-the-workflow) to re-run the pipeline.

---

## Scientific background

CHEK1 is a serine/threonine kinase that mediates the G2/M DNA-damage checkpoint. Tumour
cells that have lost p53 become heavily dependent on CHEK1-driven arrest to survive
endogenous replication stress, making CHEK1 an attractive anti-cancer target. The aim of
this project is to identify novel, drug-like CHEK1 inhibitor candidates by large-scale
virtual screening.

A key structural insight from the co-crystallised ligand in PDB **4HYI** (Compound 40,
ligand 1AO) is that it binds **without a strong hinge hydrogen bond** — the interaction
is dominated by hydrophobic-pocket occupation plus weak polar contacts. This motivated a
screening strategy that does **not** treat the hinge hydrogen bond as an indispensable
pharmacophore, but instead prioritises correct occupancy of the ATP pocket.

## Pipeline

```
1,681,854  ──►  931  ──►  929  ──►  10 final leads
 (ChemDiv)    (RF + SVM +        (Vina        (OnionNet-2
               Tanimoto)          docking)     consensus ranking)
```

| Stage | Step | Method / Tool |
|-------|------|---------------|
| 1 | Data acquisition | ChEMBL activity records for CHEK1 (target CHEMBL4630) |
| 2 | Training-set curation | binary active/inactive labels, dedup, train/test split |
| 3 | Ligand-based screening | Random Forest + SVM classifiers, Morgan fingerprints (RDKit), Tanimoto similarity to AZD7762 / prexasertib |
| 4 | Structure-based docking | AutoDock Vina 1.2.5 into the ATP pocket of PDB 4HYI |
| 5 | Pose quality control | custom filter on docking validity |
| 6 | Rescoring | OnionNet-2 (residue–atom contacting-shell CNN) |
| 7 | Consensus ranking | Vina + OnionNet-2 rank fusion |
| 8 | Final selection | drug-likeness (Lipinski Ro5) + Bemis–Murcko scaffold diversity |

## Final candidates

| Rank | Compound ID | Vina (kcal/mol) | OnionNet-2 pKd |
|------|-------------|-----------------|----------------|
| 1 | 116183 | −10.79 | 8.16 |
| 2 | 113935 | −10.91 | 8.09 |
| 3 | 128026 | −10.58 | 8.17 |
| 4 | 165154 | −11.15 | 7.92 |
| 5 | 327212 | −10.31 | 8.19 |
| 6 | 105171 | −11.09 | 7.82 |
| 7 | 163186 | −10.76 | 7.84 |
| 8 | 281324 | −10.37 | 7.92 |
| 9 | 345488 | −10.19 | 8.01 |
| 10 | 261779 | −10.39 | 7.87 |

## Repository structure

```
├── input/                                    # Target & reference ligands
│   ├── 4HYI.pdb                              # CHEK1 co-crystal structure (PDB 4HYI)
│   ├── 4HYI_protein.pdb                      # receptor-only (water removed)
│   ├── 4HYI_ligand.pdb                       # co-crystallised ligand (1AO)
│   ├── AZD7762.sdf                           # reference inhibitor (similarity query)
│   └── Prexasertib.sdf                       # reference inhibitor (similarity query)
├── scripts/                                  # screening pipeline (Python)
│   ├── download_chembl.py                    # fetch ChEMBL activity data
│   ├── clean_chembl.py                       # label + dedup + train/test split
│   ├── train_ml.py                           # train RF / SVM classifiers
│   ├── screen_library.py  (and _mp.py)       # ligand-based screening of ChemDiv
│   ├── scaffold_cluster.py                   # Bemis–Murcko scaffold clustering
│   ├── run_vina_docking.py                   # Vina docking (single-process)
│   ├── run_vina_parallel.py / vina_worker.py # Vina docking (parallel)
│   ├── check_pose_quality_vina.py            # docking pose QC
│   ├── run_onionnet_on_vina.py               # OnionNet-2 rescoring
│   ├── rank_consensus.py                     # Vina + OnionNet-2 consensus ranking
│   ├── select_final_leads.py                 # final lead selection
│   ├── check_interactions.py                 # protein–ligand interaction analysis
│   └── make_vina_complexes.py                # generate docked complexes
├── results/                                  # key output tables (CSV)
│   ├── consensus_ranking.csv                 # final consensus-ranked candidates
│   ├── final_leads.csv                       # final 10 leads
│   ├── interaction_analysis.csv              # key contacts per candidate
│   └── ...
├── complexes/                                # final 10 lead–protein complexes (PDB)
│   └── CHEK1_<id>_complex.pdb
└── figures/                                  # PyMOL renderings (4HYI & binding modes)
```

## Reproducing the workflow

1. **Prepare the receptor & reference ligands** — `input/` contains the 4HYI structure
   (water removed) and the two similarity queries.

2. **Build the training set** — `download_chembl.py` fetches CHEK1 (CHEMBL4630)
   activity records; `clean_chembl.py` converts them into a binary classification set.

3. **Train the classifiers** — `train_ml.py` fits RF and SVM models on Morgan
   fingerprints.

4. **Screen the library** — `screen_library.py` applies the classifiers and Tanimoto
   similarity to the ChemDiv library (downloaded separately from ChemDiv) to retain the
   top candidates.

5. **Dock** — `run_vina_parallel.py` docks the retained set into the 4HYI ATP pocket
   with AutoDock Vina 1.2.5; `check_pose_quality_vina.py` filters bad poses.

6. **Rescore & rank** — `run_onionnet_on_vina.py` rescoring with OnionNet-2, followed by
   `rank_consensus.py` to fuse Vina and OnionNet-2 ranks.

7. **Select & analyse** — `select_final_leads.py` applies drug-likeness and scaffold
   diversity; `check_interactions.py` + `make_vina_complexes.py` produce the complexes
   and interaction tables for visualisation in PyMOL.

## Dependencies

- Python 3, **RDKit**, **scikit-learn**, **meeko**, **Open Babel**
- **AutoDock Vina 1.2.5**
- **OnionNet-2** (https://github.com/zhenglz/onionnet-2)
- **PyMOL** (visualisation only)

## Data sources & notes

- **ChemDiv library** (~1.68 M compounds) — obtained from ChemDiv, **not** redistributed
  here; download from the vendor directly.
- **ChEMBL** activity data — public database (https://www.ebi.ac.uk/chembl/).
- **PDB 4HYI** — Huang *et al.* (2013) *Bioorg. Med. Chem. Lett.* 23, 2590–2594.
- Trained models, the full screening feature matrix, and third-party tool dependencies
  are intentionally excluded (large / re-generable). See `.gitignore`.
