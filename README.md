# 22481_Project_Group3A
Group project in course 22481 Medical Imaging (DTU). We investigate a phantom with unknown objects using Ultrasound, CT, X-ray and MRI, and use image processing and quantitative measurements to characterise the objects.

**Group members:** Adam Klovborg, Emil Günther, Rasmus Kleffel

## Table of contents
 
- [Repository structure](#repository-structure)
  - [What goes where](#what-goes-where)
- [Setup](#setup)
  - [1. Clone the repository](#1-clone-the-repository)
  - [2. Create the conda environment](#2-create-the-conda-environment)
- [Working conventions](#working-conventions)
- [Commit guide](#commit-guide)
---

## Repository structure

```
22481_Project_Group3A/
├── data/
│   ├── raw/          # Original image data from the scanners. Never edited.
│   └── processed/    # Converted, cropped or cleaned data made by our code.
├── images/           # Figures exported for the report.
├── notebooks/        # Jupyter notebooks where the analysis is done.
├── results/          # Measurement tables (CSV) and other output.
├── src/              # Reusable Python functions used by the notebooks.
├── environment.yml   # Conda environment with all required packages.
└── README.md
```

### What goes where
 
- **data/raw** – files exactly as received. Treat as read-only.
- **data/processed** – anything generated from the raw data. Should be possible to recreate by running the notebooks.
- **notebooks** – one notebook per step or modality
- **src** – functions shared across notebooks. If code is needed in more than one notebook, move it here.
- **results** – numbers that go into the report (dimensions, HU values, CNR, uncertainties), saved as CSV.
- **images** – final figures used in the report.
---
 
## Setup
 
### 1. Clone the repository

```bash
git clone <repo-url>
cd 22481_Project_Group3A
```
 
### 2. Create the conda environment
 
```bash
conda env create -f environment.yml
conda activate <env-name>
```
 
`<env-name>` is the `name:` field at the top of `environment.yml`.
 
If packages are added later, update your environment with:
 
```bash
conda env update -f environment.yml --prune
```

## Working conventions
 
- Do not modify files in `data/raw/`.
- Always keep track of pixel/voxel spacing so measurements are in mm, not pixels.
- Save report numbers to `results/` and figures to `images/` from code, so they can be reproduced.
- Pull before you start working, and commit with short, clear messages (see ref:commit-messages).
---
 
## Commit guide
 
Start each commit message with a prefix that says what kind of change it is, followed by a short description.
 
| Prefix       | Use when you…                                        | 
|--------------|------------------------------------------------------|
| `Add:`       | add something new (file, function, notebook, figure) |
| `Update:`    | change or extend something that already exists       |
| `Fix:`       | correct a bug or a wrong result                      |
| `Remove:`    | delete files or code                                 | 
| `Refactor:`  | restructure code without changing what it does       | 
| `Docs:`      | change README, comments or docstrings                | 
| `Results:`   | add or update output tables or report figures        |