# Plastic Filament Verification System (PFVS)

The **Plastic Filament Verification System** is an open-source tool designed to classify 3D printing filament materials using **spectroscopy** and machine learning. Built around the OctoPrint ecosystem, PFVS helps makerspaces reduce failed prints and filament waste by validating filament type before a print begins.

## 🔧 Features

- Real-time filament classification with NIR spectral input
- Pre-trained PCA + SVM models for PLA, PETG, and ASA
- OctoPrint plugin integration for seamless verification
- STL files for 3D-printable sensor housing
- Modular design for adding new materials and retraining models

---

## 📁 Repository Structure

### `Drawings/`
CAD files and renders of the custom PFVS housing for the NIR sensor.

- STL Files: `BASE.stl`, `LID.stl`, `BOX.stl`
- Render Images: `BASE.png`, `LID.png`, etc.
- Complete Assembly: `PFVS ASSEMBLY.png`

### `FilamentDatabase/`
Raw NIR scan data (CSV format) used to train the classifier.

- Each file contains **1000 scans** of a single filament variant.
- Naming convention: `MaterialColorfulldatabase.csv`
  - Example: `PETWfulldatabase.csv` → PETG white

### `PCA-SVM/`
Machine learning scripts and trained models for filament classification.

- `predict_material.py`: Runs prediction using saved PCA and SVM models
- `domain_shift_model.py`: Adjusts for environmental or device variation
- `models_per_color/`: Serialized ML models grouped by color
- `data_current/`, `data_old/`: Different datasets taken in different environments and times for variance in database.

### `PFVSPlugin/`
Custom OctoPrint plugin that handles NIR data collection and filament verification.

- `octoprint_pfvs/`: Core plugin package
  - `predict_material.py`, `spectrometer.py`: Machince Learning model prediction and spectrometer initilization
  - `filament_gcodes.py`: G-code insertion for verified materials
  - `templates/`, `static/`: Frontend components (HTML, CSS, JS)
  - `*.pkl`: Serialized ML artifacts (models + encoders)
- `setup.py`, `setup.cfg`, `MANIFEST.in`: Packaging and installation configs

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/PFVS.git
cd PFVS
```
### 2. Set Up OctoPrint Plugin (on Raspberry Pi)
Copy PFVSPlugin/ to your OctoPrint plugin directory

Install required dependencies:
``` bash
pip install -r requirements.txt
```

For full run down of how to build/replicate/use our system, please refer to our Users Manual.
© 2025 Arizona Board of Regents on behalf of the University of Arizona, Engineering Design Center



