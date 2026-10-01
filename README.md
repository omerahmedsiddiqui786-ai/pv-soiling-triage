# PV Soiling Triage — No-Dedicated-Hardware Prototype

A research prototype that takes a single RGB photo of a solar panel plus a location, and
returns a cleaning recommendation — **clean now / no action / insufficient evidence** —
using only a binary image classifier and free public weather data. No inverter access,
SCADA integration, or dedicated soiling sensor is required at any point.

See [`LICENSE`](LICENSE) for reuse terms and [`CITATION.cff`](CITATION.cff) for how to cite
this repository.

## What this is

- A YOLOv8n-cls binary classifier (clean vs. dusty), trained on a public Kaggle dataset,
  achieving 88.1% validation accuracy.
- An empirically calibrated confidence threshold (0.70) that defers uncertain cases to a
  human ("insufficient evidence") rather than forcing a decision.
- A rule-based decision layer combining classifier output, a short-term rain forecast
  (Open-Meteo), regional irradiance climatology (NASA POWER), and a cited regional
  soiling-rate assumption.
- A cost-sensitivity extension, evaluated by simulation across two real residential net-metered
  PV systems, multiple soiling-rate and cleaning-cost assumptions, and three cleaning
  strategies.
- A preliminary field validation (domain-shift pilot): the classifier evaluated on
  independently captured photographs from two real installations, reported honestly
  including a substantial accuracy drop under domain shift.

## What this is *not*

- Not a claim of novelty for detection-to-scheduling systems in general — several published
  systems already do this with operational/inverter data.
- Not a calibrated soiling-severity estimator — output is binary (clean/dusty) plus a
  confidence-gated "insufficient evidence" fallback, by design. No severity score exists.
- Not a claim of generalization beyond the two installations and the specific photo set
  evaluated here — see Limitations below.
- Not an economically optimal decision rule — evaluated via sensitivity analysis across a
  few assumptions, not presented as an optimizer.
- Not evidence about aerial/drone imagery, industrial-scale farms, or any capture context
  other than ground-level smartphone photography, which is the only context evaluated.

## Repository structure

```
.
├── notebooks/
│   ├── baseline_classifier_training.ipynb   # Trains and validates the classifier
│   ├── weather_decision_layer.ipynb         # Weather API integration + decision-rule dev
│   ├── scenario_cost_analysis.ipynb         # Cost-sensitivity simulation (72 runs)
│   └── field_validation.ipynb               # Domain-shift pilot on real field photos
├── src/
│   └── decision_engine.py                   # decide() / decide_with_cost_sensitivity()
├── figures/
│   └── strategy_comparison.png
├── field_validation_results.csv             # Aggregated, anonymized field-validation results
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Setup

```bash
python -m venv pv-env
# Windows:
pv-env\Scripts\activate
pip install -r requirements.txt
```

Notebooks are developed and run in Google Colab (free T4 GPU tier for training). Kaggle
credentials are read via Colab's Secrets manager (`KAGGLE_USERNAME`, `KAGGLE_KEY`) — never
uploaded as a raw `kaggle.json` file, and never printed to notebook output.

## Running the pipeline

Run in this order:

1. `baseline_classifier_training.ipynb` — produces `best.pt` and the 88.1% validation result.
2. `weather_decision_layer.ipynb` — weather API integration and confidence-threshold
   calibration (0.70 → 96.3% coverage, 98.4% accuracy on the covered subset, n=511).
3. `scenario_cost_analysis.ipynb` — cost-sensitivity simulation.
4. `field_validation.ipynb` — domain-shift pilot; requires `best.pt` and a set of
   human-labeled field photographs (not included in this repository — see Data below).

## Data sources

- **Training data**: [Solar Panel Dust Detection](https://www.kaggle.com/datasets/hemanthsai7/solar-panel-dust-detection)
  (Kaggle, hemanthsai7). Not redistributed in this repository — download it yourself via the
  Kaggle API as shown in the notebooks. Check the dataset's own listing for its license terms
  before reuse.
- **Weather/irradiance**: [Open-Meteo](https://open-meteo.com/) (rain forecast) and
  [NASA POWER](https://power.larc.nasa.gov/) (irradiance climatology), both free public APIs,
  no key required.
- **Field validation photographs**: independently captured from two real residential PV
  installations. **Not included in this repository** — these are private photographs of real
  people's property; only aggregated, anonymized results (`field_validation_results.csv`) are
  shared. If you release any field photographs publicly, strip any embedded camera metadata
  (device info, GPS coordinates) first — some phone camera apps burn this directly into the
  image pixels as a visible caption, not just EXIF metadata.

## Key results

| Metric | Value |
|---|---|
| Classifier validation accuracy (Kaggle val, n=511) | 88.1% |
| Confidence threshold | 0.70 |
| Coverage / accuracy at threshold (n=511) | 96.3% / 98.4% |
| Cost-sensitivity: image+weather+cost strategy wins (import-rate valuation) | 15/24 scenarios |
| Cost-sensitivity: image+weather+cost strategy wins (export-rate valuation) | 20/24 scenarios |
| Field validation accuracy (n=31, domain-shift pilot) | 71.0% |

## Limitations

The field validation result is this project's most important finding: accuracy drops
substantially under domain shift, with the classifier defaulting to "dusty" predictions on
the large majority of field photos, including confidently misclassifying most clean panels.
Two follow-up ablations (metadata/GPS-bar removal, and a panel-isolation crop) did not
resolve this, and complicate a simple "background clutter" explanation — see
`learnings-and-workflow` notes and commit history for the detailed diagnostic trail. Do not
use this classifier's output as-is in any real deployment without accounting for this gap.
Other scope limits: the soiling-rate assumption used (0.8%/day) is drawn from Lahore,
Pakistan, not the installations' actual location (Karachi); the system requires a human to
capture and upload the photo; and rain is treated only as a soft signal, not an automatic
soiling-removal trigger, per cited evidence that adherent soiling often survives rainfall.

## Citation

If you use or build on this code, see [`CITATION.cff`](CITATION.cff).

## License

Code in this repository is released under the MIT License — see [`LICENSE`](LICENSE).
This does not cover the third-party dataset or API terms referenced above; check those
sources' own licenses independently.

## Author

Omer Ahmed Siddiqui
