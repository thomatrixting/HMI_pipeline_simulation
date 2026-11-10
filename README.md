# HMI pipeline simulation — umbral OBS_VR artifact

Goal: show quantitatively that the ~350 m/s oscillating residual seen over
sunspot umbrae in SDO/HMI Dopplergrams is an instrumental artifact of the
Doppler pipeline, not a real velocity. The full specification is in
[`docs/instrucions.md`](docs/instrucions.md).

## Repository layout

```
.
├── docs/                    Specs and notes
│   ├── instrucions.md         specification of the simulation (Spanish)
│   └── rotacion_data.md       rotation data notes
│
├── rotation_model/          Section 1 — SDO ↔ Sun geometry and polynomial inversion
│   ├── relative_velocity_sim.py   relative LOS velocity SDO–solar point vs time/latitude
│   ├── main.ipynb                 JSOC data, HMI polynomial inversion, depth comparisons
│   ├── main2.ipynb                early exploration (drms + relative velocity)
│   └── figures/                   PNGs produced by the notebooks and the script
│
└── hmi_artifact_sim/        Section 2 — forward model of the HMI Doppler pipeline
    ├── README.md              detailed write-up (Part I synthetic, Part II real profiles)
    ├── *.py                   package modules (pipeline, profiles, zeeman, real_*, …)
    ├── notebooks/             05_plots, 06_perfiles_reales
    ├── figures/               experiment figures (exp A/B/C, R1–R10)
    └── report.html
```

## How to run

- **Section 1:** open the notebooks from inside `rotation_model/` (figures are
  saved to `rotation_model/figures/`), or run
  `python rotation_model/relative_velocity_sim.py`.
- **Section 2:** from the repository root,
  `python -m hmi_artifact_sim.tests` (synthetic, 18 checks) and
  `python -m hmi_artifact_sim.tests_real` (real profiles; first run downloads
  ~325 MB into `hmi_artifact_sim/data/`, which is git-ignored).
