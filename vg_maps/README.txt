surface_wave_tomography/
├── __init__.py               # Project-level init (optional)
├── config/
│   ├── __init__.py           # Usually empty
│   └── default.yaml
├── data/
│   └── (your CSVs, PNGs, etc.)
├── inversion/
│   ├── __init__.py
│   ├── tv_inversion.py
│   └── kernel.py
├── io/
│   ├── __init__.py
│   └── stations.py
├── plotting/
│   ├── __init__.py
│   └── plot_map.py
├── utils/
│   ├── __init__.py
│   └── geo_utils.py
├── main.py
└── requirements.txt
