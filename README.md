# ECG Analyzer — Fase 1

Pipeline reproducible de análisis de señales ECG orientado a la detección
de fibrilación auricular. **Sistema de investigación/triaje — no reemplaza
la revisión clínica de un cardiólogo.**

---

## Requisitos

- Python 3.10 o superior
- Las dependencias están fijadas en `requirements.txt`

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd ecg-analyzer

# 2. Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Ejecución del notebook de exploración

```bash
jupyter lab notebooks/01_eda_mitbih.ipynb
```

La primera celda descarga automáticamente la MIT-BIH Arrhythmia Database
(~100 MB) en `data/mitdb/`. Las descargas posteriores se omiten.

---

## Ejecución de los tests

```bash
pytest tests/ -v
```

Para reporte de cobertura:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Estructura del proyecto

```
ecg-analyzer/
├── configs/
│   └── default.yaml          # Hiperparámetros del pipeline (YAML)
├── data/                     # Datasets (gitignored)
├── notebooks/
│   └── 01_eda_mitbih.ipynb   # Exploración completa de MIT-BIH
├── src/
│   ├── io/
│   │   └── loaders.py        # Carga de registros WFDB
│   ├── preprocessing/
│   │   ├── filters.py        # Pasa-banda, notch, baseline
│   │   └── normalize.py      # Z-score, min-max, resampling
│   ├── features/
│   │   ├── rpeaks.py         # Detección picos R (NeuroKit2)
│   │   ├── hrv.py            # SDNN, RMSSD, pNN50, detector irregularidad
│   │   └── quality.py        # SQI — índice de calidad de señal
│   └── visualization/
│       └── plots.py          # Gráficos anotados con matplotlib
└── tests/
    ├── test_filters.py
    └── test_rpeaks.py
```

---

## Configuración del pipeline

Todos los parámetros relevantes viven en `configs/default.yaml`.
No es necesario modificar el código fuente para ajustar el filtrado:

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `preprocessing.bandpass.lowcut` | 0.5 Hz | Corte inferior del pasa-banda |
| `preprocessing.bandpass.highcut` | 40.0 Hz | Corte superior del pasa-banda |
| `preprocessing.notch.freq` | 50.0 Hz | Frecuencia de red (60.0 para EE.UU.) |
| `preprocessing.baseline.method` | `"highpass"` | `"highpass"` o `"wavelet"` |
| `hrv.irregularity_threshold_cv` | 0.15 | CV mínimo para marcar irregularidad RR |
| `hrv.irregularity_threshold_rmssd_ms` | 100.0 | RMSSD mínimo (ms) para marcar irregularidad |
| `quality.min_sqi` | 0.5 | Umbral mínimo de calidad de señal |

---

## Datasets

### Fase 1 — MIT-BIH Arrhythmia Database
- **Fuente**: PhysioNet (descarga automática vía `wfdb`)
- **Tamaño**: ~100 MB, 48 registros de 30 min a 360 Hz
- **Uso**: exploración, validación del pipeline

### Fase 2 — PTB-XL *(pendiente)*
- 21 799 registros, 500 Hz, 12 derivaciones, etiquetado multi-label
- Solo derivación I (simulación de smartwatch)

---

## Aviso legal

Este software es exclusivamente una **herramienta de investigación**.
Los resultados (incluyendo la detección de posible irregularidad RR)
**no constituyen diagnóstico médico**. Cualquier decisión clínica debe
ser tomada por personal sanitario cualificado.
