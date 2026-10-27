# Quick Start

## Prerequisites
- Python 3.10+ (developed on 3.12)
- Windows PowerShell (commands below are PowerShell-friendly)

## Installation
```powershell
cd E:\Latent
python -m pip install -e ".[dev]"
# Optional UI
python -m pip install -e ".[ui]"
```

## Configuration
```powershell
Copy-Item .env.example .env
```
Defaults are fine. `DLRW_MOCK_MODE=true` runs fully offline with no API key.

## Run the Application
```powershell
streamlit run app.py
```

## Run Tests
```powershell
python -m pytest          # full suite
python -m pytest -v       # verbose
```

Verified: **276 passed, 1 warning** (see `docs/AUDIT.md`).
