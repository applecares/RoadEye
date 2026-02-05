# Wildlife Roadkill Detection System

A professional-grade system for detecting, mapping, and analysing wildlife roadkill incidents across Australia.

## Features

- **AI Detection**: YOLOv8-based wildlife detection with species classification
- **Real-time Dashboard**: Leaflet.js map with live detection visualisation
- **Hotspot Prediction**: Gradient Boosting model for risk assessment
- **Edge Device Support**: Offline-capable detection on Raspberry Pi
- **Government Integration**: Darwin Core (NVA) and ArcGIS (LIST) format support
- **Threatened Species Alerts**: Priority handling for endangered wildlife

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/wildlife-roadkill.git
cd wildlife-roadkill

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Run the Server

```bash
roadkill server --port 8000
```

Then open http://localhost:8000 in your browser.

### Generate Demo Data

```bash
roadkill demo generate --count 50
```

### Run Edge Detection

```bash
roadkill edge --source 0 --device-id TEST001
```

## Project Structure

```
wildlife-roadkill/
├── src/roadkill/
│   ├── core/           # Species, events, geography
│   ├── config/         # Settings and YAML loader
│   ├── database/       # SQLAlchemy models
│   ├── detection/      # YOLO detection engine
│   ├── edge/           # Edge device components
│   ├── api/            # FastAPI endpoints
│   ├── dashboard/      # Web dashboard
│   ├── analytics/      # Hotspot prediction
│   ├── integrations/   # NVA, LIST clients
│   └── training/       # Model training
├── tests/              # pytest test suite
├── configs/            # YAML configuration
└── notebooks/          # Jupyter notebooks
```

## Configuration

Configuration is YAML-based with state-specific settings:

```yaml
# configs/states/tasmania.yaml
state:
  name: "Tasmania"
  code: "TAS"

species:
  registry:
    - code: "DEVIL"
      scientific_name: "Sarcophilus harrisii"
      common_name: "Tasmanian Devil"
      threatened_status: "Endangered"
```

## Development

```bash
# Run tests
pytest

# Run linting
ruff check src/ tests/

# Run type checking
mypy src/roadkill
```

## License

MIT License - see LICENSE file for details.
