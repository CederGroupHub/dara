# Dara Web Server Tutorial

The **Dara server** is a web backend that powers the Dara application, providing APIs for data management, phase analysis, and automated refinement. This tutorial will guide you through what the Dara server is, how to start it, and how to configure its parameters.

---

## What is the Dara Server?

The Dara server is a FastAPI-based backend that manages data storage, computation, and communication for the Dara platform. It provides:

### Core Capabilities

**⚙️ Config-Free Operation**
- The Dara server can be started and used immediately without any configuration files.
- All essential settings (host, port, database backend, etc.) have sensible defaults.
- You can override any setting via command-line arguments or environment variables if needed.
- No need for installation of MongoDB or other database systems.

**🔬 Automated Phase Identification**
- Upload powder XRD patterns in multiple formats (`.xy`, `.txt`, `.xye`, `.xrdml`, `.raw`, `.rasx`)
- Automated phase identification using crystallographic databases (COD, ICSD)
- Advanced Rietveld refinement using BGMN engine
- Machine learning-enhanced phase matching

**📊 Interactive Analysis**
- Web-based user interface with real-time visualization
- Multiple refinement solutions with quality rankings
- Interactive plots with zoom, pan, and export capabilities
- Task management and result tracking

**🔗 API Access**
- RESTful API for programmatic access
- Batch processing capabilities
- Integration with external workflows

---

## How to Start the Dara Server

### 1. Install Dara

First, ensure you have installed Dara and its dependencies:

```bash
pip install dara
```

### 2. Basic Server Startup

**Command Line Interface:**
```bash
dara server
```

This starts the server with default settings:
- Host: `127.0.0.1` (localhost)
- Port: `8898`
- Database: MontyDB (local file-based, pure python)

**Access the Web Interface:**
Open your browser and navigate to: `http://localhost:8898`

### 3. Custom Configuration

**With Command Line Arguments:**
```bash
# Start on different host/port
dara server --host 0.0.0.0 --port 9000

# Use MongoDB backend
dara server --database-backend mongodb --mongodb-host localhost --mongodb-port 27017
```

**With Environment Variables:**
```bash
# Set environment variables
export DARA_SERVER_HOST=0.0.0.0
export DARA_SERVER_PORT=9000
export DARA_SERVER_DATABASE_BACKEND=mongodb
export DARA_SERVER_MONGODB_HOST=localhost
export DARA_SERVER_MONGODB_PORT=27017
export DARA_SERVER_MONGODB_DATABASE=dara_server
export DARA_SERVER_MONGODB_USERNAME=dara_user
export DARA_SERVER_MONGODB_PASSWORD=your_password

# Start server
dara server
```

---

## Configuration Parameters

### Server Settings

| Parameter | Default | Description | Environment Variable |
|-----------|---------|-------------|---------------------|
| `host` | `127.0.0.1` | Server host address | `DARA_SERVER_HOST` |
| `port` | `8898` | Server port number | `DARA_SERVER_PORT` |

**Example:**
```bash
# Allow external connections
dara server --host 0.0.0.0 --port 8080
```

### Database Configuration

The server supports two database backends:

#### MontyDB (Default - File-based)
| Parameter | Default | Description | Environment Variable |
|-----------|---------|-------------|---------------------|
| `database_backend` | `monty` | Database type | `DARA_SERVER_DATABASE_BACKEND` |
| `montydb_path` | `~/.dara-server/montydb` | Database file location | `DARA_SERVER_MONTYDB_PATH` |

**Example:**
```bash
# Custom MontyDB location
dara server --montydb-path /data/dara/montydb
```

#### MongoDB (Production)
| Parameter | Default | Description | Environment Variable |
|-----------|---------|-------------|---------------------|
| `database_backend` | `monty` | Set to `mongodb` | `DARA_SERVER_DATABASE_BACKEND` |
| `mongodb_host` | `localhost` | MongoDB host | `DARA_SERVER_MONGODB_HOST` |
| `mongodb_port` | `27017` | MongoDB port | `DARA_SERVER_MONGODB_PORT` |
| `mongodb_database` | `dara_server` | Database name | `DARA_SERVER_MONGODB_DATABASE` |
| `mongodb_username` | `None` | Username (optional) | `DARA_SERVER_MONGODB_USERNAME` |
| `mongodb_password` | `None` | Password (optional) | `DARA_SERVER_MONGODB_PASSWORD` |

**Example:**
```bash
# MongoDB with authentication
dara server \
  --database-backend mongodb \
  --mongodb-host db.example.com \
  --mongodb-port 27017 \
  --mongodb-database dara_production \
  --mongodb-username dara_user \
  --mongodb-password your_password
```

### Advanced Features Configuration

#### Reaction Prediction Setup

To enable reaction prediction features:

1. **Set API Key:**
   ```bash
   export MP_API_KEY=your_materials_project_api_key
   ```

2. **Start server:**
   ```bash
   dara server
   ```

**Get Materials Project API Key:**
- Register at https://materialsproject.org/
- Navigate to your dashboard to find your API key

---

## API Usage

### Submit Analysis

**Endpoint:** `POST /api/submit`

**Python Example:**
```python
import requests
from pathlib import Path

def submit_analysis(
    file_path,
    precursors,
    user="researcher",
    base_url="http://localhost:8898"
):
    url = f"{base_url}/api/submit"

    with open(file_path, 'rb') as f:
        files = {'pattern_file': f}
        data = {
            'user': user,
            'precursor_formulas': str(precursors),
            'use_rxn_predictor': True,
            'temperature': 800,  # °C
            'instrument_profile': 'Aeris-fds-Pixcel1d-Medipix3',
            'wavelength': 'Cu'
        }

        response = requests.post(url, files=files, data=data)
        return response.json()

# Example usage
result = submit_analysis(
    "sample.xrdml",
    ["CaO", "TiO2"],
    user="john_doe"
)
print(f"Job submitted with ID: {result['wf_id']}")
```

### Check Task Status

**Endpoint:** `GET /api/task/{task_id}`

```python
def check_status(task_id, base_url="http://localhost:8898"):
    response = requests.get(f"{base_url}/api/task/{task_id}")
    return response.json()

status = check_status(123)
print(f"Status: {status['status']}")
```

### List All Tasks

**Endpoint:** `GET /api/tasks`

```python
def list_tasks(user=None, page=1, limit=10, base_url="http://localhost:8898"):
    params = {'page': page, 'limit': limit}
    if user:
        params['user'] = user

    response = requests.get(f"{base_url}/api/tasks", params=params)
    return response.json()

tasks = list_tasks(user="john_doe")
```

---

## Summary

The Dara server provides a powerful platform for automated XRD analysis with:

- **Easy Setup:** Single command to start (`dara server`)
- **Flexible Configuration:** Environment variables and CLI arguments
- **Multiple Backends:** MontyDB for development, MongoDB for production
- **API Access:** RESTful endpoints for integration
- **Web Interface:** User-friendly browser-based UI

For basic usage, simply run `dara server` and navigate to `http://localhost:8898`. For production deployments, configure MongoDB backend and appropriate security measures.
