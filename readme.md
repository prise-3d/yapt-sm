
# YAPT Benchmark with Snakemake

This project automates the execution of rendering scenes using [YAPT](https://github.com/...), collects the resulting `.exr` images, and extracts statistics (such as RGB channel averages) using `oiiotool`. The workflow is managed by [Snakemake](https://snakemake.readthedocs.io) and configured via a CSV file that defines each job's parameters.

## 📁 Project Structure

```
.
├── Snakefile               # Snakemake workflow
├── config.yaml            # Configuration file with paths
├── params.csv             # Scene rendering parameters
├── maps/                  # Files mapping scene name to .exr filename
├── images/                # Rendered .exr images
├── stats/                 # Extracted statistics per image
└── results/
    └── stats_summary.csv  # Summary of means for all images
```

---

## Installation

### 1. Clone this repository
```bash
git clone https://github.com/your-username/yapt-sm.git
cd yapt-sm
```

### 2. Set up a Python environment

```bash
python3 -m venv snakemake
source snakemake/bin/activate
pip install --upgrade pip
pip install snakemake pandas matplotlib
```

### 3. Install YAPT and OIIO

- **YAPT**: Clone and build yapt
- **OpenImageIO** (`oiiotool`)  
  On Ubuntu:
  ```bash
  sudo apt install openimageio-tools
  ```

---

## Configuration

### Example `config.yaml` file:
```yaml
yapt_path: /path/to/yapt/bin
function_path: functions
output_image_path: images
```

### Example `params.csv` file:

This file defines which scenes to run and with which parameters:

```csv
name,source,spp
test1,scene1.ypt,10
test2,scene2.ypt,20
```

---

## Running the Workflow

Once configured, run the full pipeline with:

```bash
snakemake --cores 4 --config params_file=params.csv
```

---

## 📊 Output

The file `results/stats_summary.csv` will contain the average value (extracted from the image stats) for each scene:

```csv
name,mean
test1,0.791
test2,0.812
```

---

## Cleaning Up

To remove all generated files and reset the workflow:

```bash
snakemake --delete-all-output --delete-temp-output
```
