# 🧬 PyProbeMapper

PyProbeMapper is a Python SDK and command-line tool designed to map GEO Platform (GPL) probe IDs to gene symbols for differential gene expression analysis within seconds. It leverages data from the HuggingFace Hub, processes mappings with multiple strategies (accession, coordinate, and direct lookup), and saves results locally for downstream bioinformatics pipelines.

It uses GPT-based inference to intelligently select relevant columns for mapping, saving time and handling variability across GEO platforms.

This tool is ideal for researchers and bioinformaticians working with GPL datasets who need accurate and efficient probe-to-gene mappings.

## 🔍 Features

- ✅ Accurate Probe-to-Gene Mapping for GEO GPL platforms within seconds
- 🤖 GPT-based column inference to automatically select relevant columns, saving time and reducing complexity across diverse GEO platforms
- ⚡ Fast retrieval of existing mappings from a HuggingFace-hosted Zarr dataset
- 🌍 Community-driven mapping: Once a platform is mapped, results are pushed to a central HuggingFace Hub repository (Tinfloz/probe-gene-map), enabling global reuse and collaboration (over 1,000 platforms already mapped!)
- 🧠 Multiple mapping strategies: accession lookup, coordinate lookup, and direct lookup
- 🖥️ Interactive CLI for ease of use
- 💾 Local storage of mappings as JSON files
- 🌐 Push to HuggingFace Hub for sharing and versioning
- 🧩 Easy integration into bioinformatics pipelines or custom scripts
- 📊 Includes a built-in human gene reference dataset (Home_sapiens.GRCh38.genes.tsv)

## 📦 Installation

Install py_probe_mapper from PyPI using your preferred package manager:

```bash
uv pip install py_probe_mapper
```

Or clone the repository and install locally:

```bash
git clone https://github.com/Tinfloz/Probe2GeneMapper
cd Probe2GeneMapper
uv pip install .
```

## 🧪 Example (Python SDK)

Use the `map_probes` function to map probe IDs to gene symbols for one or more GPL platforms:

```python
from py_probe_mapper.sdk import map_probes

# Map probes for GPL570 and GPL96
results = map_probes(
    gpl_ids=["GPL570", "GPL96"],
    output_dir="./mappings",
    force_rebuild=False
)

# Print results
for gpl_id, mappings in results.items():
    if isinstance(mappings, dict):
        print(f"{gpl_id}: Found {len(mappings)} mappings")
    else:
        print(f"{gpl_id}: {mappings}")
```

**Output (example):**
```
GPL570: Found 54675 mappings
GPL96: Found 22283 mappings
```

The mappings are saved as JSON files (e.g., `GPL570_mappings.json`) in the specified `output_dir`.

## 💻 Example (CLI)

Launch the interactive CLI to map probes with a user-friendly interface:

```bash
probe-mapper
```

The CLI will guide you through:
- Enter up to 5 GPL IDs (e.g., GPL570,GPL96)
- Specify the output directory
- Provide optional API URL and key for inference services
- Choose whether to force rebuild existing mappings
- Select a logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

Results are saved as JSON files in the specified directory.

**Sample Interaction:**
```
🌟 Welcome to the GPL Probe Mapper CLI! 🌟

🧬 Enter up to 5 GPL platform identifiers (comma-separated, e.g., GPL570,GPL96): GPL570
📂 Enter output directory (default: .): ./mappings
🔗 Enter API URL for inference service (optional, press enter to skip): 
🔑 Enter API key for inference service (optional, press enter to skip): 
🔄 Force rebuild mappings even if they exist? (default: No): No
📋 Select logging level: INFO

🚀 Starting probe mapping... Please wait! ⏳
🎉 Mapping completed! 🎉
📊 Results:
✅ GPL570: Found 54675 mappings 🧬
```

## 🧠 Mapping Strategies

The tool supports three mapping strategies to ensure robust probe-to-gene mappings:

1. **Accession Lookup**: Matches probes using accession numbers.
2. **Coordinate Lookup**: Uses genomic coordinates for precise mapping.
3. **Direct Lookup**: Directly maps probes to gene symbols when available.

Mappings are fetched from a HuggingFace dataset (Tinfloz/probe-gene-map) or built on-demand using metadata from GEO and the included `Home_sapiens.GRCh38.genes.tsv` reference.

## 📁 Project Structure

```
PyProbeMapper/
├── py_probe_mapper/
│   ├── genome_utils/
│   │   └── Home_sapiens.GRCh38.genes.tsv  # Human gene reference data
│   ├── coordinate_lookup/
│   │   ├── __init__.py
│   │   └── coordinate_lookup.py
│   ├── accession_lookup/
│   │   ├── __init__.py
│   │   └── accession_lookup.py
│   ├── direct_lookup/
│   │   ├── __init__.py
│   │   └── direct_lookup.py
│   ├── lookup_classifier/
│   │   ├── __init__.py
│   │   └── optimised_lookup_classifier.py
│   ├── metadata_builder/
│   │   ├── __init__.py
│   │   └── build_metadata.py
│   ├── __init__.py
│   ├── cli.py                            # Interactive CLI                           
│   └── sdk.py                            # Core SDK
├── pyproject.toml                        # Package configuration
├── README.md                             # This file
```

## 🛠️ Requirements

- Python 3.12+
- questionary>=2.0.0
- fsspec>=2023.1.0
- zarr>=2.14.0
- pandas>=1.5.0
- huggingface_hub>=0.17.0

Install dependencies automatically with:

```bash
pip install py_probe_mapper
```

## 📖 License

**AGPL 3.0 License**

This project is licensed under the AGPL 3.0 License.

See the LICENSE file for details.

## 📚 Usage Notes

**Data Access**: The included `Home_sapiens.GRCh38.genes.tsv` file is used for coordinate-based mapping

**HuggingFace Integration**: Mappings are stored in a Zarr dataset on HuggingFace (Tinfloz/probe-gene-map). Set `force_rebuild=True` to regenerate mappings if needed.

## 🚀 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please include tests.

## 📧 Contact

For questions or support, open an issue on the GitHub repository.