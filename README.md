# ODTQ-MMC±X Model Files

📅 **Last updated:** December 6, 2025  

---

## Reference

This repository accompanies the paper:

> **Greening, Lacy M., et al.**  
> *Integrating Order-to-Delivery Time Sensitivity in E-Commerce Middle-Mile Consolidation Network Design.*  
> *IISE Transactions*, accepted 2025.

---
## Code Description

## Code Description

All Python code files are now included and operational. The repository is organized into modular components that build, solve, and analyze the ODTQ-MMC±X models.

| File | Description |
| :--- | :--- |
| **`optFunctions.py`** | Contains data-processing utilities, helper functions for reading input data, and class definitions (e.g., `data`, `odt_data`) used by both the MIP and heuristic models. |
| **`heuristicFunctions.py`** | Implements functions for the heuristic approach. |
| **`mmtc_odt_bin_mip.py`** | Builds the binary ODTQ-MMC±X MIP model. |
| **`mmtc_odt_pw_mip.py`** | Builds the piecewise-linear ODTQ-MMC±X MIP model. |
| **`mmtc_odt_bin_mip_reject.py`**| Builds the binary MIP model allowing for order rejection (used for Table 8 results). |
| **`heuristic.py`** | Main script for running heuristic models. Integrates data loading, model selection, and solution execution for user-specified model types. |
| **`add_ltl.py`** | Script for handling Less-Than-Truckload (LTL) additions (used for Table 8 results). |

---

## Data Description

The dataset used in this study is contained in the folder:

data-GPDEM-2025-IISETr/

This folder includes all network, route, lane, and customer choice data required to reproduce the results.  
Detailed descriptions of each dataset, including column definitions and variable meanings, are provided in:

data-GPDEM-2025-IISETr/data_file_descriptions.txt


That text file contains explanations for:
- **Route Data Structure** — defining shipment routes, legs, and cost components.  
- **Lane Data Structure** — defining transportation arcs, costs, and transit times.  
- **Conversion Rate Data Structure** — defining customer purchase probabilities by lead time.

Please refer to `data_file_descriptions.txt` for complete details.

---

## Computational Environment

All runtimes were obtained under the computational conditions described in the accompanying paper.  
For consistent reproduction, ensure comparable hardware and solver configurations.



