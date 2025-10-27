# ODTQ-MMC±X Model Files

📅 **Last updated:** October 27, 2025  

---

## ⚠️ Notice: Model Files Coming Soon

The optimization model files described below will be uploaded shortly.  
All models and data will be available by **November 15, 2025**.  
Please check back after this date for full access to the complete set of files.

---

## Reference

This repository accompanies the paper:

> **Greening, Lacy M., et al.**  
> *Integrating Order-to-Delivery Time Sensitivity in E-Commerce Middle-Mile Consolidation Network Design.*  
> *IISE Transactions*, accepted 2025.

These models reproduce the numerical results reported in **Tables 2–15** and **Figures 4–6** of the study.

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

---

## Overview
 

The following table summarizes the data requirements, model files, expected outputs, and run times.

| Tables / Figures | Required Data Files | Model Files | Expected Output | Runtime Conditions |
|------------------|--------------------|--------------|-----------------|--------------------|
| **Tables 2–5; Figure 4** | Network data, customer choice data | `Binary ODTQ-MMC±X MIP` | Objective value, routes and lanes used, ODT selected | 12 hours |
| **Table 6; Figures 5–6** | Network data, customer choice data, previous solutions (for warm starts) | `ODTQ-MMC±X heuristic` | Objective value, routes and lanes used, ODT selected | 6 hours |
| **Table 7** | Network data, customer choice data | `ODTQ-MMC±0 MIP`; `ODTQ-MMC±3 MIP sensitivity` | Objective value, routes and lanes used, ODT selected | 6 hours |
| **Table 8** | Network data, customer choice data, previous solutions | `ODTQ-MMC±0 MIP`; `ODTQ-MMC±3 MIP sensitivity`; `ODTQ-MMC±3 MIP sensitivity – reject`; `ODTQ-MMC±3 MIP sensitivity – add LTL` | Objective value, routes and lanes used, ODT selected | 6 hours; *Add LTL* and *Reject* <10 s (routes & ODTs fixed) |
| **Table 11** | Network data, customer choice data | Binary and piecewise-linear `ODTQ-MMC±X MIP`; Binary and piecewise-linear `ODTQ-MMC±X heuristic` | Objective value, routes and lanes used, ODT selected | 12 hours |
| **Table 13** | Network data, customer choice data | Binary and piecewise-linear `ODTQ-MMC±X MIP`; Piecewise-linear `ODTQ-MMC±X heuristic` | Objective value, routes and lanes used, ODT selected | 1-, 3-, 6-, and 12-hour solutions |
| **Tables 14–15** | Network data, customer choice data, previous solutions | `Binary ODTQ-MMC±X MIP` | Objective value, routes and lanes used, ODT selected | 6 hours; restricted models <1 hour |


---

## Computational Environment

All runtimes were obtained under the computational conditions described in the accompanying paper.  
For consistent reproduction, ensure comparable hardware and solver configurations.



