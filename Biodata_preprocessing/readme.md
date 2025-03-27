# 🔬 Functional Gene Annotation & Abundance Workflow (KofamScan + Bowtie2)

This repository provides scripts and guidelines for annotating protein sequences with KEGG Orthologs (KOs) using **KofamScan**, and quantifying their abundance using **Bowtie2**-based RPKM calculation. It is particularly tailored for metagenomic samples with downstream support for **carbon and nitrogen metabolism pathway analysis**.

---

## 📁 Included Scripts

- `kofamscan.txt` — Bash script to run **KofamScan** for KO annotation.
- `kofamscan_bowtie_rpkm_merge.txt` — Bash script to compute **RPKM** using Bowtie2 and merge with KO annotations.
- `ko_rpkm_loop.py` — Python script to integrate KO and RPKM results across multiple samples.
- `extract_specific_pathway.py` — Python script to extract pathway-specific KO entries (e.g., carbon/nitrogen metabolism).

---

## ⚙️ Workflow Steps

1. **Prepare Input Files**  
   Concatenate metagenomic protein and nucleotide files to generate `.faa.xz` (protein) and `.fna.xz` (nucleotide) files.

2. **Decompress Protein Files**  
   Extract `.faa` files from `.faa.xz` using `xz`.

3. **KO Annotation Using KofamScan**  
   Execute `kofamscan.txt` to annotate KOs. This step leverages:
   - KO-specific **HMM score thresholds** from KEGG's `ko_list`
   - Post-filtering using **e-value ≤ 1e-10** to remove low-confidence matches

4. **Decompress Nucleotide Files**  
   Extract `.fna` files from `.fna.xz`.

5. **Extract Functional Genes**  
   Retrieve gene sequences (e.g., via Prodigal or gene mapping) for use in abundance calculation.

6. **RPKM Calculation with Bowtie2**  
   Align reads to functional genes and compute RPKM values. Use `kofamscan_bowtie_rpkm_merge.txt` to merge these with KO annotations.

7. **Integrate KO and RPKM**  
   Use `ko_rpkm_loop.py` to compile final KO-RPKM abundance tables for each sample.

8. **Extract Pathways of Interest**  
   Use `extract_specific_pathway.py` to subset data for target pathways such as **carbon** and **nitrogen metabolism**.

---

## 🛡️ Annotation Quality Control

| Criterion | Description |
|----------|-------------|
| **Score Threshold** | KofamScan applies KO-specific HMM score cutoffs from KEGG for precise annotations |
| **e-value Filtering** | Only matches with **e-value ≤ 1e-10** are retained |
| **RPKM Normalization** | Read counts are normalized by gene length and total mapped reads |

---

## ✅ Requirements

- [KofamScan](https://github.com/takaram/kofam_scan)
- [Bowtie2](http://bowtie-bio.sourceforge.net/bowtie2/index.shtml)
- Python ≥ 3.6
- `pandas`, `BioPython` for Python scripts

---

