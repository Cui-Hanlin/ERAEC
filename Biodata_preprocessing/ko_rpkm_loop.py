import os
import pandas as pd

base_output_path = "D:/ERAEC_Project/tests/output"
rpkm_base_path = "D:/ERAEC_Project/datasets/2_assembly_binning"
ko_pathway_file = "D:/ERAEC_Project/datasets/kegg_pathway_uniq_1.csv"
merged_output_dir = "D:/ERAEC_Project/tests/merged_kegg"

os.makedirs(merged_output_dir, exist_ok=True)

def process_sample_folder(sample_name):
    print(f"Processing sample: {sample_name}")

    annotation_folder = f"{base_output_path}/{sample_name}_protein_tmp"  # Adjust for _protein_tmp
    annotation_file_tsv = f"{annotation_folder}/{sample_name}_protein_kegg_filtered.tsv"
    annotation_file_csv = f"{annotation_folder}/{sample_name}_protein_annotation.csv"
    rpkm_file = f"{rpkm_base_path}/{sample_name}.rpkm"
    output_file = f"{merged_output_dir}/{sample_name}_merged_kegg.csv"
    
    if not os.path.exists(annotation_file_tsv) or not os.path.exists(rpkm_file):
        print(f"Skipping {sample_name}: missing required files.")
        return

    # 1. Change the annotation file from TSV to CSV with separated columns
    annotation_data = []
    with open(annotation_file_tsv, 'r') as f:
        for line in f:
            if line.startswith('*'):  # Filter rows starting with asterisk
                parts = line.split()  # Splitting on whitespace
                if len(parts) >= 6:
                    annotation_data.append(parts[:6])  # Only keep first 6 columns

    # Convert to DataFrame and save as CSV
    annotation_df = pd.DataFrame(annotation_data, columns=['indicator', 'gene_name', 'KO', 'threshold', 'score', 'E-value'])
    annotation_df = annotation_df.drop(columns=['indicator'])  # Drop the 'indicator' column
    annotation_df.to_csv(annotation_file_csv, index=False)

    # 2. Standardize the gene_name by removing suffixes for sub_units
    annotation_df['gene_name'] = annotation_df['gene_name'].str.replace(r'_\d+$', '', regex=True)

    # 3. Load the RPKM data and rename #Name column to gene_name for merging
    rpkm_df = pd.read_csv(rpkm_file, sep='\t', header=0, names=['gene_name', 'Length', 'Reads', 'RPKM'])
    rpkm_df = rpkm_df[['gene_name', 'RPKM']]  # Only keep gene_name and RPKM

    # Merge KO annotations with RPKM values based on gene names
    merged_df = pd.merge(annotation_df[['gene_name', 'KO']], rpkm_df, on='gene_name', how='inner')

    # Group by gene_name and KO, summing the RPKM values to avoid duplicates
    merged_df = merged_df.groupby(['gene_name', 'KO'], as_index=False).agg({'RPKM': 'sum'})

    # 4. Load KO-to-KEGG Pathway mapping file
    kegg_pathway_df = pd.read_csv(ko_pathway_file, sep=',', header=0, names=['KO', 'ko_id', 'Gene_name', 'Level1_id', 'Level1', 'Level2_id', 'Level2', 'Level3_id', 'Level3', 'ko_des', 'ec'])

    # Merge KO-RPKM data with KO-to-KEGG Pathway mapping, including 'ec'
    merged_kegg_df = pd.merge(merged_df, kegg_pathway_df[['KO', 'Level1', 'Level2', 'Level3', 'ec']], on='KO', how='inner')

    # 5. Save the merged data to CSV
    merged_kegg_df.to_csv(output_file, sep=',', index=False)

    # Summarize RPKM values by KEGG Level 1, 2, and 3 pathways
    level1_summary = merged_kegg_df.groupby('Level1')['RPKM'].sum().reset_index()
    level2_summary = merged_kegg_df.groupby('Level2')['RPKM'].sum().reset_index()
    level3_summary = merged_kegg_df.groupby('Level3')['RPKM'].sum().reset_index()

    # Save summaries to CSV files
    level1_summary.to_csv(f"{merged_output_dir}/{sample_name}_level1_abundance.csv", sep=',', index=False)
    level2_summary.to_csv(f"{merged_output_dir}/{sample_name}_level2_abundance.csv", sep=',', index=False)
    level3_summary.to_csv(f"{merged_output_dir}/{sample_name}_level3_abundance.csv", sep=',', index=False)

    print(f"Finished processing {sample_name}")

# Loop through the subfolders in the base_output_path directory
for folder_name in os.listdir(base_output_path):
    sample_folder_path = os.path.join(base_output_path, folder_name)
    if os.path.isdir(sample_folder_path) and folder_name.endswith("_protein_tmp"):  # Process only folders with '_protein_tmp' in the name
        sample_name = folder_name.replace("_protein_tmp", "")  # Extract sample name without _protein_tmp
        process_sample_folder(sample_name)

print("All samples processed!")


