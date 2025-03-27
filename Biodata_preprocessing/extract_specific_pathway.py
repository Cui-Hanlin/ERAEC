import os
import pandas as pd
import re

# Define the folder path where the CSV files are located
folder_path = r'F:\ERAEC_Project\tests\merged_kegg'
output_file = r"F:\ERAEC_Project\tests\carbon_nitrogen_cycling_aggregated.xlsx"

# Pathways related to carbon and nitrogen cycling
carbon_keywords = [
    r"carbon", r"photosynthesis", r"citrate", r"glucose", r"fatty acid", r"pyruvate", r"TCA cycle", r"glyoxylate", r"glycolysis", r"organic", r"carbohydrate", r"methan", r"carboxyl", r"yl-coa",r"serine", r"cysteine"
]
nitrogen_keywords = [
    r"nitrogen", r"nitrification", r"ammonia", r"nitrous", r"nitrate", r"nitrite",r"cyanate", r"comammox", r"anammox", r"nitric oxide",r"hydrazine",r"hydroxylamine", r"denitrification", r"ammonification", r"amino acid"
]

# KO and EC numbers related to carbon and nitrogen cycling
carbon_ko_numbers = ["K00198", "K01601"]  # Add more known KOs for carbon cycling
nitrogen_ko_numbers = ["K02586", "K00370"]  # Add more known KOs for nitrogen cycling

carbon_ec_numbers = ["1.1.1.1", "4.1.1.39"]  # Replace with more EC numbers related to carbon cycling
nitrogen_ec_numbers = ["1.18.6.1", "1.7.99.4"]  # Replace with more EC numbers related to nitrogen cycling

# Combine KO and EC numbers
all_ko_numbers = carbon_ko_numbers + nitrogen_ko_numbers
all_ec_numbers = carbon_ec_numbers + nitrogen_ec_numbers

# Function to filter based on keywords and KO/EC numbers
def filter_dataframe(df):
    # Filter based on Level2 or Level3 keywords
    keyword_filtered = df[
        df['Level2'].str.contains('|'.join(carbon_keywords + nitrogen_keywords), case=False, na=False) |
        df['Level3'].str.contains('|'.join(carbon_keywords + nitrogen_keywords), case=False, na=False)
    ]

    # Filter based on KO and EC numbers
    ko_filtered = df[df['KO'].isin(all_ko_numbers)]
    ec_filtered = df[df['ec'].isin(all_ec_numbers)]

    # Combine all filtered results and drop duplicates
    combined_filtered = pd.concat([keyword_filtered, ko_filtered, ec_filtered]).drop_duplicates()

    return combined_filtered

# Initialize an empty list to store data for each sample
results = []

# Loop through CSV files starting with 'sample_'
for filename in os.listdir(folder_path):
    if filename.startswith("sample_") and filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        try:
            # Load the CSV file
            df = pd.read_csv(file_path)

            # Ensure required columns are present
            required_columns = {'Level2', 'Level3', 'KO', 'ec', 'RPKM'}
            if not required_columns.issubset(df.columns):
                print(f"Skipping {filename}: Missing required columns")
                continue

            # Extract the sample ID from the filename (assuming the sample ID is part of the filename)
            sample_id = filename.split('_')[1].split('.')[0]

            # Perform aggregation to group by Level3 and sum RPKM values
            df_agg = df.groupby("Level3")["RPKM"].sum().reset_index()

            # Filter the dataframe
            filtered_df = filter_dataframe(df)

            # Re-aggregate by Level3 after filtering
            combined_filtered_agg = filtered_df.groupby("Level3")["RPKM"].sum().reset_index()

            # Add sample ID and create a row of results
            combined_filtered_agg.insert(0, 'sample_id', sample_id)

            # Append the row to the results list
            results.append(combined_filtered_agg)

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Concatenate all results into a single DataFrame
if results:
    final_results_df = pd.concat(results, ignore_index=True)

    # Pivot the DataFrame to have one row per sample and Level3 functions as columns
    final_pivot_df = final_results_df.pivot(index='sample_id', columns='Level3', values='RPKM').reset_index()
    # Fill NaN values with zero in the pivoted DataFrame
    final_pivot_df.fillna(0, inplace=True)

    # Save the final DataFrame to an Excel file
    final_pivot_df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")
else:
    print("No valid data found to process.")

#AY