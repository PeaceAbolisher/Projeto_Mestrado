import pandas as pd
import os

# File path to the filtered CSV file
filtered_csv_path = "C:/Users/Rafael Fonseca/Desktop/Mestrado/Ano2/ProjetoMestrado/code/data/filtered_proteome4_data.csv"

# File path to save the extracted Protein IDs
output_file = "C:/Users/Rafael Fonseca/Desktop/Mestrado/Ano2/ProjetoMestrado/code/data/LISTA.txt"

# Check if the input file exists
if os.path.exists(filtered_csv_path):
    try:
        # Read the CSV file into a DataFrame
        filtered_df = pd.read_csv(filtered_csv_path)

        # Check if the 'Protein ID' column exists
        if "Protein ID" in filtered_df.columns:
            # Extract all Protein IDs into a list
            protein_ids = filtered_df["Protein ID"].tolist()

            # Save the Protein IDs to the output file
            with open(output_file, 'a') as f:
                for pid in protein_ids:
                    f.write(pid + '\n')

            print(f"Protein IDs have been successfully extracted and saved to '{output_file}'.")
        else:
            print("The column 'Protein ID' was not found in the CSV file.")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
else:
    print(f"The file '{filtered_csv_path}' does not exist. Please check the file path.")
