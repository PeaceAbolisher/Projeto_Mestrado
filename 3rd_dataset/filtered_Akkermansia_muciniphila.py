import pandas as pd
import re

# File path to the filtered CSV file
filtered_csv_path = "C:/Users/Rafael Fonseca/Desktop/Mestrado/Ano2/ProjetoMestrado/code/data/filtered_proteome_data.csv"

# Read the CSV file into a DataFrame
filtered_df = pd.read_csv(filtered_csv_path)

# Initialize a list to store the extracted Protein IDs
protein_ids = []

# Extract the text inside | | from the "Protein ID" column using regex
if "Protein ID" in filtered_df.columns:
    for protein_id in filtered_df["Protein ID"]:
        match = re.search(r'\|([^|]+)\|', protein_id)
        if match:
            protein_ids.append(match.group(1))  # Extracted Protein ID

    # Display the extracted Protein IDs
    print("Extracted Protein IDs:")
    for pid in protein_ids:
        print(pid)

    # Save the extracted Protein IDs to a file named "LISTA.txt"
    output_file = "C:/Users/Rafael Fonseca/Desktop/Mestrado/Ano2/ProjetoMestrado/code/data/LISTA.txt"
    with open(output_file, 'a') as f:
        for pid in protein_ids:
            f.write(pid + '\n')

    print(f"\nProtein IDs have been saved to '{output_file}'.")
else:
    print("The column 'Protein ID' was not found in the CSV file.")
