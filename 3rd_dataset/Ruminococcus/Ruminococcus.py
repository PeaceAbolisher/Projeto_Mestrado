import pandas as pd

# Function to parse the FASTA file and extract Protein IDs along with sequences
def parse_fasta_and_extract_ids(file_path):
    protein_data = []
    try:
        with open(file_path, 'r') as file:
            protein_id = None
            sequence = []
            
            for line in file:
                line = line.strip()
                if line.startswith(">"):
                    if protein_id:
                        # Save the previous protein's data
                        protein_data.append({
                            "Protein ID": protein_id,
                            "Sequence": ''.join(sequence)
                        })
                    # Extract the Protein ID from the header line
                    protein_id = line.split()[0][1:]  # Remove the ">" character
                    sequence = []
                else:
                    sequence.append(line)
            
            # Add the last protein's data
            if protein_id:
                protein_data.append({
                    "Protein ID": protein_id,
                    "Sequence": ''.join(sequence)
                })

        # Return the protein data as a DataFrame
        return pd.DataFrame(protein_data)

    except Exception as e:
        print(f"Error processing file: {e}")
        return pd.DataFrame()

# Specify the file path to the FASTA file
file_path = "C:/Users/Rafael Fonseca/Desktop/Mestrado/Ano2/ProjetoMestrado/code/data/proteome4_data.fasta"

# Parse the FASTA file
protein_df = parse_fasta_and_extract_ids(file_path)

# Specify the output CSV file path
output_csv_path = "C:/Users/Rafael Fonseca/Desktop/Mestrado/Ano2/ProjetoMestrado/code/data/filtered_proteome4_data.csv"

# Save the DataFrame to a CSV file
protein_df.to_csv(output_csv_path, index=False)

# Print success message
print(f"Protein data has been extracted and saved to '{output_csv_path}'.")
