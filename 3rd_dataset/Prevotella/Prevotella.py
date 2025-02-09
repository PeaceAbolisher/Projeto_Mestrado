import pandas as pd
import os

# File path to the FASTA file
file_path = "C:/Users/Rafael Fonseca/Desktop/Mestrado/Ano2/ProjetoMestrado/code/data/proteome5_data.fasta"

# Function to parse the FASTA file
def parse_fasta(file_path):
    protein_data = []
    try:
        with open(file_path, 'r') as file:
            protein_id = None
            description = None
            sequence = []

            for line in file:
                line = line.strip()
                if line.startswith(">"):
                    if protein_id:
                        # Save the previous protein's data
                        protein_data.append({
                            "Protein ID": protein_id,
                            "Description": description,
                            "Sequence": ''.join(sequence)
                        })
                    # Start a new protein entry
                    header_parts = line.split(maxsplit=1)
                    protein_id = header_parts[0][1:]  # Remove the ">" character
                    description = header_parts[1] if len(header_parts) > 1 else ""
                    sequence = []
                else:
                    sequence.append(line)

            # Add the last protein's data
            if protein_id:
                protein_data.append({
                    "Protein ID": protein_id,
                    "Description": description,
                    "Sequence": ''.join(sequence)
                })

        return pd.DataFrame(protein_data)

    except FileNotFoundError:
        print(f"File not found: {file_path}. Please check the path and try again.")
        return pd.DataFrame()  # Return an empty DataFrame if the file is not found

# Function to extract additional details from the description
def parse_description(desc):
    details = {}
    if "OS=" in desc:
        details["Organism"] = desc.split("OS=")[1].split(" ")[0]
    if "GN=" in desc:
        details["Gene Name"] = desc.split("GN=")[1].split(" ")[0]
    if "PE=" in desc:
        details["Protein Existence"] = desc.split("PE=")[1].split(" ")[0]
    if "SV=" in desc:
        details["Sequence Version"] = desc.split("SV=")[1].split(" ")[0]
    return details

# Main script logic
if os.path.exists(file_path):
    # Parse the FASTA file
    protein_df = parse_fasta(file_path)

    # If the DataFrame is empty, exit
    if protein_df.empty:
        print("No data to process. Exiting.")
    else:
        # Add parsed details to the DataFrame
        parsed_details = protein_df["Description"].apply(parse_description)
        details_df = pd.json_normalize(parsed_details).fillna("")

        # Combine all relevant columns
        full_protein_df = pd.concat([protein_df, details_df], axis=1)

        # Filter proteins where Protein Existence (PE) is 1, 2, or 3
        filtered_proteins = full_protein_df[full_protein_df["Protein Existence"].isin(["1", "2", "3"])]

        # Save to a new CSV file
        csv_output_path = "C:/Users/Rafael Fonseca/Desktop/Mestrado/Ano2/ProjetoMestrado/code/data/filtered_proteome5_data.csv"
        filtered_proteins.to_csv(csv_output_path, index=False)

        print(f"Filtered proteins saved to '{csv_output_path}'.")
else:
    print(f"File not found: {file_path}. Please check the path and try again.")