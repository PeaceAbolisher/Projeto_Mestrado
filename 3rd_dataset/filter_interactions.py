import pandas as pd

dataset_path = 'C:/Users/Rafael Fonseca/Desktop/Mestrado/Ano2/ProjetoMestrado/code/data/BIOGRID-ALL-4.4.239.tab3.txt'
organism_ids_path = 'data/unique_organisms.csv'  # Output of extract_organisms.py

verified_organisms = pd.read_csv(organism_ids_path)

# Automatically filter gut microbiome IDs based on known gut-related keywords
gut_keywords = ['Bacteroides', 'Lactobacillus', 'Clostridium', 'Faecalibacterium', 'Escherichia', 'gut']
gut_microbiome_ids = verified_organisms[
    verified_organisms['Organism Name'].str.contains('|'.join(gut_keywords), case=False, na=False)
]['Organism ID'].dropna().tolist()

print(f"Identified gut microbiome taxonomy IDs: {gut_microbiome_ids}")

# Load the full dataset for filtering
columns = [
    "BioGRID Interaction ID", "Entrez Gene Interactor A", "Entrez Gene Interactor B",
    "BioGRID ID Interactor A", "BioGRID ID Interactor B", "Systematic Name Interactor A",
    "Systematic Name Interactor B", "Official Symbol Interactor A", "Official Symbol Interactor B",
    "Synonyms Interactor A", "Synonyms Interactor B", "Experimental System", "Experimental System Type",
    "Author", "Publication Source", "Organism ID Interactor A", "Organism ID Interactor B",
    "Throughput", "Score", "Modification", "Qualifications", "Tags", "Source Database",
    "SWISS-PROT Accessions Interactor A", "TREMBL Accessions Interactor A",
    "REFSEQ Accessions Interactor A", "SWISS-PROT Accessions Interactor B",
    "TREMBL Accessions Interactor B", "REFSEQ Accessions Interactor B", "Ontology Term IDs",
    "Ontology Term Names", "Ontology Term Categories", "Ontology Term Qualifier IDs",
    "Ontology Term Qualifier Names", "Ontology Term Types", "Organism Name Interactor A",
    "Organism Name Interactor B"
]

data = pd.read_csv(dataset_path, sep='\t', comment='#', header=None, names=columns, low_memory=False)

# Filter for gut microbiome interactions with the rest of the human body
print("Filtering for gut microbiome interactions with the rest of the human body...")
gut_human_body_interactions = data[
    ((data["Organism ID Interactor A"].isin(gut_microbiome_ids)) & (data["Organism ID Interactor B"] == 9606)) |
    ((data["Organism ID Interactor B"].isin(gut_microbiome_ids)) & (data["Organism ID Interactor A"] == 9606))
]

print(f"Found {len(gut_human_body_interactions)} interactions between the gut microbiome and the rest of the human body.")

print("Dataset is ready for analysis.")
print(gut_human_body_interactions.head())
