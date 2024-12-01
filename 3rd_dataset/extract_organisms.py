import pandas as pd

# File path to BioGRID dataset
file_path = 'C:/Users/Rafael Fonseca/Desktop/Mestrado/Ano2/ProjetoMestrado/code/data/BIOGRID-ALL-4.4.239.tab3.txt'

# Define column names (BioGRID files often lack headers)
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

data = pd.read_csv(file_path, sep='\t', comment='#', header=None, names=columns, low_memory=False)

# Step 1: Identify unique organism IDs and names, dropping duplicates
organism_data_a = data[['Organism ID Interactor A', 'Organism Name Interactor A']].drop_duplicates()
organism_data_a = organism_data_a.rename(columns={'Organism ID Interactor A': 'Organism ID', 'Organism Name Interactor A': 'Organism Name'})

organism_data_b = data[['Organism ID Interactor B', 'Organism Name Interactor B']].drop_duplicates()
organism_data_b = organism_data_b.rename(columns={'Organism ID Interactor B': 'Organism ID', 'Organism Name Interactor B': 'Organism Name'})

# Combine and deduplicate organism data
all_organism_data = pd.concat([organism_data_a, organism_data_b]).drop_duplicates().sort_values(by='Organism ID')
all_organism_data.to_csv('data/unique_organisms.csv', index=False)
print("Unique organism IDs and names saved to 'unique_organisms.csv'.")