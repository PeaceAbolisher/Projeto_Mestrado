#após visualizar o dataset eu percebi que as colunas de taxon_rank_level e relative_abundance estavam trocadas (quando se faz download). Os resultados de taxon_rank_level têm a relative_abundance
#e relative_abundance tem os resultados de taxon_rank_level (species visto que tudo o que foi downloaded é informação de espécies)
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings

healthy_data_path = r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\data\healthy_samples"

healthy_data = pd.DataFrame()

for filename in os.listdir(healthy_data_path):
    if filename.startswith("relative_abundance_for_curated") and filename.endswith(".txt"):
        file_path = os.path.join(healthy_data_path, filename)

        df = pd.read_csv(file_path, sep="\t", comment="#")

        # Mudar as Colunas para não criar confusão
        df = df.rename(columns={"taxon_rank_level": "relative_abundance", "relative_abundance": "taxon_rank_level"})

        df = df[["ncbi_taxon_id", "relative_abundance", "scientific_name"]]

        df["sample"] = filename  

        healthy_data = pd.concat([healthy_data, df], ignore_index=True)

healthy_data.to_csv("healthy_microbiome_data.csv", index=False)

# Display the first few rows
print(healthy_data)


unique_taxa = healthy_data["ncbi_taxon_id"].nunique()
print(f"Number of unique ncbi_taxon_id values: {unique_taxa}")

#7759 entradas, com 345 sendo únicas quer dizer que muitas são repetidas

num_negative_ones = (healthy_data["ncbi_taxon_id"] == -1).sum()
print(f"Number of rows with ncbi_taxon_id = -1: {num_negative_ones}")


#remover as entradas com -1 e guardar os dados limpos
healthy_data_cleaned = healthy_data[healthy_data["ncbi_taxon_id"] != -1]

# Print the number of entries with -1 after removal
num_negative_ones_cleaned = (healthy_data_cleaned["ncbi_taxon_id"] == -1).sum()
print(f"Number of rows with ncbi_taxon_id = -1 after cleaning: {num_negative_ones_cleaned}")

# Save the cleaned data
cleaned_file_path = os.path.join(healthy_data_path, "cleaned_healthy_microbiome_data.csv")
healthy_data_cleaned.to_csv(cleaned_file_path, index=False)


healthy_data_cleaned['relative_abundance'] = pd.to_numeric(healthy_data_cleaned['relative_abundance'], errors='coerce')

#Agregar dados por abundância
aggregated_data = healthy_data_cleaned.groupby("scientific_name")["relative_abundance"].sum().reset_index()

aggregated_data = aggregated_data.sort_values(by="relative_abundance", ascending=False)

top_20 = aggregated_data.head(20)

# Plot
plt.figure(figsize=(12, 10))
plt.pie(top_20["relative_abundance"], labels=top_20["scientific_name"], autopct='%1.1f%%', startangle=140, pctdistance=0.85)
plt.title("Top 20 Most Abundant Species Across All Healthy Samples")
plt.axis('equal')  # Ensures the pie chart is drawn as a circle
plt.show()


#contar nr entradas
species_count = healthy_data_cleaned['scientific_name'].value_counts().reset_index()
species_count.columns = ['scientific_name', 'count']

top_50_count = species_count.head(20)

#Plot
plt.figure(figsize=(14, 10))
plt.barh(top_50_count['scientific_name'], top_50_count['count'])
plt.xlabel('Number of Entries')
plt.ylabel('Species')
plt.title('Top 50 Most Common Species by Number of Entries for Healthy Samples')

#visualizacao
plt.xlim(0, top_50_count['count'][1:].max() + 20)
plt.gca().invert_yaxis()

for i, v in enumerate(top_50_count['count']):
    plt.text(top_50_count['count'][1:].max() + 5, i, str(v), color='black', va='center')

plt.show()

#------------------------------------------------------- DIABETIC ------------------------------------------------------------
diabetic_data_path = r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\data\diabetic_samples"

diabetic_data = pd.DataFrame()

for filename in os.listdir(diabetic_data_path):
    if filename.startswith("relative_abundance_for_curated") and filename.endswith(".txt"):
        file_path = os.path.join(diabetic_data_path, filename)

        df = pd.read_csv(file_path, sep="\t", comment="#")

        # Mudar as Colunas para não criar confusão
        df = df.rename(columns={"taxon_rank_level": "relative_abundance", "relative_abundance": "taxon_rank_level"})

        df = df[["ncbi_taxon_id", "relative_abundance", "scientific_name"]]

        df["sample"] = filename  

        diabetic_data = pd.concat([diabetic_data, df], ignore_index=True)

diabetic_data.to_csv("diabetic_microbiome_data.csv", index=False)

# Display the first few rows
print(diabetic_data)

unique_taxa = diabetic_data["ncbi_taxon_id"].nunique()
print(f"Number of unique ncbi_taxon_id values: {unique_taxa}")

num_negative_ones = (diabetic_data["ncbi_taxon_id"] == -1).sum()
print(f"Number of rows with ncbi_taxon_id = -1: {num_negative_ones}")

# Remover as entradas com -1
diabetic_data_cleaned = diabetic_data[diabetic_data["ncbi_taxon_id"] != -1]

num_negative_ones_cleaned = (diabetic_data_cleaned["ncbi_taxon_id"] == -1).sum()
print(f"Number of rows with ncbi_taxon_id = -1 after cleaning: {num_negative_ones_cleaned}")

cleaned_file_path = os.path.join(diabetic_data_path, "cleaned_diabetic_microbiome_data.csv")
diabetic_data_cleaned.to_csv(cleaned_file_path, index=False)

diabetic_data_cleaned['relative_abundance'] = pd.to_numeric(diabetic_data_cleaned['relative_abundance'], errors='coerce')

aggregated_data = diabetic_data_cleaned.groupby("scientific_name")["relative_abundance"].sum().reset_index()
aggregated_data = aggregated_data.sort_values(by="relative_abundance", ascending=False)

top_20 = aggregated_data.head(20)

plt.figure(figsize=(12, 10))
plt.pie(top_20["relative_abundance"], labels=top_20["scientific_name"], autopct='%1.1f%%', startangle=140, pctdistance=0.85)
plt.title("Top 20 Most Abundant Species Across All Diabetic Samples")
plt.axis('equal')
plt.show()

species_count = diabetic_data_cleaned['scientific_name'].value_counts().reset_index()
species_count.columns = ['scientific_name', 'count']

top_50_count = species_count.head(20)

plt.figure(figsize=(14, 10))
plt.barh(top_50_count['scientific_name'], top_50_count['count'])
plt.xlabel('Number of Entries')
plt.ylabel('Species')
plt.title('Top 50 Most Common Species by Number of Entries')
plt.xlim(0, top_50_count['count'][1:].max() + 20)
plt.gca().invert_yaxis()

for i, v in enumerate(top_50_count['count']):
    plt.text(top_50_count['count'][1:].max() + 5, i, str(v), color='black', va='center')

plt.show()
