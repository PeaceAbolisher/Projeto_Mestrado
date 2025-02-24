import pandas as pd

diabetic_dataset = pd.read_csv(r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\data\diabetic_samples\cleaned_diabetic_microbiome_data.csv")
healthy_dataset = pd.read_csv(r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\data\healthy_samples\cleaned_healthy_microbiome_data.csv")

# Add labels
diabetic_dataset['healthy'] = 1
healthy_dataset['healthy'] = 0

all_data = pd.concat([diabetic_dataset, healthy_dataset], ignore_index=True)

#droppar sample e nome cientifico
columns_to_drop = ['sample', 'scientific_name']
all_data = all_data.drop(columns=[col for col in columns_to_drop if col in all_data.columns])

all_data.to_csv(r"C:\Users\Rafael Fonseca\Desktop\Mestrado\Ano2\ProjetoMestrado\parte_2\data\all_data_samples\all_data_samples.csv", index=False)

print(all_data.head(20))
