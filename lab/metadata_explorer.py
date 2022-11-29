import pandas as pd

from src.config import config

df = pd.read_excel(config['metadata_path'])
species_names = df['Espèce cible'].unique()
print(*species_names, sep='\n')