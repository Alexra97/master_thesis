# Cargar las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem

### Función que obtiene el listado de moléculas a partir de sus smiles (conservando su ID) ###
def getMols(smiles_df):
    return [(ids, Chem.MolFromSmiles(sml)) for ids, sml in smiles_df.itertuples(index=False)]

### Función que obtiene los Fingerprints de una lista de moléculas y los devuelve como DataFrame (conservando su ID) ###   
def getMorganFP(mols_df):
    # Obtener los Fingerprints + ID en una lista de Series
    m_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    morgan_FPs = [pd.concat([pd.Series(ids), pd.Series(np.array(m_generator.GetFingerprint(mol)))], ignore_index=True) for ids, mol in mols_df]
    
    # Generar los nombres de las columnas
    names = [f'FP_{n}' for n in range(2048)]
    names.insert(0, "Molecule ChEMBL ID")
    
    # Generar el DataFrame final con los datos en forma de columnas
    df_morgan_FPs = pd.DataFrame(data=morgan_FPs)
    df_morgan_FPs.set_axis(names, axis=1, inplace=True)
    return df_morgan_FPs

### Función que crea dos subconjuntos con un determinado tamaño para un dataframe ###   
def subsetData(df, pcent, seed):
    train = df.sample(frac=pcent, random_state=seed)
    test = df.drop(train.index)
    return train, test
