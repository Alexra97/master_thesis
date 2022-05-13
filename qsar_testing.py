# Cargar las librerias
import sys
import pandas as pd
from qsar_pckg import qsar_modelling as qm
from qsar_pckg import chembl_screening as cs


# Cargar los datos del script main

## Importar los modelos
rf_max, svm_max, nb_max, xgb_max, ann_max = qm.loadModels("_max_DNMT1")
rf_mean, svm_mean, nb_mean, xgb_mean, ann_mean = qm.loadModels("_mean_DNMT1")
rf_max3, svm_max3, nb_max3, xgb_max3, ann_max3 = qm.loadModels("_max_DNMT3")
rf_mean3, svm_mean3, nb_mean3, xgb_mean3, ann_mean3 = qm.loadModels("_mean_DNMT3")
rf_maxO, svm_maxO, nb_maxO, xgb_maxO, ann_maxO = qm.loadModels("_max_Oleacin")
rf_meanO, svm_meanO, nb_meanO, xgb_meanO, ann_meanO = qm.loadModels("_mean_Oleacin")

## Importar los conjuntos de test
test_max_X, test_max_Y, test_mean_X, test_mean_Y = qm.loadTestData("_DNMT1")
test3_max_X, test3_max_Y, test3_mean_X, test3_mean_Y = qm.loadTestData("_DNMT3")
testO_max_X, testO_max_Y, testO_mean_X, testO_mean_Y = qm.loadTestData("_Oleacin")

## Cargar los datos para el semáforo de la similaridad
colnames_dice = ['PubChem ID', 'Dice Similarity']
colnames_tani = ['PubChem ID', 'Tanimoto Similarity']
dice_mean = pd.read_csv(sys.argv[3])
tani_mean = pd.read_csv(sys.argv[4])
dice3_mean = pd.read_csv(sys.argv[5])
tani3_mean = pd.read_csv(sys.argv[6])
diceO_max = pd.read_csv(sys.argv[7])
taniO_max = pd.read_csv(sys.argv[8])
dice_mean.columns = dice3_mean.columns = diceO_max.columns = colnames_dice
tani_mean.columns = tani3_mean.columns = taniO_max.columns = colnames_tani

## Combinar los datos de similaridad para cada set
sim_mean = pd.merge(dice_mean, tani_mean, on="PubChem ID")
sim_mean3 = pd.merge(dice3_mean, tani3_mean, on="PubChem ID")
sim_maxO = pd.merge(diceO_max, taniO_max, on="PubChem ID")


# Evaluar los modelos

## Crear listas de los modelos para iterarlos
models_max = [rf_max, svm_max, nb_max, xgb_max, ann_max]
models_mean = [rf_mean, svm_mean, nb_mean, xgb_mean, ann_mean]
models_max3 = [rf_max3, svm_max3, nb_max3, xgb_max3, ann_max3]
models_mean3 = [rf_mean3, svm_mean3, nb_mean3, xgb_mean3, ann_mean3]
models_maxO = [rf_maxO, svm_maxO, nb_maxO, xgb_maxO, ann_maxO]
models_meanO = [rf_meanO, svm_meanO, nb_meanO, xgb_meanO, ann_meanO]

## Evaluar y exportar las métricas
qm.QSARtesting(models_max, models_mean, test_max_X, test_max_Y, test_mean_X, test_mean_Y, "_DNMT1")
qm.QSARtesting(models_max3, models_mean3, test3_max_X, test3_max_Y, test3_mean_X, test3_mean_Y, "_DNMT3")
qm.QSARtesting(models_maxO, models_meanO, testO_max_X, testO_max_Y, testO_mean_X, testO_mean_Y, "_Oleacin")


# Cargar y transformar los datos del estudio por Docking

## Cargar los datos de Docking
dock_data = pd.read_excel(sys.argv[1])

## Obtener los códigos SMILES
data_dnmt1_dock = cs.getSmiles(dock_data, 1)
data_dnmt1_dock.to_csv('data_dnmt1_dock.csv', index=False)

## Cargar los datos ya estandárizados
data_dnmt1_dock = pd.read_csv(sys.argv[2], sep='\t', index_col=[0])
data_dnmt1_dock = data_dnmt1_dock[["compound", "standardized_molecule", "properties_log"]]

## Eliminación de las moléculas incorrectas
data_dnmt1_dock = cs.molPropCorrect(data_dnmt1_dock)
data_dnmt1_dock.drop(["properties_log"], axis=1, inplace=True)
data_dnmt1_dock.to_csv('data_dnmt1_dock_std.csv', index=False)

## Obtener los fingerprints de los códigos SMILE
mols_dnmt1_dock = qm.getMols(data_dnmt1_dock[["compound", "standardized_molecule"]])
data_dnmt1_dock = qm.getMorganFP(mols_dnmt1_dock)
data_dnmt1_dock.rename(columns = {'Molecule ChEMBL ID':'PubChem ID'}, inplace = True)


# Evaluar los datos de Docking con el mejor modelo de cada caso de estudio

## Predecir sobre el conjunto de test
pred_dnmt1_dock = svm_mean.predict(data_dnmt1_dock.drop(["PubChem ID"], axis=1))
pred_dnmt3_dock = rf_mean3.predict(data_dnmt1_dock.drop(["PubChem ID"], axis=1))
pred_dnmtO_dock = rf_maxO.predict(data_dnmt1_dock.drop(["PubChem ID"], axis=1))

## Crear un dataframe con las predicciones y su similaridad
semaphore_dnmt1 = pd.merge(pd.DataFrame({'PubChem ID' : data_dnmt1_dock["PubChem ID"], 'Activity' : pred_dnmt1_dock}), sim_mean, on='PubChem ID')
semaphore_dnmt3 = pd.merge(pd.DataFrame({'PubChem ID' : data_dnmt1_dock["PubChem ID"], 'Activity' : pred_dnmt3_dock}), sim_mean3, on='PubChem ID')
semaphore_dnmtO = pd.merge(pd.DataFrame({'PubChem ID' : data_dnmt1_dock["PubChem ID"], 'Activity' : pred_dnmtO_dock}), sim_maxO, on='PubChem ID')

# Exportar los resultados finales
semaphore_dnmt1.to_csv("results_dnmt1.csv", index=False)
semaphore_dnmt3.to_csv("results_dnmt3.csv", index=False)
semaphore_dnmtO.to_csv("results_O.csv", index=False)








