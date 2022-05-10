# Cargar las librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
from tensorflow import keras
from keras.layers import Dense
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from keras.wrappers.scikit_learn import KerasClassifier
import pickle

### Función que divide un dataframe en dos conjuntos en proporción 80-20 ###
def splitData(df, filename):
    # Obtener la variable respuesta y las descriptoras
    X = df.drop(["Activity"], axis=1)
    Y = df["Activity"].cat.codes
    
    # Dividir en train y test
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, stratify=Y)
    
    # Almacenar los SMILES de entrenamiento
    train_X.join(train_Y.to_frame()).to_csv(filename+'.csv', index=False)
    
    return train_X, test_X, train_Y, test_Y, X, Y

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

### Función que obtiene los Fingerprints para el conjunto de train y el de test ###
def getTrainTestFP(train_X, test_X, X):
    # Obtener las moléculas del conjunto
    mols_train = getMols(train_X)
    mols_test = getMols(test_X)
    
    # Obtener los fingerprints de estas moléculas
    fingerprints_train = getMorganFP(mols_train)
    fingerprints_test = getMorganFP(mols_test)
    
    # Actualizar los conjuntos
    train_X = pd.merge(fingerprints_train, X, on='Molecule ChEMBL ID').drop(["Molecule ChEMBL ID", "standardized_molecule"], axis=1)
    test_X = pd.merge(fingerprints_test, X, on='Molecule ChEMBL ID').drop(["Molecule ChEMBL ID", "standardized_molecule"], axis=1) 
    
    return train_X, test_X

### Función que genera una red neuronal artificial con los parámetros establecidos ###
def buildANN(n):
    model = keras.Sequential([
        Dense(n, activation='relu', input_shape=(2048,)),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="SGD", loss='binary_crossentropy', metrics=['accuracy'])
    return model

### Función que devuelve las métricas obtenidas a partir de la predicción de un modelo ###
def getQualityMetrics(y, pred):
    # Obtener la sensibilidad y la especificidad
    tn, fp, fn, tp = metrics.confusion_matrix(y, pred).ravel()
    sens = round(tp / (tp+fn),2)
    spec = round(tn / (tn+fp),2)
    
    # Obtener el valor del área bajo la curva ROC
    fpr, tpr, _ = metrics.roc_curve(y, pred, pos_label=1)
    auc = round(metrics.auc(fpr, tpr),2)
    
    # Obtener las demás métricas
    fm = round(metrics.f1_score(y, pred),2)
    ba = round(metrics.balanced_accuracy_score(y, pred),2)
    
    # Devolver una fila con los datos
    return pd.Series([ba, sens, spec, round(tp/(tp+fp),2), round(fp/(tp+fp),2), fm, auc])

### Función que devuelve el porcentaje de True Positives respecto al total de positivos ###
def tp_score(y, pred):
    _, fp, _, tp = metrics.confusion_matrix(y, pred).ravel()
    total = tp+fp
    if (total == 0): return 0
    else: return round(tp/total,2)

### Función que devuelve el porcentaje de False Positives respecto al total de positivos ###
def fp_score(y, pred):
    _, fp, _, tp = metrics.confusion_matrix(y, pred).ravel()
    total = tp+fp
    if (total == 0): return 0
    else: return round(fp/total,2)

### Función que entrena un modelo mediante tunning ###
def trainModel(model, params, kfold, trainX, trainY, model_name):
    # Establecer un scoring con métricas personalizadas
    scoring = {
        'balanced_accuracy': metrics.make_scorer(metrics.balanced_accuracy_score),
        'sensitivity': metrics.make_scorer(metrics.recall_score),
        'specificity': metrics.make_scorer(metrics.recall_score,pos_label=0),
        'true_positives' : metrics.make_scorer(tp_score),
        'false_positives' : metrics.make_scorer(fp_score),
        'F-measure' : metrics.make_scorer(metrics.f1_score),
        'AUC' : metrics.make_scorer(metrics.roc_auc_score)
    }
    
    # Realizar la búsqueda de los mejores hiperparámetros del modelo
    grid = GridSearchCV(estimator=model, param_grid=params, cv=kfold, scoring=scoring, refit="balanced_accuracy")
    grid.fit(trainX, trainY)
    
    # Realizar la predicción con el mejor modelo
    best_model = grid.best_estimator_
    
    # Obtener las métricas de calidad para el entrenamiento
    ind = grid.best_index_
    scores = pd.Series([round(grid.cv_results_['mean_test_balanced_accuracy'][ind],2), round(grid.cv_results_['mean_test_sensitivity'][ind],2),
                        round(grid.cv_results_['mean_test_specificity'][ind],2), round(grid.cv_results_['mean_test_true_positives'][ind],2),
                        round(grid.cv_results_['mean_test_false_positives'][ind],2), round(grid.cv_results_['mean_test_F-measure'][ind],2), 
                        round(grid.cv_results_['mean_test_AUC'][ind],2)])
    
    # Crear un dataframe auxiliar con las métricas como columnas
    df_metrs = pd.DataFrame()
    df_metrs = df_metrs.append(scores, ignore_index=True)
    
    # Imprimir los parámetros del modelo ganador
    print("Parámetros ganadores para "+model_name+": ",grid.best_params_)
    
    # Devolver la combinación de parámetros ganadora y sus métricas para la fase de entrenamiento y de test
    return df_metrs, best_model

### Función que formatea y exporta las métricas de los modelos QSAR ###
def formatResults(metrics_df, name):    
    ### Añadir los nombres de las columnas del df
    metrics_df.columns = ['Balanced Accuracy', 'Sensitivity', 'Specificity', 'True Positives Ratio', 'False Positives Ratio', 'F-measure', 'AUC']
    
    ### Añadir columnas descriptivas
    metrics_df.insert(0, "Algorithm", [ a for a in ['Random Forest', 'Support Vector Machines', 'Naive Bayes', 'XGradientBoostTree', 'Artificial Neural Network']
                                          for i in range(2)], True)
    
    metrics_df.insert(1, "Version", [ s for i in range(5) for s in ['Max', 'Mean']], True)
    
    ### Exportar el dataframe a un .CSV
    metrics_df.to_csv('results_'+name+'.csv', index=False)

### Función que realiza el entrenamiento de los modelos QSAR para un conjunto de entrenamiento concreto ###
def QSARtraining(train_max_X, train_max_Y, train_mean_X, train_mean_Y, kf, params_rf, params_svm, params_nb, params_xgb, params_ann, name):
    ### Random Forest
    rf_metrs_max, rf_model_max = trainModel(ensemble.RandomForestClassifier(), params_rf, kf, train_max_X, train_max_Y)
    rf_metrs_mean, rf_model_mean = trainModel(ensemble.RandomForestClassifier(), params_rf, kf, train_mean_X, train_mean_Y)
    
    ### Support Vector Machines
    svm_metrs_max, svm_model_max = trainModel(SVC(), params_svm, kf, train_max_X, train_max_Y)
    svm_metrs_mean, svm_model_mean = trainModel(SVC(), params_svm, kf, train_mean_X, train_mean_Y)
    
    ### Naive Bayes
    nb_metrs_max, nb_model_max = trainModel(BernoulliNB(), params_nb, kf, train_max_X, train_max_Y)
    nb_metrs_mean, nb_model_mean = trainModel(BernoulliNB(), params_nb, kf, train_mean_X, train_mean_Y)
    
    ### XGradientBoostTree
    xgb_metrs_max, xgb_model_max = trainModel(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), params_xgb, 
                                                      kf, train_max_X, train_max_Y)
    xgb_metrs_mean, xgb_model_mean = trainModel(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), params_xgb, 
                                                        kf, train_mean_X, train_mean_Y)
    
    ### Artificial Neural Network
    ann_metrs_max, ann_model_max = trainModel(KerasClassifier(build_fn=buildANN, verbose=0), params_ann, kf, train_max_X, train_max_Y)
    ann_metrs_mean, ann_model_mean = trainModel(KerasClassifier(build_fn=buildANN, verbose=0), params_ann, kf, train_mean_X, train_mean_Y)
    
    # Exportar los modelos QSAR
    pickle.dump(rf_model_max, open('rf_model_max'+name+'.sav', 'wb'))
    pickle.dump(rf_model_mean, open('rf_model_mean'+name+'.sav', 'wb'))
    pickle.dump(svm_model_max, open('svm_model_max'+name+'.sav', 'wb'))
    pickle.dump(svm_model_mean, open('svm_model_mean'+name+'.sav', 'wb'))
    pickle.dump(nb_model_max, open('nb_model_max'+name+'.sav', 'wb'))
    pickle.dump(nb_model_mean, open('nb_model_mean'+name+'.sav', 'wb'))
    pickle.dump(xgb_model_max, open('xgb_model_max'+name+'.sav', 'wb'))
    pickle.dump(xgb_model_mean, open('xgb_model_mean'+name+'.sav', 'wb'))
    ann_model_max.model.save('ann_model_max'+name+'.h5')
    ann_model_mean.model.save('ann_model_mean'+name+'.h5')
    
    # Exportar las métricas del entrenamiento
    formatResults(pd.concat([rf_metrs_max, rf_metrs_mean, svm_metrs_max, svm_metrs_mean, nb_metrs_max, nb_metrs_mean, 
                    xgb_metrs_max, xgb_metrs_mean, ann_metrs_max, ann_metrs_mean]), "train"+name)

### Función que exporta los conjuntos de test ###
def saveTestData(test_max_X, test_max_Y, test_mean_X, test_mean_Y, name):
    test_max_X.to_csv('test_max_X'+name+'.csv', index=False)
    test_max_Y.to_csv('test_max_Y'+name+'.csv', index=False)
    test_mean_X.to_csv('test_mean_X'+name+'.csv', index=False)
    test_mean_Y.to_csv('test_mean_Y'+name+'.csv', index=False)