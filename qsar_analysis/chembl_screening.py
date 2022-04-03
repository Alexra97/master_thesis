# Cargar las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

### Función que elimina las moléculas que no se encuentran dentro de los límites estándar ###
def molPropCorrect(df):
    return df.loc[df['properties_log'] == 'Molecule correct. Molecular properties (MW,NR) within standard limits'];

### Función que elimina las moléculas duplicadas (por su InchiKey) ###
def delDupMols(df):
    # Obtener InchiKeys
    inchikeys = df[["standardized_molecule_InchiKey"]]
    
    # Ordenar alfabéticamente
    iks_sort = inchikeys.sort_values("standardized_molecule_InchiKey")
    
    # Declarar una lista auxiliar para almacenar los duplicados
    dups = []
    
    # Comprobar duplicados y almacenarlos
    for i in range(len(iks_sort)):
        if (i != len(iks_sort)-1):
            if (iks_sort.iat[i,0] == iks_sort.iat[i+1,0]):
                if (iks_sort.index[i] not in dups): dups.append(iks_sort.index[i])
                if (iks_sort.index[i+1] not in dups): dups.append(iks_sort.index[i+1])
                
    if not dups:
        return 0;
    else:
        # Generar los dos nuevos data frames a partir de los no duplicados
        df_max = df.drop(dups)
        df_mean = df.drop(dups)
        
        # Obtener un subset de los duplicados
        dups = df.loc[dups,]
        
        # Eliminar los duplicados
        ## Mientras haya duplicados en el subset...
        while not dups.empty:
            # Obtener el índice y el inchikey
            d = dups.index[0]
            inchikey_d = dups.loc[d,"standardized_molecule_InchiKey"]
            
            # Obtener todos los duplicados para ese inchikey
            dup_aux_df = dups.loc[dups["standardized_molecule_InchiKey"] == inchikey_d,]
            
            # Obtener el duplicado con el pChEBML máximo
            dup_max = dup_aux_df.loc[dup_aux_df["pChEMBL Value"].idxmax()]
            
            # Obtener la media de pChEMBLs
            mean_p_value = dup_aux_df["pChEMBL Value"].sum()/len(dup_aux_df)
            
            # Extraer una fila cualquiera como base y colocar en ella el valor medio
            dup_mean = dup_aux_df.iloc[0,]
            dup_mean["pChEMBL Value"] = mean_p_value
            
            # Añadir cada fila a su data set correspondiente
            df_max = df_max.append(dup_max)
            df_mean = df_mean.append(dup_mean)
            
            # Descartar los duplicados locales de la tabla de duplicados global para que no se repita
            # la iteración sobre ellos
            dups = dups.drop(dup_aux_df.index)
        
        return df_max, df_mean;
       
### Función para mostrar el Elbow Plot y obtener el valor óptimo de epsilon ### 
def elbowPlot(df, n):
    # Obtener las distancias hacia los n vecinos más cercanos
    nearest_neighbors = NearestNeighbors(n_neighbors=n)
    neighbors = nearest_neighbors.fit(df)
    
    distances, indices = neighbors.kneighbors(df)
    distances = np.sort(distances[:,14], axis=0)
    
    # Obtener el Elbow point
    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    
    # Mostrar el gráfico de codo
    knee.plot_knee()
    plt.xlabel("Puntos")
    plt.ylabel("Distancia")
    
    return distances[knee.knee];       
     
### Función para mostrar el gráfico del análisis DBSCAN ###   
def dbscanPlot(df, knee, n, title):
    # Identificación de outliers
    dbscan_cluster = DBSCAN(eps=knee, min_samples=n)
    dbscan_cluster.fit(df)
    
    # Visualizar DBSCAN
    plt.scatter(df["Molecular Weight"], df["pChEMBL Value"], c=dbscan_cluster.labels_)
    plt.xlabel("Molecular Weight")
    plt.ylabel("pChEMBL Value")
    plt.title(title)
    plt.show()
    
    # Número de clusters
    labels=dbscan_cluster.labels_
    N_clus=len(set(labels))-(1 if -1 in labels else 0)
    print('Número de clusters: %d' % N_clus)
    
    # Número de outliers
    n_noise = list(dbscan_cluster.labels_).count(-1)
    print('Número de outliers: %d' % n_noise)
    
### Función para cactorizar el pChEMBL como activo o inactivo ###      
def labelData(df): 
    df['Activity'] = 'Active'
    df.loc[df['pChEMBL Value'] < 7, 'Activity'] = 'Inactive'
    return df;
    
### Función que elimina las columnas sobrantes para el entrenamiento ###   
def dropCols(df):
    return df.drop(["standardized_molecule_InchiKey", "properties_log", 
                    "Molecular Weight","pChEMBL Value"], axis=1, inplace=True);
        
        
        
        
        
