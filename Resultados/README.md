# Result files  description:

The following files show the final data sets of the screening and standardization phases, that will be used to train the QSAR models. The data frames are composed by:
 - Chembl ID
 - Morgan Fingerprint bit vector (2048 columns)
 - Activity flag (Active for inhibitory molecule)

Four versions have been considered:
 - **data_dnmt1_226_max.csv**: Molecules with inhibitory activity against DNMT1 at confidence level 9 (maximum pChembl in case of duplicated molecule).
 - **data_dnmt1_226_mean.csv**: Molecules with inhibitory activity against DNMT1 at confidence level 9 (mean of pChembl in case of duplicated molecule).
 - **data_dnmt1_430_max.csv**: Molecules with inhibitory activity against DNMT1 at confidence level 8 (maximum pChembl in case of duplicated molecule).
 - **data_dnmt1_430_mean.csv**: Molecules with inhibitory activity against DNMT1 at confidence level 8 (mean of pChembl in case of duplicated molecule).
