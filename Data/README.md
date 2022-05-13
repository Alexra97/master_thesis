# Data files  description:

The following files show the original and standardized data sets used in the project to train the QSAR models. The main columns are:
 - **Molecule Chembl ID**
 - **standardized_molecule**: Standardized SMILES to get the Morgan Fingerprint bit vector (2048 columns).
 - **pChEMBL Value**: Activity value to label the data. Above or equal to 6 for Active (inhibitory) molecules.

Data are labelled by:
 - **Target**: DNMT1, DNMT3A and DNMT3B.
 - **Number of molecules (Confidence Level)**: 226 molecules for a confidence level of 9. 430 molecules for a confidence level of 8.
 - **Standardization**: If the filename has the suffix *_std*.

Exceptions:
 - oleacein.csv: Just the SMILES for oleacein.
