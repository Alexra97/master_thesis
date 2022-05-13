# Result files  description:

The following files show the final results of the study. They are divided in three main blocks:
 - **results_train**: Train metrics for each QSAR model developed.
 - **results_test**: Test metrics for each QSAR model developed.
 - **results**: Set of predictions over the Docking data with the best QSAR model based on its metrics. The predictions are composed by PubChem ID, predicted activity (0 for Inactive, 1 for Active) and the similar molecules of the train set based on Dice and Tanimoto Similarity. NA values represent 0 similar molecules (similarity below 0.7).

Data are labelled by:
 - **Case study**: Original (DNMT1), addition of DNMT3A and DNMT3B (DNMT3) and addition of Oleacin (Oleacin).
 - **Number of molecules (Confidence Level)**: 226 molecules for a confidence level of 9. 430 molecules for a confidence level of 8.

Finally, there is a folder called "Best models" with the best QSAR model for each of the six case studies. So anyone can import and use them.
