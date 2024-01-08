### Notes on Similarity Matrix

Small-molecules with reported CDK8 activity was downloaded from the ChEMBL Database (version 33)
Active compounds were retained. This included compounds with EC50, IC50, Ki, and Kd.

Data was preprocessed using the KNIME worklfow. Diverse structures were selected using 
the Datamol module with the pick_diverse() function. 30 compounds were selected. 
The final table includes the 30 compounds with the 5 hit CDK8 inhibitors identified in this study. 

Similarity Matrix was generated from tanimoto similarity calculations between each molecule using RDKit . Compounds 
were converted to 2048 Morgan fingerprints. The heatmap and the dendogram was generated using Seaborn. 