import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mlt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, PandasTools

file_path = '/Users/tonyelin/Coding/cdk8-classifier/resources/similarity-matrix-input/CDK8_diverse_structures.csv'
cdk8_active = pd.read_csv(file_path)

# Variables
nBits = 2048
radius = 2
smiles_col = 'smiles'
smiles = cdk8_active['smiles']
cmap = 'Greens'

# Create list for Tanimoto scores
Tanimoto = []

# Add ROMol to DataFrame
PandasTools.AddMoleculeColumnToFrame(cdk8_active, smilesCol=smiles_col)

for compound in cdk8_active['smiles']:
	# Create references for smiles, molecules, fingerprints
    ref_smiles = compound
    ref_mol = Chem.MolFromSmiles(ref_smiles)
    ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, radius=radius, nBits=nBits)
    bulk_fp = [AllChem.GetMorganFingerprintAsBitVect(x,2) for x in cdk8_active['ROMol']]

    # Similarity to reference molecule
    similarity = [DataStructs.FingerprintSimilarity(ref_fp,x) for x in bulk_fp]

    # Append list with similarity score
    Tanimoto.append(similarity)

# Create list of compound name
compound_name = cdk8_active['Compound ID'].tolist()

# Loop Tanimoto Score to each Compound name
for score in Tanimoto:
    cdk8_active[compound_name] = Tanimoto

matrix = cdk8_active.drop(['Compound ID', 'smiles', 'ROMol'], axis=1)
# Rename the index using a dictionary comprehension
new_index_mapping = {old_index: new_name for old_index, new_name in enumerate(compound_name)}
matrix = matrix.rename(index=new_index_mapping)

mlt.rcParams['figure.dpi'] = 300  # increase matplotlib dpi

# Adjust annotations size using the annot_kws
# Turn off Annotations by removing annot=TRUE
# Adjust color based on needs:
# cmap='Blues', YlOrBr, Greens, BuPu, etc

# For dendrogram
image = sns.clustermap(matrix,
                       annot=False,  # Add annotations to the table
                       cmap=cmap,
                       linewidth=0.1,
                       linecolor='black',
                       vmin=0, vmax=1,
                       dendrogram_ratio=0.03,
                       cbar_kws={"orientation": "vertical", 'ticks': [0.0, 0.5, 1.0]},
                       cbar_pos=(0.9, 0.03, 0.01, 0.1))

plt.tight_layout

# Remove labels
ax = image.ax_heatmap
ax.set_ylabel('')
ax.set_xlabel('')

plt.savefig('matrix.png')
plt.show()

# Output matrix DataFrame as .csv file
# Get the order of rows and columns
row_order = image.dendrogram_row.reordered_ind
col_order = image.dendrogram_col.reordered_ind

# Reorder the DataFrame based on the cluster map order
matrix_order = matrix.iloc[row_order, col_order]

# Save the reordered data to a .csv file
matrix_order.to_csv('cluster_map_data.csv')
