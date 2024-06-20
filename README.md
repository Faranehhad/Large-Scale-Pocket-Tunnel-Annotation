# large-scale-pocket-tunnel-annotation
This repository contains scripts for the paper:<br>
Large-scale Annotation of Biochemically Relevant Pockets and Tunnels in Cognate Enzyme-Ligand Complexes.
O. Vavra, J. Tyzack, F. Haddadi, J. Stourac, J. Damborsky, S. Mazurenko, J. Thornton, D. Bednar
bioRxiv 2023.03.29.534735; doi: https://doi.org/10.1101/2023.03.29.534735

# Abstract:
Tunnels in enzymes with buried active sites are key structural features allowing the entry of substrates and the release of products, thus contributing to the catalytic efficiency. Targeting the bottlenecks of protein tunnels is also a powerful protein engineering strategy. However, the identification of functional tunnels in multiple protein structures is a non-trivial task that can only be addressed computationally. We present a pipeline integrating automated structural analysis with an in-house machine-learning predictor for the annotation of protein pockets, followed by the calculation of the energetics of ligand transport via biochemically relevant tunnels. A thorough validation using eight distinct molecular systems revealed that CaverDock analysis of ligand un/binding is on par with time-consuming molecular dynamics simulations, but much faster. The optimized and validated pipeline was applied to annotate more than 17,000 cognate enzyme-ligand complexes. Analysis of ligand un/binding energetics indicates that the top priority tunnel has the most favourable energies in 75 % of cases. Moreover, energy profiles of cognate ligands revealed that a simple geometry analysis can correctly identify tunnel bottlenecks only in 50 % of cases. Our study provides essential information for the interpretation of results from tunnel calculation and energy profiling in mechanistic enzymology and protein engineering. We formulated several simple rules allowing identification of biochemically relevant tunnels based on the binding pockets, tunnel geometry, and ligand transport energy profiles.

## Dependencies
Python: 3.9.7, NumPy: 1.26.2, Pandas: 1.4.3, Scikit-learn: 1.1.1

## Dataset
Train set for the 3-class predictor: ML_subset2_with_seq.csv <br>
Test set for the 3-class predictor: ML_subset-TESTING.csv <br> 
Train set for the 2-class predictor: ML_subset2_with_seq_2class.csv <br>
Test set for the 2-class predictor: ML_subset-TESTING_2class.csv

## How to run
Place the datasets in the same directory as pocket_tunnel_annotation.py and run the script:
```bash
python pocket_tunnel_annotation.py
```
- The script will perform the following tasks:<br>
- Load and preprocess the datasets.<br>
- Perform GridSearchCV to find the best hyperparameters for each classifier.<br>
- Evaluate the models on training and test data.<br>
- Plot confusion matrices and learning curves for each classifier.

## Results
The script provides detailed results for each classifier, including:<br>

- Best hyperparameters found using GridSearchCV.<br>
- Training and test accuracy, precision, recall, and F1 scores.<br>
- Confusion matrices for both training and test data.<br>
- Learning curves to analyze model performance over varying training set sizes.<br>
<br>
The results for all predictors on train and test sets are saved in: <br>
- train.csv <br>
- test.csv






