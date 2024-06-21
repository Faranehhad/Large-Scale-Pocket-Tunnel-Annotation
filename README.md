# Large Scale Pocket and Tunnel Annotation
This repository contains scripts for the paper:<br>
Large-scale Annotation of Biochemically Relevant Pockets and Tunnels in Cognate Enzyme-Ligand Complexes. <br>
O. Vavra, J. Tyzack, F. Haddadi, J. Stourac, J. Damborsky, S. Mazurenko, J. Thornton, D. Bednar
bioRxiv 2023.03.29.534735; doi: https://doi.org/10.1101/2023.03.29.534735 <br>

The predictor was applied to annotate pockets and tunnels in 17,000 proteins as part of the pipeline for validation.

## Dependencies
Python: 3.9.7, NumPy: 1.26.2, Pandas: 1.4.3, Scikit-learn: 1.1.1

## Dataset
Train set for the 3-class predictor: ML_subset_TRAIN_3class.csv <br>
Test set for the 3-class predictor:ML_subset_TEST_3class.csv <br> 
Train set for the 2-class predictor: ML_subset_TRAIN_2class.csv <br>
Test set for the 2-class predictor: ML_subset_TEST_2class.csv <br> <br>
For training, we manually labeled 200 proteins with calculated pockets from the dataset. By analyzing the distribution of EC classes, we randomly sampled proteins from each class to match the overall distribution, ensuring that the datasets provided representative sampling across different EC classes. The proteins were annotated based on visual inspection. <br><br>
The files 'train.csv' and 'test.csv' contain the output predictions from the predictors.
## How to run
Place the datasets in the same directory as pocket_tunnel_annotation.py and run the script:
```bash
python pocket_tunnel_annotation.py
```
The script will perform the following tasks:<br>
- Load and preprocess the datasets.<br>
- Perform GridSearchCV to find the best hyperparameters for each classifier.<br>
- Evaluate the models on training and test data (accuracy, precision, recall, f1 scores, and confusion matrices)<br>
- Plot confusion matrices and learning curves for each classifier. <br>
- Save the results in train.csv and test.csv

## License
For licensing terms, please contact the corresponding author, Dr. David Bednar, at 222755@mail.muni.cz.






