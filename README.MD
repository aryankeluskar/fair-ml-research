# fair-ml-research
Researching impact of homomorphic encryption on fairness/bias of a dataset. In this repository, I have implemented Linear Regression and Logistic Regression with CKKS or BFV Schemes as of Jan 13th. 

## Linear Regression - Observations
In the folder LinReg, there are two python files: one with CKKS Encryption and one without. After executing both the files 20 times, there is an average of 0.05% drop in accuracy while using CKKS Encryption.
UPDATE: I have used the boston dataset (http://lib.stat.cmu.edu/datasets/boston) to make graphs and therefore test accuracy of every feature against Median value of owner-occupied homes. Results can be found in LinReg/Combined, LinReg/sklearn_ckks, LinReg/sklearn_plain. Log Files with metrics can be found in CKKSResults.txt, PlainResults.txt, and sklearnResults.txt <br>
To replicate:
1) use LinReg/boston.txt 
2) execute TextToDF.py
3) execute LinRegCKKS.py, LinRegPlain.py, LinRegSklearn.py
4) run allGrapher.py


## Logarthmic Regression - Method & Observation
### Datasets:
Adult (http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data): Sensitive Attribute is 'sex', target is 'income'
<br>
Compas (https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-raw.csv) - Requires Multinomial Logistic Regression so put on hold temporarily 
<br>
German (https://online.stat.psu.edu/stat857/sites/onlinecourses.science.psu.edu.stat857/files/german_credit/index.csv): Senstive Attribute is 'foreign worker', target is 'credibility'
<br>
These three are used since they are the most popular datasets in fairness research as per [this article](https://link.springer.com/article/10.1007/s10618-022-00854-z)

### Pre-processing:
For Adult, there is a two-step pre-processing
1) To remove the rows that are missing certain data (code in AdultDataToCsv.py, output stored in adultFinal.csv)
2) To use OneHotEncoder and binarize catagorical columns (code in AdultPreProcess.py, output stored in adultPreprocessed.csv)

### Model Training and Evaluation:
AdultLogRegPlain.py - Python Code for Adult Dataset without CKKS
<br>
Adult_LogRegCKKS.py - Python Code for Adult Dataset using CKKS 
<br>
Similar to Linear Regression, there are separate files for Plain and Homomorphic Training of this dataset. Documentation is present in the files as well, but to give an overview, PyTorch is used for training and testing the model. The dataset is randomly split 30% for testing and 70% for training. For encryption, Tenseal library is used. In every experiment run, the following is calculated:
1) Accuracy
2) Confusion Matrix
3) True Postive Rate and False Positive Rate
4) Equalized Odds for protected group

### Results:
The output is stored in the following files:
1) PlainAdultCorrected.txt - Results for Adult Dataset without CKKS 
2) CKKSAdultCorrected.txt - Results for Adult Dataset using CKKS

### Observations:
After 10 iterations, there is an average of 4.81% drop in accuracy while using CKKS Encryption. However, the hope shines while comparing the average Equalized Odds for the protected group. Without CKKS (~0.6055) to the version with CKKS (~0.5111), CKKS has led to a significant improvement in fairness, indicating a more equitable outcome for the protected group in this particular project.