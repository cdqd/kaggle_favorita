# Kaggle - Porto Seguro - learning points

[Link to competition](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction) 

I entered this with the aim of developing knowledge in:

- dplyr / other utilities packages from tidyverse
- Implementing a variety of ML packages in R
- Stacking and blending
- Validation techniques in model building

My time frame to complete the competition was around 2 weeks.

## Summary of data and competition

- Tabular data provided:
  - One row for each policy.
  - About 50 anonymized characteristics for each policy, with flags for whether they are numeric or categorical.
  - Binary indicator of whether or not claim was filed for each policy.
  
- Ambiguities about the data:
  - Time period for training set policies vs evaluation set policies was not specified.
  
- Competition task was to rank-order a subset of the policies in the evaluation set (the private set) on the probability of filing a claim. The metric used was the Gini coefficient.
  - The Gini score on the public set was avaiable as instant feedback.
  
## Summary of approach and results

### Data prep

[Data prep script](https://github.com/cdqd/cdqd.github.io/blob/master/kaggle_portoseguro/01-dataprep.R)

Quick data cleaning:

- Removed high cardinality categorical variables
- Removed variables with a large percentage of missing values.
- Assumed missing values to be Missing Not At Random; replaced with specific values (-999 for numerics, Missing category for categoricals)

A quick and dirty xgboost was then roughly tuned & built to assess the importance of the each of the remaining variables. This was used to guide variable creation.

- Added count of number of missing variables for each row.
- Added indicator variables for whether the 1st/2nd most important variables (according to quick and dirty model) were missing for each row.
- Added indicators for whether the variables with a a large % of missing values was missing for that row.

- Two separate datasets were then made, one with categoricals left alone, the other with categoricals turned to numerics via one-hot-encoding.

### Model structure, validation, results

[Quick and dirty models](https://github.com/cdqd/cdqd.github.io/blob/master/kaggle_portoseguro/02-models_quick.R)

[Stage 1 modelling](https://github.com/cdqd/cdqd.github.io/blob/master/kaggle_portoseguro/03-stage1_tune.R)

[Stage 2 modelling](https://github.com/cdqd/cdqd.github.io/blob/master/kaggle_portoseguro/04-stage2_tune.R)

- Aim was to build models in two stages:
  - Stage 1 on original data
  - Stage 2 on predictions of models in stage 1, either using stacking or blending.

Used a hold-out validation for stage 1 models. Training data was stratified by target value, then an 80 train/20 valid split was made while retaining the distribution of the most important variable (ps_car_13) for each stratum.

Many models were trialled for Stage 1 (can be found in stage 1 trial script). Lightgbm (with categoricals passed through lgb's in-built encoding) and xgboost were the strongest single models. 

Stacking with logistic regression was built, with variables selected manually in a step-wise manner:
  - xgboost with OHE data
  - lgb with categoricals encoded using lgb's in-built categorical encoding (similar to target encoding)
  - same lgb as above, but trained on the most important variable cluster only.

The stacked model had the best cv score on the stage 2 subset and was selected for submission.

The same three stage 1 models were also included in a blend as a last-minute trial. Harmonic means, rank averaging, and standard averaging were attempted and submitted. These had worse gini scores compared to the stacked model, but rank average had the best public leaderboard score and was the second submission selected.
 
To create the submission files, all model setups were re-trained on the entire dataset. In the end, the rank average blending performed best on the private leaderboard.

## Reflections and going forward

In my view, competitions where the variables are anonymized have the aim of finding innovative **computational** techniques, but cannot offer additional insight into underlying business/customer/market mechanics. I used this competition for my own learning and a very different purpose, so there are many areas of improvement that could have been made throughout the process:

  - Data prep: Do target encoding for categorical variables
  - Data prep: Trial other types of imputation for missing values; trial different methods for different columns. Much more experimentation could have been done in this space.
  - Validation: Use multiple folds and cross-validate to reduce bias in estimates
  - Stage 2 validation: Average multiple folds to increase the amount of data points used, again to reduce bias.
  - Logistic regression/elastic-net: Trial different methods for variable selection (consider what the best loss criteria would be, etc.)

Most of of the above are ideas gathered from the Kaggle discussion forum. There were also more extensive tasks done by top performers:  
  - Feature engineering by trying different combinations of variables
  - Stacking a large amount of Stage 1 models 
  - Blending with other public submission files

However, I believe these methods do not make intuitive sense or are inefficient in practice, which in my view defeats the purpose of the competition.
