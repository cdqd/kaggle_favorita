# Time series model diagnostics

#### Purpose of personal project
The purpose of this project was to perform model diagnostics on the most popular modelling methodology used in the [Corporacion Favorita Sales Forecasting competition](https://www.kaggle.com/c/favorita-grocery-sales-forecasting).

#### Model information
A description of the modeling methodology and discussion of potential extensions can be found here:
- [Original kernel](https://www.kaggle.com/npa02012/ceshine-s-lgbm-starter-in-r-lb-0-529)
- [Further ideas](https://www.kaggle.com/vrtjso/lgbm-one-step-ahead)

I have re-written the original script to incorporate these additional features:
- [R script used to generate model](00_train_val.R)

In summary:
- The model is a composite of 16 gradient boosted decision tree ensembles, one for each day of the prediction period. - - Variables used in the model were:
    - The combination of item number and store number (the most granular level of the data provided)
    - Statistics of past sales for each combination of item number and store number. This included means, day-of-week means, mins, maxs, different quantiles and standard deviations over measurement periods ranging from the most recent sale to the past 140 sales.
    - The sum of the number of times the item was on promotion.

## Model diagnostics

This exercise was based on the "training" (fully-labelled) data provided for this competition only. The data was split into train and validation periods, and only the performance of the model on the validation period has been assessed. The report provides detailed analysis and summary of key findings of the performance of the model in relation to its ability to predict sales:

[Link to model diagnostics](02_model_diagnostics.Rmd)

Although we have looked at how the model may be improved given the restrictions of the competition, an extension to this task may be to compare the performance of the model against current industry-standard models. A better assessment of the effectiveness of the model in forecasting sales can be made if all available information (such as pricing) is considered in the modelling process, not just historical sales.

To reproduce the report, please run the `00_train_val.R`, `01_libraries_data.R`, `02_model_diagnostics.Rmd` files (in that order).

## Additional notes

I have also made a [personal log](favorita_log.md) of my experience with the Kaggle competition itself, which includes a detailed description of the ideas behind the modelling methodology. Due to time constraints, I was unable to explore my own ideas in further detail for the competition.
