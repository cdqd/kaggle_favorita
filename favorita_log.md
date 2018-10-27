## Competition, methodology and results overview

### Competition task
This was a time-series problem requiring the prediction of:
- Unit sales over a two week period
- For over 160,000 item/store combinations for a grocery chain
- Having information regarding past sales and a promotional indicator, but no information on prices

### Data cleaning
The data itself required some cleaning:
- Training data excluded instances where there were no sales made on that day.
    - It is unclear whether this was due to no stock or no demand.
    - To prevent overstating total sales over the two week prediction period, these missing instances were filled with 0.
- Daily data from August 2013 to August 2017 was provided -- this resulted in a 4.8gb training file.
    - Only 2017 data was read in for training/validation. This reduces memory usage, and more recent data is likely to be more predictive.

### Training data setup and modelling

#### What didn't work
I tried a few methods for different parts of the modelling which did not work well:
- Aggregating sales at the Item Type - Item Class level
- Aggregating stores at the City - Type - Cluster - State level
- The reason for collapsing the data as above was to further reduce memory usage and in order to train and validate across a longer period of time (at least 1 year) so that monthly effects could be captured. **The aggregation turned out to be very ineffective.** It was unclear whether the validation strategy would have worked.
- Using daily rolling averages as predictors in the training set, then using chunk averages for the predictors in the validation/test period. The effectiveness of this was unclear.

#### Ceshine Lee's separated-gradient-boosting method
This was one of the most effective methods for the competition, and by far the most popular. It involved viewing the problem in the following way:
- We are given one chunk of data that stops the day before the prediction period. This chunk must then be used to predict the sales:
    - 1 day away from the end of the chunk, (t1)
    - 2 days away, (t2)
    - 3 days away, etc., (t3)
until the last day of the test period (16 days). 
- So a separate model can be built for each of t1, t2, ..., t16. The exact same design matrix will be used in the training data for each model, but the target value will change.

- A single hold-out dataset (the validation set) can be created to mimic the prediction period as closely as possible in order to tune the parameters of each gradient boosting model.
- We can create the training data by considering different 'chunks' leading up to the validation/prediction period, as long as t1, t2, ..., t16 does not fall into the val/prediction period.
- To accurately represent the prediction period, and to mitigate any extraneous effect the days of the week may have on each of t1, t2, ..., t16, we always end the 'chunks' on the same weekday.
    - For example, if our validation period starts on 2017-07-26, the latest date a training chunk can end is 2017-07-04.
    - This is so the targets (t1, t2, ..., t16) can span 2017-07-05 to 2017-07-20.

- Feature engineering can include calculating summary statistics of past sales for each item/store combination. 
- Although this was not done in the public kernel, for the final submission we can then refresh the training data to include data closer to the prediction period.

