# This script builds off:
# - The R translation of Ceshine's LGBM starter:
#   (https://www.kaggle.com/npa02012/ceshine-s-lgbm-starter-in-r-lb-0-529)
# - Elements, ideas, and discussions from the LGBM starter improvement written in Python:
#   (https://www.kaggle.com/vrtjso/lgbm-one-step-ahead)
#
# In summary, I have added the following elements to the original R script:
# - DOW means, and longer-term means and promotion totals
# - Quantiles from 0 to 100%, by 10%, for variables that include >= 20 observations of past data.
#   - e.g. mean_30, mean_60, dow20, ...
# - Standard deviations of the observations used for these same variables
# - Stacked on more training data
# - Used the evaluation weights in model training
#
# The code for creating quantile/sd fields was optimized by using matrixes in conjunction with fast row-calc functions
# xgboost is used at the end instead of lightgbm, simply because I wished to test xgboost's histogram method

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Loading and Cleaning Data
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
rm(list = ls()); gc()

cat("Setting up Enviornment\n")
library(data.table)
library(tidyverse)
library(lubridate)
library(matrixStats)
library(xgboost)

# delete function
delete <- function(DT, del.idxs) 
{ 
  varname = deparse(substitute(DT))
  
  keep.idxs <- setdiff(DT[, .I], del.idxs)
  cols = names(DT);
  DT.subset <- data.table(DT[[1]][keep.idxs])
  setnames(DT.subset, cols[1])
  
  for (col in cols[2:length(cols)]) 
  {
    DT.subset[, (col) := DT[[col]][keep.idxs]]
    DT[, (col) := NULL];  # delete
  }
  
  assign(varname, DT.subset, envir = globalenv())
  return(invisible())
}

sink("val_script.log", split = T, append = T)

cat("Loading train")
df_2017 <- fread("train.csv", sep=",", na.strings="", skip = 90000000,
               col.names=c("id","date","store_nbr","item_nbr","unit_sales","onpromotion"))

start_time <- Sys.time()

df_2017[, id := NULL]

delete(df_2017, which(df_2017$date < "2017-01-01"))
gc()
cols <- names(df_2017)

test <- fread("test.csv", sep=",", na.strings = "")

cat("Setting onpromotion to integer")
df_2017[, ":=" (onpromotion = as.integer(onpromotion)
              , unit_sales = log1p((unit_sales > 0) * unit_sales))]

test[, onpromotion := as.integer(onpromotion)]

# Making wide table of onpromotion information
vars = c("item_nbr", "store_nbr", "date","onpromotion")
promo_2017_train <- dcast(df_2017[, vars, with = F], store_nbr + item_nbr ~ date, value.var = "onpromotion")
promo_2017_test <- dcast(test[, vars, with = F], store_nbr + item_nbr ~ date, value.var = "onpromotion")
promo_2017 <- left_join(promo_2017_train, promo_2017_test, by = c("store_nbr","item_nbr")) %>% as.data.table
promo_2017[is.na(promo_2017)] <- 0
rm(promo_2017_train, promo_2017_test)

# Make wide table of unit_sales information
df_2017 <- dcast(df_2017, store_nbr + item_nbr ~ date, value.var = "unit_sales")
df_2017[is.na(df_2017)] <- 0

# - Functions
prepare_dataset <- function(t2017, is_train = T){
  m1_names = as.character(seq.Date(t2017 - 1, by = "days", length.out = 1))
  m3_names = as.character(seq.Date(t2017 - 3, by = "days", length.out = 3))
  m7_names = as.character(seq.Date(t2017 - 7, by = "days", length.out = 7))
  m14_names = as.character(seq.Date(t2017 - 14, by = "days", length.out = 14))
  m30_names = as.character(seq.Date(t2017 - 30, by = "days", length.out = 30))
  m60_names = as.character(seq.Date(t2017 - 60, by = "days", length.out = 60))
  m140_names = as.character(seq.Date(t2017 - 140, by = "days", length.out = 140))
  p14_names = as.character(seq.Date(t2017 - 14, by = "days", length.out = 14))
  p60_names = as.character(seq.Date(t2017 - 60, by = "days", length.out = 60))
  p140_names = as.character(seq.Date(t2017 - 140, by = "days", length.out = 140))
  
  # day of week means
  for(j in 1:7){
    assign(paste0("dow4names", as.character(wday(t2017 - j, label = T))),
           as.character(seq.Date(t2017 - j - 3 * 7, by = "weeks", length.out = 4)))
    assign(paste0("dow20names", as.character(wday(t2017 - j, label = T))),
           as.character(seq.Date(t2017 - j - 19 * 7, by = "weeks", length.out = 20)))
  }
  
  X <- data.table( store_nbr = df_2017[, store_nbr]
                   ,item_nbr = df_2017[, item_nbr]
                   ,mean_1_2017 = df_2017[, rowMeans(.SD), .SDcols = m1_names]
                   ,mean_3_2017 = df_2017[, rowMeans(.SD), .SDcols = m3_names]
                   ,mean_7_2017 = df_2017[,rowMeans(.SD), .SDcols = m7_names]
                   ,mean_14_2017 = df_2017[,rowMeans(.SD), .SDcols = m14_names]
                   ,mean_30_2017 = df_2017[, rowMeans(.SD), .SDcols = m30_names]
                   ,mean_60_2017 = df_2017[, rowMeans(.SD), .SDcols = m60_names]
                   ,mean_140_2017 = df_2017[, rowMeans(.SD), .SDcols = m140_names]
                   ,promo_14_2017 = promo_2017[, rowSums(.SD), .SDcols = p14_names]
                   ,promo_60_2017 = promo_2017[, rowSums(.SD), .SDcols = p60_names]
                   ,promo_140_2017 = promo_2017[, rowSums(.SD), .SDcols = p140_names]
                   ,mean_dow4_Mon = df_2017[, rowMeans(.SD), .SDcols = dow4namesMon]
                   ,mean_dow4_Tue = df_2017[, rowMeans(.SD), .SDcols = dow4namesTue]
                   ,mean_dow4_Wed = df_2017[, rowMeans(.SD), .SDcols = dow4namesWed]
                   ,mean_dow4_Thu = df_2017[, rowMeans(.SD), .SDcols = dow4namesThu]
                   ,mean_dow4_Fri = df_2017[, rowMeans(.SD), .SDcols = dow4namesFri]
                   ,mean_dow4_Sat = df_2017[, rowMeans(.SD), .SDcols = dow4namesSat]
                   ,mean_dow4_Sun = df_2017[, rowMeans(.SD), .SDcols = dow4namesSun]
                   ,mean_dow20_Mon = df_2017[, rowMeans(.SD), .SDcols = dow20namesMon]
                   ,mean_dow20_Tue = df_2017[, rowMeans(.SD), .SDcols = dow20namesTue]
                   ,mean_dow20_Wed = df_2017[, rowMeans(.SD), .SDcols = dow20namesWed]
                   ,mean_dow20_Thu = df_2017[, rowMeans(.SD), .SDcols = dow20namesThu]
                   ,mean_dow20_Fri = df_2017[, rowMeans(.SD), .SDcols = dow20namesFri]
                   ,mean_dow20_Sat = df_2017[, rowMeans(.SD), .SDcols = dow20namesSat]
                   ,mean_dow20_Sun = df_2017[, rowMeans(.SD), .SDcols = dow20namesSun]
                  )
  
  # additional stats
  matrix_names <- c("m30_names", "m60_names", "m140_names", 
                    paste0("dow20names", as.character(wday(1:7, label = T))))
  for(k in matrix_names){
    curr_matrix <- as.matrix(df_2017[, get(k), with = F])
    stats_matrix <- rowQuantiles(curr_matrix, probs = seq(0, 1, by = 0.1))
    stats_matrix <- cbind(stats_matrix, rowSds(curr_matrix))
  X[, (c(paste0("p", seq(0, 1, by = 0.1), "_", k), paste0("sd_", k))) := as.data.table(stats_matrix)]
  }
  
  # Get onpromotion information for t2017 and the following 15 days.
  for(i in 0:15) {
    new_var = paste0("promo_",i)
    var = as.character(t2017 + i) 
    X[, (new_var) := promo_2017[, var, with = F]]
  }
  # Get unit_sales information for t2017 and following 15 days
  if(is_train) {
    y_dates = as.character(seq.Date(t2017, by="days", length.out = 16))
    y = df_2017[, c("store_nbr", "item_nbr", y_dates), with = F]
    colnames(y) = c("store_nbr", "item_nbr", "y_1","y_2","y_3","y_4","y_5","y_6","y_7","y_8",
                    "y_9","y_10","y_11","y_12","y_13","y_14","y_15","y_16")
    return(list(X = X, y = y))
  } else {
    return(list(X = X))
  }
}

cat("Making X_train and y_train\n")
t2017 = as.Date("2017-05-31")
# 6 'sets' of train data
for(i in 0:5) {
  # Make X_tmp and Y_tmp
  delta <- 7 * i
  results <- prepare_dataset(t2017 + delta)
  X_tmp = results$X
  y_tmp = results$y
  # Concatenating X_l
  if(i == 0) {
    X_train <- X_tmp
    y_train <- y_tmp
  } else {
    X_train = rbindlist(list(X_train, X_tmp))
    y_train = rbindlist(list(y_train, y_tmp))
  }
}
val_date <- as.Date("2017-07-26")
results <- prepare_dataset(val_date)
X_val <- results$X
y_val <- results$y

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Preparing data for model
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# get weights
items_info <- fread("items.csv", na.strings = c(""))

weights_train <- X_train %>% 
  left_join(y = items_info, by = c("item_nbr")) %>% 
  pull(perishable)

weights_train <- (weights_train == 1) * 0.25 + 1

weights_val <- X_val %>% 
  left_join(y = items_info, by = c("item_nbr")) %>% 
  pull(perishable)

weights_val <- (weights_val == 1) * 0.25 + 1

cat("Initializing prediction tables")
val_pred <- X_val[, .(store_nbr,item_nbr)]

cat("Setting up X and Y matrices")
X_train_in <- data.matrix(X_train[, -c("store_nbr", "item_nbr")])
X_val_in <- data.matrix(X_val[, -c("store_nbr", "item_nbr")])

feature_names <- colnames(X_train_in)

end_time <- Sys.time()
writeLines("Time taken to create train and validation sets:")
print(end_time - start_time)

save(X_train, y_train, X_val, y_val, file = "modelling_data.RData")

cat("Cleaning up before building model")
rm(df_2017, promo_2017, X_train, X_val); gc()
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Tune model against val and get predictions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
cat("Setting Parameters\n")
MAX_ROUNDS <- 500
params <- list(objective = "reg:linear"
               ,booster = "gbtree"
               ,tree_method = "hist"
               ,min_child_weight = 300
               ,eta = .1
               ,colsample_bytree = .6
               ,subsample = .8
               ,eval_metric = "rmse")

cat("Set up xgb.Dmatrix\n")

dtrain <- xgb.DMatrix(X_train_in, 
                      label = seq_len(nrow(X_train_in)),
                      weight = weights_train)
dval <- xgb.DMatrix(X_val_in, 
                    label = seq_len(nrow(X_val_in)), 
                    weight = weights_val)

rm(X_train_in, X_val_in); gc()

val_dates <- seq.Date(as.Date("2017-07-26"), by = 1, length.out = 16)
results_table <- data.table(date = val_dates,
                            day = wday(val_dates, label = T),
                            num_trees = double(16),
                            val_rmse = double(16))

cat("Tune Model and get validation results\n")
start_time <- Sys.time()
for(i in 0:15) {
  cat("Round:", i + 1, "of 16\n")
  y_col = paste0("y_",i + 1)
  
  setinfo(dtrain, "label", y_train[[y_col]])
  setinfo(dval, "label", y_val[[y_col]])
  
  set.seed(i * 2)
  bst <- xgb.train(data = dtrain,
                   params = params,
                   watchlist = list(val = dval),
                   nround = MAX_ROUNDS,
                   early_stopping_rounds = 50,
                   verbose = 1,
                   print_every_n = 40)
  
  results_table[i + 1, num_trees := bst$best_ntreelimit]
  results_table[i + 1, val_rmse := bst$best_score]
  
  # assign(paste0("importance_", i), xgb.importance(model = bst))
  
  saveRDS(bst, file = paste0("./models/bst_", i, ".rds"))
  
  # Make predictions
  
  val_pred[, (y_col) := predict(object = bst, newdata = dval, ntreelimit = bst$best_ntreelimit)]
  rm(bst); gc()
}
end_time <- Sys.time()

writeLines("Time taken to build/tune 16 xgboosts:")
print(end_time - start_time)

# save(list = c("feature_names", grep("^importance_", ls(), value = T)), 
#      file = "models/variable_importance.RData")
save(results_table, val_pred, file = "predictions_results.RData")

sink()