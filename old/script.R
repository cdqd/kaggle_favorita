# This is a translation of Ceshine Lee's LGBM Starter kernel (written in Python) found here:
# https://www.kaggle.com/ceshine/lgbm-starter

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Loading and Cleaning Data
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
rm(list = ls()); gc()

start_time <- Sys.time()
cat("Setting up Enviornment\n")
library(data.table)
library(lightgbm)
library(dplyr)
library(lubridate)
library(magrittr)

cat("Loading test and train")
df_2017 <- fread("../train.csv", sep=",", na.strings="", skip = 90000000,
               col.names=c("id","date","store_nbr","item_nbr","unit_sales","onpromotion"))

df_2017[, id := NULL]
gc()
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

delete(df_2017, which(df_2017$date < "2017-01-01"))
gc()
cols <- names(df_2017)

test <- fread("../test.csv", sep=",", na.strings = "")

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
                   ,p0_30_2017 = df_2017[, apply(.SD, 1, function(x) quantile(x, 0)), .SDcols = m30_names]
                   ,p0_60_2017 = df_2017[, apply(.SD, 1, function(x) quantile(x, 0)), .SDcols = m60_names]
                   ,p25_30_2017 = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.25)), .SDcols = m30_names]
                   ,p25_60_2017 = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.25)), .SDcols = m60_names]
                   ,p50_30_2017 = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.5)), .SDcols = m30_names]
                   ,p50_60_2017 = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.5)), .SDcols = m60_names]
                   ,p75_30_2017 = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.75)), .SDcols = m30_names]
                   ,p75_60_2017 = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.75)), .SDcols = m60_names]
                   ,p100_30_2017 = df_2017[, apply(.SD, 1, function(x) quantile(x, 1)), .SDcols = m30_names]
                   ,p100_60_2017 = df_2017[, apply(.SD, 1, function(x) quantile(x, 1)), .SDcols = m60_names]
                   ,p0_dow20_Mon = df_2017[, apply(.SD, 1, function(x) quantile(x, 0)), .SDcols = dow20namesMon]
                   ,p0_dow20_Tue = df_2017[, apply(.SD, 1, function(x) quantile(x, 0)), .SDcols = dow20namesTue]
                   ,p0_dow20_Wed = df_2017[, apply(.SD, 1, function(x) quantile(x, 0)), .SDcols = dow20namesWed]
                   ,p0_dow20_Thu = df_2017[, apply(.SD, 1, function(x) quantile(x, 0)), .SDcols = dow20namesThu]
                   ,p0_dow20_Fri = df_2017[, apply(.SD, 1, function(x) quantile(x, 0)), .SDcols = dow20namesFri]
                   ,p0_dow20_Sat = df_2017[, apply(.SD, 1, function(x) quantile(x, 0)), .SDcols = dow20namesSat]
                   ,p0_dow20_Sun = df_2017[, apply(.SD, 1, function(x) quantile(x, 0)), .SDcols = dow20namesSun]
                   ,p25_dow20_Mon = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.25)), .SDcols = dow20namesMon]
                   ,p25_dow20_Tue = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.25)), .SDcols = dow20namesTue]
                   ,p25_dow20_Wed = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.25)), .SDcols = dow20namesWed]
                   ,p25_dow20_Thu = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.25)), .SDcols = dow20namesThu]
                   ,p25_dow20_Fri = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.25)), .SDcols = dow20namesFri]
                   ,p25_dow20_Sat = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.25)), .SDcols = dow20namesSat]
                   ,p25_dow20_Sun = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.25)), .SDcols = dow20namesSun]
                   ,p50_dow20_Mon = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.5)), .SDcols = dow20namesMon]
                   ,p50_dow20_Tue = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.5)), .SDcols = dow20namesTue]
                   ,p50_dow20_Wed = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.5)), .SDcols = dow20namesWed]
                   ,p50_dow20_Thu = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.5)), .SDcols = dow20namesThu]
                   ,p50_dow20_Fri = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.5)), .SDcols = dow20namesFri]
                   ,p50_dow20_Sat = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.5)), .SDcols = dow20namesSat]
                   ,p50_dow20_Sun = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.5)), .SDcols = dow20namesSun]
                   ,p75_dow20_Mon = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.75)), .SDcols = dow20namesMon]
                   ,p75_dow20_Tue = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.75)), .SDcols = dow20namesTue]
                   ,p75_dow20_Wed = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.75)), .SDcols = dow20namesWed]
                   ,p75_dow20_Thu = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.75)), .SDcols = dow20namesThu]
                   ,p75_dow20_Fri = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.75)), .SDcols = dow20namesFri]
                   ,p75_dow20_Sat = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.75)), .SDcols = dow20namesSat]
                   ,p75_dow20_Sun = df_2017[, apply(.SD, 1, function(x) quantile(x, 0.75)), .SDcols = dow20namesSun]
                   ,p100_dow20_Mon = df_2017[, apply(.SD, 1, function(x) quantile(x, 1)), .SDcols = dow20namesMon]
                   ,p100_dow20_Tue = df_2017[, apply(.SD, 1, function(x) quantile(x, 1)), .SDcols = dow20namesTue]
                   ,p100_dow20_Wed = df_2017[, apply(.SD, 1, function(x) quantile(x, 1)), .SDcols = dow20namesWed]
                   ,p100_dow20_Thu = df_2017[, apply(.SD, 1, function(x) quantile(x, 1)), .SDcols = dow20namesThu]
                   ,p100_dow20_Fri = df_2017[, apply(.SD, 1, function(x) quantile(x, 1)), .SDcols = dow20namesFri]
                   ,p100_dow20_Sat = df_2017[, apply(.SD, 1, function(x) quantile(x, 1)), .SDcols = dow20namesSat]
                   ,p100_dow20_Sun = df_2017[, apply(.SD, 1, function(x) quantile(x, 1)), .SDcols = dow20namesSun]
  )
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
  results <- prepare_dataset(t2017+delta)
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
# Can't produce results$y since that is what we are predicting
results <- prepare_dataset(as.Date("2017-08-16"), is_train = F)
X_test <- results$X
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Preparing data for model
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
cat("Ordering all data.tables")
# Train set
setorderv(X_train, c("store_nbr", "item_nbr"))
setorderv(y_train, c("store_nbr", "item_nbr"))
# Val set
setorderv(X_val, c("store_nbr", "item_nbr"))
setorderv(y_val, c("store_nbr", "item_nbr"))
# Test set
setorderv(X_test, c("store_nbr", "item_nbr"))

# get weights
items_info <- fread("../items.csv", na.strings = c(""))

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
test_pred <- X_test[, .(store_nbr,item_nbr)]

cat("Setting up X and Y matrices")
X_train_in = as.matrix(X_train[, -c("store_nbr", "item_nbr")])
X_val_in = as.matrix(X_val[, -c("store_nbr", "item_nbr")])
X_test_in = as.matrix(X_test[, -c("store_nbr","item_nbr")])

save(X_train, y_train, X_val, y_val, X_test, file = "modelling_data.RData")

cat("Cleaning up before building model")
rm(df_2017, promo_2017, X_train, X_val, X_test); gc()
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Build Model and Getting Predictions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
cat("Setting Parameters\n")
MAX_ROUNDS = 500
params <- list(num_leaves = 31
               ,objective = "regression_l2"
               ,min_data_in_leaf = 300
               ,learning_rate = .1
               ,feature_fraction = .8
               ,feature_fraction_seed = 242
               ,bagging_fraction = .8
               ,bagging_freq = 2
               ,bagging_seed = 383
               ,metric = "rmse"
               ,num_threads = 4)

cat("Building Model")

num_trees <- numeric(0)
for(i in 0:15) {
  cat("Round:",i,"of 15\n")
  y_col = paste0("y_",i+1)
  dtrain <- lgb.Dataset(X_train_in, 
                        label = as.matrix(y_train[, y_col, with = F]),
                        weight = weights_train)
  dval <- lgb.Dataset(X_val_in, 
                      label = as.matrix(y_val[, y_col, with = F]), 
                      weight = weights_val,
                      reference = dtrain)
  bst <- lgb.train(params, dtrain, nrounds = MAX_ROUNDS
                   , valids = list(test=dval), early_stopping_rounds = 50
                   , verbose = 1)
  
  num_trees[i + 1] <- bst$best_iter
  lgb.save(bst, filename = paste0("./models/lgbm_starter/bst_", i, ".txt"))
  
  # Make predictions
  
  val_pred[, eval(y_col) := bst$predict(X_val_in, num_iteration = bst$best_iter)]
  test_pred[, eval(y_col) := bst$predict(X_test_in, num_iteration = bst$best_iter)]
}

save(val_pred, file = "val_predictions.RData")
# 
# # colnames are originally y_1, y_2, ... y_16. Switching to test dates for submission
# colnames(test_pred) <- c("store_nbr", "item_nbr", sort(unique(test$date)))
# # Make wide table long (columns are: store_nbr, item_nbr, date, unit_sales)
# test_pred_long <- melt(test_pred, measure.vars = sort(unique(test$date)),
#                        variable.name="date", value.name="unit_sales")
# test_pred_long[, date := as.character(date)]
# 
# # Merge with original test set
# test <- test %>% 
#   left_join(y = test_pred_long,
#             by = c("store_nbr", "item_nbr", "date"))
# setDT(test)
# test[unit_sales < 0, unit_sales := 0]
# 
# end_time <- Sys.time()
# print(end_time - start_time)
# 
# # save(test, file = "test_1.RData")
# # rm(list = ls()); gc()
# # load("test_1.RData")
# 
# # no imputation -------------------------------------------
# 
# test[which(is.na(unit_sales)), unit_sales := 0]
# 
# cat("Making submission")
# test[, `:=`(id = bit64::as.integer.integer64(id),
#             unit_sales = expm1(unit_sales))]
# fwrite(test[, .(id, unit_sales)], "./submissions/LGBM_sub_noimpute.csv", 
#        sep = ",", dec = ".", quote = F, row.names = F)
# # ---------------------------------------------------------