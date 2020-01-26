# Some examples of tuning scripts used to select optimal parameters for each stage 1 model

# Libraries ---------------------------------------
library(dplyr)
library(randomForest)
library(ranger)
library(rpart)
library(rpart.plot)
library(xgboost)
library(glmnet)

# Functions ---------------------------------------

fastAUC <- function(prediction, truth) {
  x1 <- prediction[truth == 1]; n1 = length(x1); 
  x2 <- prediction[truth == 0]; n2 = length(x2);
  r <- rank(c(x1,x2))  
  (sum(r[1:n1]) - n1 * (n1 + 1) / 2) / n1 / n2
}

# Additional dataprep -----------------------------
# split stage 1 and stage 2 training via row numbers
index_target <- which(train_y == "C")

pscar13_target <- 
  index_target[order(train_x_dummy$ps_car_13[index_target])]
pscar13_non_target <- 
  setdiff(1:nrow(train_x_dummy), index_target)[order(train_x_dummy$ps_car_13[-index_target])]

stage2_target <- pscar13_target[seq(5, length(pscar13_target), by = 5)]
stage2_non_target <- pscar13_non_target[seq(5, length(pscar13_non_target), by = 5)]

train_x_dummy$stage <- character(nrow(train_x_dummy))
train_x_dummy$stage[c(stage2_non_target, stage2_target)] <- "s2"
train_x_dummy$stage[-c(stage2_non_target, stage2_target)] <- "s1"
train_x_dummy$stage <- as.factor(train_x_dummy$stage)

rm(index_target, pscar13_target, pscar13_non_target, stage2_target, stage2_non_target)

# check distribution equality
ggplot(data = train_x_dummy, aes(x = ps_car_13, fill = stage)) + 
  geom_density(position = "dodge", bw = 0.05) + facet_grid( ~ stage)

ggplot(data = cbind(train_x_dummy,train_y), aes(x = train_y, fill = stage)) + 
  geom_bar(position = "fill", stat = "count")

# stage 1 models - tuning on dummified dataset--------------------------------------------------

stage <- pull(train_x_dummy, stage)
train_x_dummy$stage <- NULL

# trial ranger -------
# when holdout specified, "predicitions" should only be generated on holdout
# i.e. wherever it refers to "out of bag samples" in the documentation, it means the holdout set
sink(file = "./models/ranger.log", append = T, split = T)

grid_rf <- expand.grid(mtry = 2 * c(1:15))
writeLines("##Tuning RF model--")

start_time <- Sys.time()
for(i in 1:nrow(grid_rf)){
  
  rf <- ranger(data = cbind(train_x_dummy, train_y),
               num.trees = 2000,
               mtry = grid_rf[i, "mtry"],
               write.forest = F,
               probability = T,
               case.weights = as.numeric(stage == "s1"),
               holdout = T,
               verbose = T,
               seed = 1234,
               dependent.variable.name = "train_y")
  gini <- fastAUC(prediction = rf$predictions[!is.nan(rf$predictions[, "C"]), "C"],
                   truth = as.numeric(train_y[stage == "s2"] == "C")) * 2 - 1
  grid_rf$holdout_gini[i] <- gini
  
  writeLines(paste("##Iteration", i, "complete--"))
  writeLines(paste("## Parms and results--", paste(colnames(grid_rf), collapse = "|")))
  writeLines(paste("######################", paste(round(grid_rf[i, ], 3), collapse = "|")))
  
  if (i == 1) best_rf_pred <- rf$predictions[!is.nan(rf$predictions[, "C"]), "C"]
  
  if (i > 1 & gini == max(grid_rf$holdout_gini, na.rm = T)){
      best_rf_pred <- rf$predictions[!is.nan(rf$predictions[, "C"]), "C"]
    }
  
  rm(rf)
  gc()
}
end_time <- Sys.time()
writeLines(paste("Time taken to tune ranger:"))
start_time - end_time
sink() 

# trial rpart ---------------------------------------------------------
sink(file = "./models/rpart.log", append = T, split = T)

writeLines("##Fitting CART to full stage 1 training set--")
start_time <- Sys.time()
cart <- rpart(cbind(train_y[stage == "s1"], train_x_dummy[stage == "s1", ]),
              method = "class",
              model = F,
              x = F,
              y = F,
              parms = list(split = "information"),
              control = rpart.control(
                minsplit = 1,
                minbucket = 1,
                cp = 0,
                xval = 0
              ))
end_time <- Sys.time()
writeLines("Time taken to build full rpart:")
end_time - start_time

grid_cart <- expand.grid(cp = unname(cart$cptable[, "CP"])[1:floor(2/3 * nrow(cart$cptable))])
writeLines("##CP grid created with 2/3 of possible CPs")

writeLines("##Tuning CART model--")
start_time <- Sys.time()
for(i in 1:nrow(grid_cart)){
  cart_p <- prune(cart, grid_cart[i, 1])
  gini <- fastAUC(prediction = unname(predict(cart_p, train_x_dummy[stage == "s2", ])[, "C"]),
                   truth = as.numeric(train_y[stage == "s2"] == "C")) * 2 - 1
  grid_cart$holdout_gini[i] <- gini
  
  writeLines(paste("##Iteration", i, "complete--"))
  writeLines(paste("## Parms and results--", paste(colnames(grid_cart), collapse = "|")))
  writeLines(paste("######################", paste(round(grid_cart[i, ], 3), collapse = "|")))
  
  if (i == 1) best_cart_pred <- unname(predict(cart_p, train_x_dummy[stage == "s2", ])[, "C"])
  
  if (i > 1 & gini == max(grid_cart$holdout_gini, na.rm = T)){
    best_cart_pred <- unname(predict(cart_p, train_x_dummy[stage == "s2", ])[, "C"])
    writeLines("Best cart predictions updated")
  }
  
  rm(cart_p)
  gc()
}
end_time <- Sys.time()
writeLines("Time taken to tune rpart:")
end_time - start_time

sink()

# trial xgboost -------------------------------------------------------

# knowing that the parameters from the initial xgboost are optimal relatively to a wider search,
# these searches should be closer in proximity to those initial parameters.
# leave eta constant at 0.1
# for now, leave gamma and subsample alone (0 and 1 respectively)

sink(file = "./models/xgb.log", append = T, split = T)
writeLines("##Setting up grid search and datasets--")
grid_xgb <- expand.grid(colsample_bytree = c(0.2, 0.3, 0.4),
                        max_depth = c(4, 5),
                        min_child_weight = c(9, 10.5, 12))

dtrain <- xgb.DMatrix(data.matrix(train_x_dummy[stage == "s1", ]), 
                      label = as.numeric(train_y[stage == "s1"] == "C"))

dtest <- xgb.DMatrix(data.matrix(train_x_dummy[stage == "s2", ]), 
                      label = as.numeric(train_y[stage == "s2"] == "C"))

writeLines("##Tuning xgboost--")
start_time <- Sys.time()
for (i in 1:nrow(grid_xgb)){
  writeLines(paste("##Tuning iteration:", i))
  set.seed(1234)
  xgb <- xgb.train(data = dtrain,
                   params = list(objective = "binary:logistic",
                                  eta = 0.1,
                                  colsample_bytree = grid_xgb[i, "colsample_bytree"],
                                  max_depth = grid_xgb[i, "max_depth"],
                                  min_child_weight = grid_xgb[i, "min_child_weight"]),
                   nrounds = 200,
                   eval_metric = "auc",
                   maximize = T,
                   watchlist = list(train = dtrain, test = dtest),
                   early_stopping_rounds = 20, 
                   verbose = 1,
                   print_every_n = 20)
  
  grid_xgb$iter[i] <- xgb$best_iteration
  grid_xgb$holdout_gini[i] <- unname(xgb$best_score)

  writeLines(paste("##Iteration", i, "complete--"))
  writeLines(paste("## Parms and results--", paste(colnames(grid_xgb), collapse = "|")))
  writeLines(paste("######################", paste(round(grid_xgb[i, ], 3), collapse = "|")))
  
  if (i == 1) best_xgb_pred <- predict(xgb, dtest)
  
  if (i > 1 & xgb$best_score == max(grid_xgb$holdout_gini, na.rm = T)){
    best_xgb_pred <- predict(xgb, dtest)
  }
rm(xgb)
gc()
}
end_time <- Sys.time()
writeLines("Time taken to tune xgboost:")
end_time - start_time

sink()

# trial glmnet -------------------------------------------------------
# we'll use the default lambda sequence supplied by glmnet
sink("./models/glmnet.log", append = T, split = T)

grid_glmnet <- expand.grid(alpha = seq(0, 1, by = 0.05))

train_y_glmnet <- as.factor(ifelse(train_y == "C", "B", "A"))  # B is target (required by glmnet)

start_time <- Sys.time()
for(j in 1:nrow(grid_glmnet)){
  writeLines(paste("##Training alpha iteration", j))
  glmnet <- glmnet(x = data.matrix(train_x_dummy[stage == "s1", ]),
                   y = train_y_glmnet[stage == "s1"],  
                   family = "binomial",
                   alpha = grid_glmnet[j, "alpha"])  
  
  glmnet_pred <- predict(glmnet,
                         newx = data.matrix(train_x_dummy[stage == "s2", ]),
                         type = "response")
  
  best_lambda <- glmnet$lambda[1]
  best_lambda_gini <- fastAUC(prediction = unname(glmnet_pred[, 1]),
                               truth = as.numeric(train_y[stage == "s2"] == "C")) * 2 - 1
  
  writeLines(paste("##Lambda optimisation start--"))
  
  start_time_1 <- Sys.time()
  for (i in 2:ncol(glmnet_pred)){
    gini <- fastAUC(prediction = unname(glmnet_pred[, i]),
                     truth = as.numeric(train_y[stage == "s2"] == "C")) * 2 - 1
    if(gini > best_lambda_gini){
      best_lambda <- glmnet$lambda[i]
      best_lambda_gini <- gini
      writeLines(paste("Best lambda updated, lambda =", glmnet$lambda[i], "| holdout gini =", gini))
    }
  }
  end_time_1 <- Sys.time()
  writeLines(paste("##Lambda optimisation for alpha iteration", j, "finished. Time:"))
  end_time_1 - start_time_1
  
  grid_glmnet$lambda[j] <- best_lambda
  grid_glmnet$holdout_gini[j] <- best_lambda_gini
  gc()
}
end_time <- Sys.time()
writeLines("Time taken to tune glmnet:")
end_time - start_time

sink()

# trial lgb ----------------------------------
load(train_x)  # load a version of the factor-preserved dataset

cat_features <- grep("_cat", names(train_x), value = T)

# prepare lgb
dtrain_lgb <- lgb.Dataset(data = data.matrix(train_x[stage == "s1", ]), 
                          label = as.numeric(train_y[stage == "s1"] == "C"),
                          free_raw_data = F)

dtest_lgb <- lgb.Dataset(data = data.matrix(train_x[stage == "s2", ]), 
                         label = as.numeric(train_y[stage == "s2"] == "C"),
                         free_raw_data = F)

grid_lgb <- expand.grid(feature_fraction = seq(0.4, 1, by  = 0.2),
                        max_depth = seq(4, 10, by = 2),
                        bagging_fraction = seq(0.5, 1, by = 0.25),
                        min_data_in_leaf = c(50, 100))


sink(file = "./models/lgb.log", append = T, split = T)

writeLines("##Tuning lgboost--")
start_time <- Sys.time()
for (i in 1:nrow(grid_lgb)){
  writeLines(paste("##Starting tuning iteration", i, "----"))
  lightgbm <- lgb.train(data = dtrain_lgb,
                        valids = list(test = dtest_lgb),
                        categorical_feature = cat_features,
                        verbose = -1,
                        objective = "binary",
                        eval = "auc",
                        learning_rate = 0.1,
                        nrounds = 500,
                        max_depth = grid_lgb[i, "max_depth"],
                        min_data_in_leaf = grid_lgb[i, "min_data_in_leaf"],
                        feature_fraction = grid_lgb[i, "feature_fraction"],
                        feature_fraction_seed = 2,
                        bagging_fraction = grid_lgb[i, "bagging_fraction"],
                        bagging_freq = 1,
                        bagging_seed = 1234,
                        early_stopping_round = 40)
  
  grid_lgb$iter[i] <- lightgbm$best_iter
  grid_lgb$holdout_gini[i] <- 2 * lightgbm$record_evals$test$auc$eval[[lightgbm$best_iter]] - 1  
  
  writeLines(paste(round(grid_lgb[i, ], 6), collapse = "|"))
  
  if (i == 1) gen2_best_lgb_pred <- predict(lightgbm, data.matrix(train_x[stage == "s2", ]),
                                            num_iteration = lightgbm$best_iter)
  
  if (i > 1 & 2 * lightgbm$record_evals$test$auc$eval[[lightgbm$best_iter]] - 1 == 
      max(grid_lgb$holdout_gini, na.rm = T)){
    gen2_best_lgb_pred <- predict(lightgbm, data.matrix(train_x[stage == "s2", ]),
                                  num_iteration = lightgbm$best_iter)
    writeLines("##Best LGB predictions updated----")
  }
  rm(lightgbm)
  gc()
}

end_time <- Sys.time()
writeLines("Time taken to tune lgboost:")
end_time - start_time

sink()