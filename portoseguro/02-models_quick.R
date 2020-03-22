# Quick and dirty models
library(caret)
library(xgboost)

# quick and dirty xgboost(s) ---------------------------------------------

# parameters determined via caret tuning
# also tried upscaling; had little effect

load(train_x_dummy)

dtrain <- xgb.DMatrix(data = data.matrix(train_x_dummy),
                      label = ifelse(train_y == "C", 1, 0))

event_rate <- sum(getinfo(dtrain, "label"))/nrow(dtrain)
  
xgb.params <- list(eta = 0.1, 
                   max_depth = 5, 
                   objective = "binary:logistic",
                   colsample_bytree = 0.5,
                   min_child_weight = 1/sqrt(event_rate))

xgb_0 <- xgb.train(data = dtrain,
                   params = xgb.params,
                   nrounds = 200,
                   eval_metric = "auc"
                   )

# Initial submission - benchmark 

score_xgb_0 <- predict(xgb_0, newdata = xgb.DMatrix(data.matrix(test_x_dummy)))

options(scipen = 999)
subm_xgb_0 <- cbind(id = test_id, target = score_xgb_0)

write.csv(subm_xgb_0, ".\\submissions\\subm_xgb_0.csv", row.names = F)
