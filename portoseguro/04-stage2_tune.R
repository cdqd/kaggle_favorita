# tuning stage 2 using best predictions vectors from stage 1 ('best_pred')
# 'final_pred' are those models trained on the full training set with optimized parameters
# Libraries and load data --------------------
library(caret)
library(tidyverse)

load("models/holdout_preds.RData")
load("models/lgbc_holdout_preds.RData")
load("prepped_data_dense.RData")
rm(test_x, train_x)

### rank averaging -----------------
# check correlations
hpred <- cbind(best_cart_fp_pred,
               best_cart_pred,
               best_glmnet_pred,
               best_lgb_c1_pred,
               best_lgb_c2_pred,
               best_lgb_c3_pred,
               best_rf_fp_pred,
               best_rf_pred,
               best_xgb_pred,
               gen2_best_xgb_pred,
               gen2_best_lgb_pred)

cor_hpred <- cor(hpred)

cor_avg <- mean(cor_hpred[upper.tri(cor_hpred)])
cor_each <- colMeans(cor_hpred)

# drop unwanted, some manual tuning required
hpred <- data.frame(cbind(
                          # best_cart_fp_pred,
                          # best_cart_pred,
                          # best_glmnet_pred,
                          best_lgb_c1_pred,
                          best_lgb_c2_pred,
                          best_lgb_c3_pred,
                          best_xgb_pred,
                          gen2_best_xgb_pred,
                          gen2_best_lgb_pred))
hrank <- mutate_all(hpred, rank)
names(hrank) <- gsub("_pred", "_rank", names(hrank))

rank_avg_simple <- apply(hrank, 1, mean)
rank_avg_pred <- rank_avg_simple / sum(rank_avg_simple)

# evaluate model
2  * fastAUC(rank_avg_pred, as.numeric(train_y[stage == "s2"] == "C")) - 1

# Try harmonic means
a <- 1 / hpred
harm_pred <- apply(a, 1, function(x){
  1 / mean(x)
})

# evaluate model
2  * fastAUC(harm_pred, as.numeric(train_y[stage == "s2"] == "C")) - 1

# try glm ---------------------
folds <- createFolds(train_y[stage == "s2"], k = 5)

hpred_try <- data.frame(cbind(best_cart_fp_pred,
                   best_cart_pred,
                   best_glmnet_pred,
                   best_lgb_c1_pred,
                   best_lgb_c2_pred,
                   best_lgb_c3_pred,
                   best_xgb_pred,
                   gen2_best_xgb_pred,
                   gen2_best_lgb_pred))

# cross val loop, hand tune ----------------
res <- numeric()
for(i in 1:5){
  gc()
  train <- cbind(y = as.numeric(train_y[stage == "s2"][-folds[[i]]] == "C"),
                 hpred_try[-folds[[i]], 
                           # -c(1, 2, 3, 4, 5, 6, 8)
                           c(6, 7, 9)
                           ])
  glm_0 <- glm(formula = train, family = "binomial", model = F, x = F, y = F)
  
  preds <- predict(glm_0, hpred_try[folds[[i]], ], type = "response")
  
  res[i] <- fastAUC(preds, as.numeric(train_y[stage == "s2"][folds[[i]]] == "C"))
}
2 * mean(res) - 1  # with all variables, 0.27925, so worse than best individual model
                   # final result 0.2822 - keep xgb, lgb, lgb_c3 what we've already done
                  
train <- cbind(y = as.numeric(train_y[stage == "s2"]== "C"),
               hpred_try[ , c(6, 7, 9)])
glm_0 <- glm(train, family = "binomial", model = F, x = F, y = F)

###### make final submissions -------------
load("models/xgbs_tpred.RData")
load("models/lgbc_tpred.RData")
load("models/lgbf_tpred.RData")
load("prepped_data_dense.RData")

# rank average
tpred <- data.frame(cbind(gen2_xgb_final_pred,
                          xgb_final_pred,
                          gen2_lgb_final_pred))

trank <- mutate_all(tpred, rank)
names(trank) <- gsub("_pred", "_rank", names(trank))

rank_avg_simple <- apply(trank, 1, mean)
rank_avg_pred <- rank_avg_simple / sum(rank_avg_simple)

options(scipen = 999)
write.csv(cbind(id = test_id, target = rank_avg_pred),
          file = paste0("../submissions/submission_", 5, ".csv"),
          row.names = F)

# harmonic mean average
a <- 1 / tpred
harm_pred <- apply(a, 1, function(x){
  1 / mean(x)})
  
options(scipen = 999)
write.csv(cbind(id = test_id, target = harm_pred),
          file = paste0("../submissions/submission_", 6, ".csv"),
          row.names = F)

# glm with variables 6, 7, 9 = lgb_c3, xgb_best, gen2_lgb

tpred_try <- data.frame(cbind(final_lgb_c3_pred,
                   xgb_final_pred,
                   gen2_lgb_final_pred))
subm_glm <- glm_0$coefficients[[1]] +
  glm_0$coefficients[[2]]*tpred_try[, 1] +
  glm_0$coefficients[[3]]*tpred_try[, 2] +
  glm_0$coefficients[[4]]*tpred_try[, 3]

subm_glm <- exp(subm_glm)/(1 + exp(subm_glm))

options(scipen = 999)
write.csv(cbind(id = test_id, target = subm_glm),
          file = paste0("./submissions/submission_", 9, ".csv"),
          row.names = F)