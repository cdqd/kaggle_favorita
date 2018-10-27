# Load libraries and data

rm(list = ls()); gc()
library(tidyverse)
library(data.table)
library(xgboost)
library(ggplot2)

# Raw data
load("modelling_data.RData")
load("predictions_results.RData")
rm(X_train, y_train, X_val)
# xgb models
for (i in 0:15){
  assign(paste0("bst_", i), 
         xgb.Booster.complete(readRDS(paste0("models/bst_", i, ".rds"))))
}

# variable importance
varimp <- list()
for (i in 0:15) {
  varimp[[i+1]] <-
    xgb.importance(model = get(paste0("bst_", i)))
}

varimp_all <-
  bind_rows(varimp, .id = "bst")%>%
  mutate(bst = as.numeric(bst) - 1)

# item info and store info
items <- read_csv("items.csv")
stores <- read_csv("stores.csv")

# helper function to fix dplyr bug
summary_helper <- 
  function(data, groupby, logscale){
    sum1 <- function(x){
      sum(if(logscale == F) {expm1(x)} else {x}, 
          na.rm = T)
    }
    groupby <- substitute(groupby)
    data %>%  
      group_by(!!groupby) %>% 
      summarise(y_1 = sum1(y_1),
                y_2 = sum1(y_2),
                y_3 = sum1(y_3),
                y_4 = sum1(y_4),
                y_5 = sum1(y_5),
                y_6 = sum1(y_6),
                y_7 = sum1(y_7),
                y_8 = sum1(y_8),
                y_9 = sum1(y_9),
                y_10 = sum1(y_10),
                y_11 = sum1(y_11),
                y_12 = sum1(y_12),
                y_13 = sum1(y_13),
                y_14 = sum1(y_14),
                y_15 = sum1(y_15),
                y_16 = sum1(y_16)) %>% 
      ungroup()
  }
