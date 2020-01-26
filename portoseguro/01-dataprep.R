# Data prep
library(tidyverse)
library(ggplot2)
# Base on all available information (test set included)
train_raw <- read_csv("train.csv", 
                      col_names = T, 
                      col_types = cols(.default = "d"), 
                      na = "-1")

test_raw <- read_csv("test.csv", 
                     col_names = T, 
                     col_types = cols(.default = "d"), 
                     na = "-1")

# split matrices into: predictors, IDs, target
train_x <- select(train_raw, -matches("target|id"))
test_x <- select(test_raw, -matches("id"))

train_id <- pull(train_raw, id)
test_id <- pull(test_raw, id)

train_y <- as.factor(ifelse(pull(train_raw, target) == 0, "N", "C"))

all_x <- bind_rows("train" = train_x, "test" = test_x, .id = "data")

# Missing value analysis ------------
all_missing <- 
  summarise_all(group_by(all_x, data), 
                  function(x){
                    sum(is.na(x))/length(x)*100
                })
all_missing <-
  gather(all_missing, variable, missing_pct, -data)

ggplot(data = all_missing[all_missing$missing_pct > 0, ], 
       aes(y = missing_pct, x = variable, fill = data)) +
  geom_col(position = "dodge") + coord_flip()

  # two variables with considerably higher missing value proportions in each dataset - drop these
  # ps_car_05_cat ; ps_car_03_cat

ggplot(data = all_missing[all_missing$missing_pct > 0, ], 
       aes(y = missing_pct, x = variable, fill = data)) +
  geom_col(position = "dodge") +
  coord_flip(ylim = c(0,5)) 
  
  # otherwise, there seems to be a very even distribution of missing values in each of the test/train sets - good

# Factor levels analysis ------------
# Due to unlabelled columns, we can't be sure whether columns are ordinal
all_fctr_lvls <- 
  summarise_at(.tbl = all_x, 
               .vars = ends_with("_cat", vars = names(all_x)),
               .funs = function(x){
                 length(unique(x[!is.na(x)]))
               })
all_fctr_lvls <-
  gather(all_fctr_lvls, variable, factor_levels)

ggplot(data = all_fctr_lvls, 
       aes(y = factor_levels, x = variable)) +
  geom_col() + coord_flip()

  # Highest factor level is 104 - too many levels, look to coarse class later on
  # ps_car_11_cat

# Drop variables that require further investigation
all_x <-
  select(all_x, -ps_car_05_cat, -ps_car_03_cat, -ps_car_11_cat)

# Recode categorical variables for clarity
LETTERS_ext <- c(LETTERS, paste0(LETTERS, LETTERS))

levels_conversion <-
  data.frame(cbind(numbers = 1:length(LETTERS_ext)-1,
                   letters = LETTERS_ext), stringsAsFactors = F)
levels_conversion$numbers <- as.numeric(levels_conversion$numbers)

convert_cat <- function(x){
  as.factor(
    ifelse(is.na(
      levels_conversion[match(x, levels_conversion[, "numbers"]), "letters"]),
      "Miss", levels_conversion[match(x, levels_conversion[, "numbers"]), "letters"]))
}

all_x <-
  mutate_at(.tbl = all_x,
            .vars = ends_with("_cat", vars = names(all_x)),
            .funs = convert_cat
            )

all_x <- 
  mutate_at(.tbl = all_x,
            .vars = -ends_with("_cat", vars = names(all_x)),
            .funs = function(x){
              ifelse(is.na(x), -999, x)
            })

save(all_x, file = "all_x.RData")

rm(all_fctr_lvls, all_missing, test_raw, test_x, train_raw, train_x, LETTERS_ext, levels_conversion)

# dummify --------------------------

dm <- caret::dummyVars(~., all_x, fullRank = T, sep = ".")
all_x_dummy <- data.frame(predict(dm, all_x))

save(all_x_dummy, file = "all_x_dummy.RData")  # missing values as distinct category
load(file = "all_x_dummy.RData")

train_x_dummy <- filter(.data = all_x_dummy,
                        datatrain == 1)[, -1]

test_x_dummy <- filter(.data = all_x_dummy,
                       datatrain == 0)[, -1]

save(train_x_dummy, test_x_dummy, train_y, train_id, test_id, file = "prepped_data_v0.RData")