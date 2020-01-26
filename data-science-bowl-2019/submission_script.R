rm(list = ls()); gc()

library(data.table)
library(glue)
library(logger)
library(magrittr)
library(mlr)

MAIN_DIR <- ".."
source("../usr/lib/data_science_bowl_2019_utils/data_science_bowl_2019_utils.R")
options(scipen = 999)
configureMlr(on.par.without.desc = "warn")
INPUT_PATH <- glue("{MAIN_DIR}/input/data-science-bowl-2019")
CUSTOM_DATA_DIR <- glue("{MAIN_DIR}/input/custom-event-id-mapping")

set.seed(1204)
# Insert optimal parameters from tuning here
XGBOOST_PARAMS <- list(
  eta = 0.02,
  max_depth = 10,
  min_child_weight = 50,
  colsample_bytree = 0.7,
  nrounds = 150
)
# For feature engineering
ROLLING_N_SESSIONS <- c(3L)

read_event_data <- function(type) {
  log_info("Loading {type} event data...")
  path <- if (type == "train") glue("{INPUT_PATH}/train.csv") else glue("{INPUT_PATH}/test.csv")
  fread(path, drop = c("event_data"))
}

read_custom_event_mapping <- function() {
  fread(glue("{CUSTOM_DATA_DIR}/custom_event_id_mapping.csv"))
}

get_evaluated_assessments <- function(type) {
  log_info("Getting evaluated assessments for {type} data...")
  if (type == "train") {
    .data <- fread(glue("{INPUT_PATH}/train_labels.csv"))
    result <- .data[, .(game_session, installation_id, accuracy_group)]
  } else if (type == "test") {
    .data <- fread(glue("{INPUT_PATH}/test.csv"))
    setorderv(.data, c("installation_id", "timestamp"), c(1, -1))
    .data[, desc_rank := seq_len(.N), by = "installation_id"]
    result <- .data[desc_rank == 1, .(game_session, installation_id)]
    result[, accuracy_group := 0]
  }

  result
}

get_assessed_installations <- function(event_data, assessment_data) {
  installation_ids <- unique(assessment_data[["installation_id"]])
  setorderv(event_data, "installation_id")

  event_data[installation_id %in% installation_ids]
}

set_ohe <- function(.data, factor_cols, factor_levels) {
  for (i in seq_along(factor_cols)) {
    col_name <- factor_cols[i]
    unique_vals <- factor_levels[[i]]
    alloc.col(.data, length(.data) + length(unique_vals))
    set(.data, j = paste0(col_name, "_", unique_vals), value = 0L)
    for (unique_value in unique_vals) {
      matched_rows <- which(as.character(.data[[col_name]]) == unique_value)
      set(.data, i = matched_rows, j = paste0(col_name, "_", unique_value), value = 1L)
    }
    set(.data, j = paste0(col_name, "_", "NA"), value = 0L)
    matched_na <- which(is.na(.data[[col_name]]))
    set(.data, i = matched_na, j = paste0(col_name, "_", "NA"), value = 1L)
  }
}

build_session_data <- function(type, rolling_n_sessions, thresholds_hours) {
  tr <- read_event_data(type)
  cat("Mem used:\n"); print(pryr::mem_used())
  assessed <- get_evaluated_assessments(type)
  if (type == "train") {
    # Get rid of installation ids that didn't take an assessment
    tr_a <- get_assessed_installations(tr, assessed)
  } else {
    tr_a <- tr
  }
  rm(tr)
  gc()
  
  # Custom data
  event_id_map <- read_custom_event_mapping()
  
  # Session level features
  log_info("{type} dataset - building session-level features...")
  tr_a[event_id_map, custom_event_code := i.custom_event_code, on = "event_id"]
  tr_a[, event_id := NULL]
  
  factor_cols <- c("event_code", "custom_event_code")
  factor_levels <- list(event_codes, custom_event_codes)
  for (i in seq_along(factor_levels)) {
    tr_a[, (factor_cols[i]) := factor(get(factor_cols[i]), factor_levels[[i]])]
  }
  setorderv(tr_a, c("installation_id", "timestamp", "event_count"))
  
  # For times, work in seconds, not milliseconds
  tr_a[, game_time := game_time / 1000]
  tr_a[
    ,
    prev_game_time := shift(game_time, n = 1, type = "lag"),
    by = c("installation_id", "game_session")
  ]
  tr_a[, time_between_events := game_time - prev_game_time]
  
  # Do one-hot encoding by reference to save memory
  set_ohe(tr_a, factor_cols, factor_levels)
  event_code_cols <- grep("^event_code_", names(tr_a), value = T)
  custom_event_code_cols <- grep("^custom_event_code_", names(tr_a), value = T)

  # Timing dummies
  for (col in c(event_code_cols, custom_event_code_cols)) {
    # tsgss = time since game session start
    tsgss <- tr_a[["game_time"]] * tr_a[[col]]
    set(tr_a, j = glue("tsgss_{col}"), value = tsgss)
  }
  tsgss_cols <- grep("^tsgss_", names(tr_a), value = T)

  tr_sum_dummies <- tr_a[
    ,
    lapply(.SD, sum, na.rm = T),
    .SDcols = c(event_code_cols, custom_event_code_cols),
    keyby = c("game_session")
  ]
  tr_avg_ts <- tr_a[
    ,
    lapply(.SD, mean, na.rm = T),
    .SDcols = c(tsgss_cols),
    keyby = c("game_session")
  ]

  # Main session summary
  tr_gs <- tr_a[,
    .(
      start_time = min(timestamp),
      installation_id = data.table::first(installation_id),
      type = data.table::first(type),
      world = data.table::first(world),
      title = data.table::first(title),
      total_game_time = max(game_time),
      avg_time_between_events = mean(time_between_events, na.rm = T),
      sd_time_between_events = sd(time_between_events, na.rm = T),
      n_distinct_events = length(unique(event_code)),
      n_distinct_custom_events = length(unique(custom_event_code)),
      total_events = max(event_count)
    ),
    keyby = c("game_session")
  ]
  rm(tr_a)
  gc()
  tr_gs[, start_time := as.POSIXct(lubridate::fast_strptime(start_time, "%Y-%m-%dT%H:%M:%OS%z"))]

  for (col in c("avg_time_between_events", "sd_time_between_events")) {
    set(tr_gs, which(is.nan(tr_gs[[col]])), col, NA)
  }
  for (col in names(tr_avg_ts)) {
    set(tr_avg_ts, which(is.nan(tr_avg_ts[[col]])), col, NA)
  }
  
  # Attach on event flags for each session
  event_dummy_cols <- setdiff(names(tr_sum_dummies), "game_session")
  event_ts_cols <- setdiff(names(tr_avg_ts), "game_session")
  tr_gs[
    tr_sum_dummies,
    (event_dummy_cols) := mget(paste0("i.", event_dummy_cols)),
    on = "game_session"
  ]
  tr_gs[
    tr_avg_ts,
    (event_ts_cols) := mget(paste0("i.", event_ts_cols)),
    on = "game_session"
  ]
  
  rm(tr_sum_dummies, tr_avg_ts)
  gc()
  
  # Within-game-session stats
  tr_gs[, `:=`(
    all_rounds_beat = (custom_event_code_round_beat >= custom_event_code_round_start)*1
    , correct_more = (custom_event_code_feedback_correct > custom_event_code_feedback_incorrect)*1
    , dropped_accurately_more = (custom_event_code_dropped_item > custom_event_code_dropped_item_incorrectly)*1
    , clicked_irrelevant_more = (custom_event_code_clicked_irrelevant > custom_event_code_dropped_item_incorrectly + custom_event_code_dropped_item)*1
    , avg_time_rounds_all_rounds_beat = ifelse(
      custom_event_code_round_start >= custom_event_code_round_beat,
      tsgss_custom_event_code_round_beat - tsgss_custom_event_code_round_start,
      NA
    )
    , tutorial_time_taken = tsgss_custom_event_code_tutorial_beat - tsgss_custom_event_code_tutorial_start
  )]
  
  cat("Mem used:\n"); print(pryr::mem_used())
  # Building user behavioural features for each session based on previous sessions
  log_info("{type} dataset - building user behavioural features for each session based on previous sessions")
  setorderv(tr_gs, c("installation_id", "start_time"))

  factor_cols_start <- c("type", "world", "title")
  factor_levels_start <- list(types, worlds, titles)
  for (i in seq_along(factor_cols_start)) {
    tr_gs[, (factor_cols_start[i]) := factor(get(factor_cols_start[i]), factor_levels_start[[i]])]
  }
  set_ohe(tr_gs, factor_cols_start, factor_levels_start)
  type_cols <- grep("^type_", names(tr_gs), value = T)
  world_cols <- grep("^world_", names(tr_gs), value = T)
  title_cols <- grep("^title_", names(tr_gs), value = T)

  tr_gs[, prev1_start_time := shift(start_time, n = 1, type = "lag"), by = "installation_id"]
  tr_gs[, time_since_prev1_session_start := as.numeric(start_time - prev1_start_time, "secs")]
  tr_gs[, session_number := seq_len(.N), by = "installation_id"]

  # Count columns
  count_cols <- c(
    "total_game_time",
    "n_distinct_events",
    "n_distinct_custom_events",
    "total_events",
    "all_rounds_beat",
    "correct_more",
    "dropped_accurately_more",
    "clicked_irrelevant_more",
    "avg_time_rounds_all_rounds_beat",
    "tutorial_time_taken",
    event_dummy_cols,
    event_ts_cols,
    title_cols,
    type_cols,
    world_cols
  )
  # Column that represents time of day during which user plays
  # TODO
  
  cat("Mem used:\n"); print(pryr::mem_used())
  # Calculating cumulative stats
  log_info("{type} dataset - calculating cumulative stats...")
  tr_gs[, time_since_first_session_start := cumsum_skip_na(time_since_prev1_session_start), by = "installation_id"]

  for (calc in c("sum", "mean")) {
    cumfun <- match.fun(glue("cum{calc}_skip_na"))
    tr_gs[,
      glue("cum{calc}_{count_cols}") := lapply(.SD, cumfun),
      by = "installation_id",
      .SDcols = count_cols
    ]
  }
  cumsum_cols <- grep("^cumsum_", names(tr_gs), value = T)
  cummean_cols <- grep("^cummean_", names(tr_gs), value = T)
  
  cat("Mem used:\n"); print(pryr::mem_used())
  # Calculating rolling stats
  log_info("{type} dataset - calculating rolling stats...")
  for (n in rolling_n_sessions) {
    tr_gs[,
      (glue("time_since_prev{n}_session_start")) := frollsum(time_since_prev1_session_start, n, na.rm = T),
      by = "installation_id"
    ]

    for (calc in c("sum", "mean")) {
      rollfun <- match.fun(glue("froll{calc}"))
      tr_gs[,
        (glue("roll{calc}{n}_{count_cols}")) := lapply(.SD, rollfun, n = n, na.rm = T),
        by = "installation_id",
        .SDcols = count_cols
      ]
    }
  }
  
  # Drop unneccessary count columns
  to_drop <- c(
    "total_game_time",
    "n_distinct_events",
    "n_distinct_custom_events",
    "total_events",
    "all_rounds_beat",
    "correct_more",
    "dropped_accurately_more",
    "clicked_irrelevant_more",
    "avg_time_rounds_all_rounds_beat",
    "tutorial_time_taken",
    grep("^event_code_", count_cols, value = T),
    grep("^custom_event_code_", count_cols, value = T),
    grep("^tsgss_", count_cols, value = T)
  )
  tr_gs[, (to_drop) := NULL]
  
  # All cumulative and rolling features need to be lagged by 1 because the current session is included in the calculation
  to_lag <- c(
    grep("^cumsum_", names(tr_gs), value = T),
    grep("^cummean_", names(tr_gs), value = T),
    grep("^rollsum", names(tr_gs), value = T),
    grep("^rollmean", names(tr_gs), value = T)
  )
  tr_gs[,
    (glue("prev1_{to_lag}")) := lapply(.SD, shift, n = 1, type = "lag"),
    by = "installation_id",
    .SDcols = to_lag
  ]
  tr_gs[, (to_lag) := NULL]

  cat("Mem used:\n"); print(pryr::mem_used())
  # Ratios, normalised metrics
  log_info("{type} dataset - calculating ratios...")
  tr_gs[, `:=`(
    prev1_sessions_per_time = (session_number - 1) / time_since_first_session_start,
    prev1_game_time_per_session = prev1_cumsum_total_game_time / (session_number - 1),
    cum_game_time_proportion = prev1_cumsum_total_game_time / time_since_first_session_start
  )]
  for (n in rolling_n_sessions) {
    total_game_time <- tr_gs[[glue("prev1_rollsum{n}_total_game_time")]]
    total_time <- tr_gs[[glue("time_since_prev{n}_session_start")]]
    set(tr_gs, j = glue("prev{n}_game_time_proportion"), value = total_game_time / total_time)
  }
  # Normalise by:
  # 1. the "age" of the user
  # 2. the total game time
  ## Cumulative ratios
  cum_time_vars <- c("prev1_cumsum_total_game_time")
  
  cum_measure_vars <- grep("^prev1_cumsum_", names(tr_gs), value = T)
  for (time_var in cum_time_vars) {
    for (measure_var in cum_measure_vars) {
      if (measure_var %in% cum_time_vars) next()
      set(tr_gs, j = glue("ratio_{measure_var}_div_{time_var}"), value = tr_gs[[measure_var]] / tr_gs[[time_var]])
    }
  }

  ## Rolling ratios
  for (n in rolling_n_sessions) {
    roll_time_vars <- glue("prev1_rollsum{n}_total_game_time")
    
    roll_measure_vars <- grep(glue("^prev1_rollsum{n}_"), names(tr_gs), value = T)
    for (time_var in roll_time_vars) {
      for (measure_var in roll_measure_vars) {
        if (measure_var %in% roll_time_vars) next()
        set(tr_gs, j = glue("ratio_{measure_var}_div_{time_var}"), value = tr_gs[[measure_var]] / tr_gs[[time_var]])
      }
    }
  }

  # Extract assessment records only from training data & join the labels
  tr_ass <- merge(
    tr_gs,
    assessed[, .(game_session, accuracy_group)],
    by = "game_session"
  )
  
  rm(tr_gs); gc()
  cat("Mem used:\n"); print(pryr::mem_used())
  # Drop other non-modelled columns
  log_info("{type} dataset - cleaning up features...")
  final_drop <- c(
    "start_time",
    "type",
    "world",
    "title",
    "avg_time_between_events",
    "sd_time_between_events",
    "prev1_start_time",
    grep("^type_", names(tr_ass), value = T),
    # Drop raw cumsum and rollsum cols, ratios seem to be more effective
    grep("^prev1_cumsum_(event_code|custom_event_code|title|type|world)_", names(tr_ass), value = T),
    grep("^prev1_rollsum[0-9]{1}_(event_code|custom_event_code|title|type|world)_", names(tr_ass), value = T)
  )
  
  tr_ass[, c(final_drop) := NULL]

  tr_ass
}

sample_per_installation <- function(data, seed = 2806) {
  # Ensure one installation id per row only in training
  if (!("installation_id" %in% names(data))) stop("installation_id column required")
  data <- copy(data)
  
  set.seed(seed)
  data[, random_nums := runif(nrow(data))]
  setorderv(data, c("installation_id", "random_nums"))
  data[, rn_by_installation := seq_len(.N), by = "installation_id"]
  data <- data[rn_by_installation == 1]
  data[, c("random_nums", "rn_by_installation") := NULL]
  
  data[]
}

prepare_model_dt <- function(data, include_target = T, target_as_factor = F, na_value = -999) {
  .data <- copy(data[, setdiff(names(data), c("game_session", "installation_id")), with = F])
  for (col in names(.data)) {
    column_values <- .data[[col]]
    missing_rows <- which(
      is.nan(column_values) |
      is.infinite(column_values) |
      is.na(column_values)
    )
    set(.data, missing_rows, col, na_value)
  }
  names(.data) <- make.names(names(.data))
  
  if (target_as_factor) {
    .data[, accuracy_group := factor(accuracy_group, levels = c(0, 1, 2, 3))]
  }
  
  if (!include_target) .data[, accuracy_group := NULL]
  
  .data[]
}

round_regr_predictions <- function(predictions, min_integer, max_integer) {
  pmin(pmax(round(predictions, 0), min_integer), max_integer)
}

f_wkappa <- function(task, model, pred, feats, extra.args, target_levels = c(0, 1, 2, 3)) {
  # get confusion matrix
  truth <- factor(pred[["data"]][["truth"]], target_levels)
  response_raw <- pred[["data"]][["response"]] 
  if (is.numeric(response_raw)) {
    response <- round_regr_predictions(response_raw, min(target_levels), max(target_levels)) %>% 
      factor(target_levels)
  } else {
    response <- factor(response_raw, target_levels)
  }
  
  conf.mat <- table(truth, response)
  conf.mat <- conf.mat / sum(conf.mat)
  
  # get expected probs under independence
  rowsum <- rowSums(conf.mat)
  colsum <- colSums(conf.mat)
  expected.mat <- rowsum %*% t(colsum)
  
  # get weights
  class.values <- seq_along(levels(truth)) - 1L
  weights <- outer(class.values, class.values, FUN = function(x, y) (x - y)^2)
  
  # calculate weighted kappa
  1 - sum(weights * conf.mat) / sum(weights * expected.mat)
}

wkappa_measure <- function() {
  makeMeasure(
    id = "wkappa",
    minimize = F,
    properties = c("regr", "classif", "classif.multi"),
    fun = f_wkappa
  )
}

rf_model_specs <- function(train_data) {
  specs <- list()
  specs[["data"]] <- prepare_model_dt(train_data, target_as_factor = T, na_value = -999)
  setDF(specs[["data"]])
  specs[["task"]] <- makeClassifTask(
    data = specs[["data"]], 
    target = "accuracy_group"
  )
  specs[["learner"]] <-  makeLearner(
    cl = "classif.ranger",
    predict.type = "prob",
    num.trees = 2000,
    num.threads = 16,
    importance = "impurity"
  )
  specs[["measure"]] <- wkappa_measure()
  
  specs
}

get_rf_predictions_train <- function(train_data) {
  log_info("Generating RandomForest predictions for train dataset...")
  
  specs <- rf_model_specs(train_data)
  rdesc <- makeResampleDesc("CV", iters = 5)
  resample_results <- resample(specs[["learner"]], specs[["task"]], rdesc, specs[["measure"]], keep.pred = T)
  predictions_table <- resample_results[["pred"]][["data"]]
  setDT(predictions_table)
  # Reorder resample results to align with original data frame
  setorderv(predictions_table, "id")  
  prob_cols <- grep("prob", names(predictions_table), value = T)
  result <- predictions_table[, prob_cols, with = F]
  names(result) <- paste0("rf_", names(result))
  
  result
}

train_rf <- function(train_data) {
  log_info("Training full RF model...")
  specs <- rf_model_specs(train_data)
  
  train(specs[["learner"]], specs[["task"]])
}

get_rf_predictions_test <- function(rf_model, test_data) {
  log_info("Generating RF predictions for test dataset...")
  test_dt <- prepare_model_dt(test_data, include_target = F, na_value = -999)
  setDF(test_dt)
  predictions_table <- predict(rf_model, newdata = test_dt)[["data"]]
  setDT(predictions_table)
  prob_cols <- grep("prob", names(predictions_table), value = T)
  result <- predictions_table[, prob_cols, with = F]
  names(result) <- paste0("rf_", names(result))
  
  result
}

glmnet_model_specs <- function(train_data) {
  specs <- list()
  specs[["data"]] <- prepare_model_dt(train_data, target_as_factor = F, na_value  = 0)
  setDF(specs[["data"]])
  specs[["task"]] <- makeRegrTask(
    data = specs[["data"]], 
    target = "accuracy_group"
  )
  specs[["learner"]] <-  makeLearner(
    cl = "regr.cvglmnet",
    alpha = 1,
    nfolds = 3L
  )
  specs[["measure"]] <- wkappa_measure()
  
  specs
}

get_glmnet_predictions_train <- function(train_data) {
  log_info("Generating glmnet predictions for train dataset...")
  
  specs <- glmnet_model_specs(train_data)
  rdesc <- makeResampleDesc("CV", iters = 3)
  resample_results <- resample(specs[["learner"]], specs[["task"]], rdesc, specs[["measure"]], keep.pred = T)
  predictions_table <- resample_results[["pred"]][["data"]]
  setDT(predictions_table)
  # Reorder resample results to align with original data frame
  setorderv(predictions_table, "id")  
  result <- predictions_table[, .(response)]
  names(result) <- paste0("glmnet_", names(result))
  
  result
}

train_glmnet <- function(train_data) {
  log_info("Training full glmnet model...")
  specs <- glmnet_model_specs(train_data)
  
  train(specs[["learner"]], specs[["task"]])
}

get_glmnet_predictions_test <- function(glmnet_model, test_data) {
  log_info("Generating glmnet predictions for test dataset...")
  test_dt <- prepare_model_dt(test_data, include_target = F, na_value = 0)
  setDF(test_dt)
  predictions_table <- predict(glmnet_model, newdata = test_dt)[["data"]]
  setDT(predictions_table)
  result <- predictions_table[, .(response)]
  names(result) <- paste0("glmnet_", names(result))
  
  result
}


train_model <- function(
  train_data, 
  rf_predictions,
  glmnet_predictions,
  params, 
  return_type = "train", 
  tune_param_set = NULL
) {
  # validation_type: train, validate, tune
  train_dt <- list(
    prepare_model_dt(train_data, include_target = T, target_as_factor = F, na_value = -999), 
    rf_predictions,
    glmnet_predictions
  ) %>% as.data.table()
  setDF(train_dt)
  
  regr_task <- makeRegrTask(
    data = train_dt,
    target = "accuracy_group"
  )
  regr_lrn <- makeLearner(
    cl = "regr.xgboost",
    nthread = 16,
    objective = "reg:squarederror",
    eval_metric = "rmse",
    subsample = 1,
    tree_method = "hist",
    eta = params[["eta"]],
    max_depth = params[["max_depth"]],
    min_child_weight = params[["min_child_weight"]],
    colsample_bytree = params[["colsample_bytree"]],
    nrounds = params[["nrounds"]]
  )
  regr_measure <- wkappa_measure()
  
  train_only <- function(regr_lrn, regr_task) {
    log_info("Training {regr_lrn[['id']]}...")
    
    train(regr_lrn, regr_task)
  }
  
  validate_fit <- function(regr_lrn, regr_task, regr_measure, save_models = T) {
    log_info("Validating {regr_lrn[['id']]}...")
    rdesc <- makeResampleDesc("CV", iters = 5)
    
    resample(regr_lrn, regr_task, rdesc, regr_measure, models = save_models)
  }
  
  tune_hyperparams <- function(regr_lrn, regr_task, regr_measure, param_set) {
    log_info("Tuning hyperparams for {regr_lrn[['id']]}...")
    
    rdesc <- makeResampleDesc("CV", iters = 5)
    ctrl <- makeTuneControlMBO(continue = T, mbo.control = makeMBOControl(save.on.disk.at = seq_len(30)))
    ctrl <- setMBOControlTermination(ctrl, max.evals = 50)
    
    tuneParams(
      learner = regr_lrn,
      task = regr_task,
      resampling = rdesc,
      measures = list(regr_measure),
      par.set = param_set,
      control = ctrl
    )
  } 
  
  switch(
    return_type,
    "train" = train_only(regr_lrn, regr_task),
    "validate" = validate_fit(regr_lrn, regr_task, regr_measure),
    "tune" = tune_hyperparams(regr_lrn, regr_task, regr_measure, tune_param_set)
  )
}

predict_model <- function(mlr_model, test_data, rf_predictions, glmnet_predictions) {
  log_info("Predicting on test set...")
  test_dt <- list(
    prepare_model_dt(test_data, include_target = F, na_value = -999),
    rf_predictions,
    glmnet_predictions
  ) %>% as.data.table()
  setDF(test_dt)
  
  pred <- predict(mlr_model, newdata = test_dt)

  pred[["data"]][["response"]]
}

produce_submission <- function(
  sample_per_installation,
  rolling_n_sessions, 
  thresholds_hours, 
  xgboost_params, 
  filename
) {
  xy_train <- build_session_data("train", rolling_n_sessions, thresholds_hours)
  if (sample_per_installation) {
    xy_train <- sample_per_installation(xy_train)
  }
  rf_predictions_train <- get_rf_predictions_train(xy_train)
  glmnet_predictions_train <- get_glmnet_predictions_train(xy_train)
  model_full <- train_model(xy_train, rf_predictions_train, glmnet_predictions_train, xgboost_params, "train")
  rf_model <- train_rf(xy_train)
  glmnet_model <- train_glmnet(xy_train)
  rm(xy_train); gc()
  
  xy_test <- build_session_data("test", rolling_n_sessions, thresholds_hours)
  rf_predictions_test <- get_rf_predictions_test(rf_model, xy_test)
  glmnet_predictions_test <- get_glmnet_predictions_test(glmnet_model, xy_test)

  predictions_raw <- predict_model(model_full, xy_test, rf_predictions_test, glmnet_predictions_test)
  xy_test[, accuracy_group := round_regr_predictions(predictions_raw, 0, 3)]
  submission <- xy_test[, .(installation_id, accuracy_group)]
  fwrite(submission, filename)
}