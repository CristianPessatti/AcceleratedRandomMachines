#-------------------------------------------------------------------------------
# run_benchmark
#
# Repeated holdout benchmarking comparing:
# - full model trained on all training data
# - localized sampling
# - nearest enemy sampling
#
# Returns a tibble with per-holdout accuracy and training time for each method.
#-------------------------------------------------------------------------------
run_benchmark <- function(
    df,
    n_holdouts,
    train_prop,
    sample_prop,
    seed,
    alpha = 1,
    heterogeneous_prop = 0.8
) {
  stopifnot("y" %in% names(df))
  df$y <- as.factor(df$y)
  feature_names <- setdiff(names(df), "y")

  purrr::map_dfr(seq_len(n_holdouts), function(h) {
    split <- stratified_holdout(df$y, train_prop = train_prop, seed = seed + h)
    train_df <- df[split$train, , drop = FALSE]
    test_df  <- df[split$test, , drop = FALSE]

    n_samp <- max(1L, round(nrow(train_df) * sample_prop))

    train_df_model <- train_df[, c(feature_names, "y"), drop = FALSE]
    test_df_model  <- test_df[,  c(feature_names, "y"), drop = FALSE]

    # Localized sampling and NE sampling on training set
    loc_train <- localized_sampling(
        train_df,
        y_var = "y",
        final_sample_size = n_samp,
        heterogeneous_prop = heterogeneous_prop
    )
    ne_train  <- nearest_enemy_sampling(
        train_df,
        y_var = "y",
        final_sample_size = n_samp,
        alpha = alpha)

    loc_train_model <- loc_train[, c(feature_names, "y"), drop = FALSE]
    ne_train_model  <- ne_train[,  c(feature_names, "y"), drop = FALSE]

    safe_fit <- function(train_df_model) {
      if (nrow(train_df_model) < 2 || length(unique(train_df_model$y)) < 2) {
        return(list(value = NULL, ms = NA_real_))
      }
      tryCatch({
        time_ms({ 
            kernlab::ksvm(y ~ .,
                data = train_df_model,
                kernel = "rbfdot",
                scaled = TRUE)
        })
      }, error = function(e) list(value = NULL, ms = NA_real_))
    }
    safe_acc <- function(fit, test_df_model) {
      if (is.null(fit$value)) return(NA_real_)
      pred <- tryCatch({
        predict(fit$value, newdata = test_df_model)
      }, error = function(e) NULL)
      if (is.null(pred)) return(NA_real_)
      accuracy(test_df_model$y, pred)
    }

    # Full training
    full_fit <- safe_fit(train_df_model)
    full_acc <- safe_acc(full_fit, test_df_model)

    # Localized sampling model
    loc_fit <- safe_fit(loc_train_model)
    loc_acc <- safe_acc(loc_fit, test_df_model)

    # Nearest enemy sampling model
    ne_fit <- safe_fit(ne_train_model)
    ne_acc <- safe_acc(ne_fit, test_df_model)

    tibble::tibble(
      holdout = h,
      model   = c("full", "localized", "nearest_enemy"),
      accuracy = c(full_acc, loc_acc, ne_acc),
      train_time_ms = c(full_fit$ms, loc_fit$ms, ne_fit$ms),
      train_n = c(
        nrow(train_df_model),
        nrow(loc_train_model),
        nrow(ne_train_model)
      )
    )
  })
}