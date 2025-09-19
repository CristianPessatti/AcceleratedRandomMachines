require(dplyr)
require(tibble)
require(kernlab)
require(rpart)

source("functions/sampling/localized_sampling.R")
source("functions/sampling/nearest_enemy_sampling.R")
source("functions/utils/time_ms.R")
source("functions/metrics/metrics.R")

# Active Learning routine
# - train_df: data.frame with features x1, x2, ... and target y
# - valid_df: same structure as train_df (used only for evaluation)
# - initial_n: initial labeled pool size (selected with localized_sampling)
# - kernel: ksvm kernel (e.g., "rbfdot", "vanilladot", "polydot", ...)
# - alpha: weighting parameter for nearest_enemy_sampling
# - heterogeneous_prop: pass-through to localized_sampling for initial pool
# - stopping_delta: min absolute improvement in mean accuracy to continue
# - stopping_patience: stop after this many consecutive small improvements
# - max_additions: optional cap on number of points to add
activeLearning <- function(
  train_df,
  valid_df,
  initial_n,
  kernel = "rbfdot",
  alpha = 1,
  heterogeneous_prop = 0.8,
  stopping_delta = 1e-3,
  stopping_patience = 5,
  max_additions = NULL,
  verbose = TRUE
) {
  stopifnot("y" %in% names(train_df), "y" %in% names(valid_df))
  train_df <- as.data.frame(train_df)
  valid_df <- as.data.frame(valid_df)
  train_df$y <- as.factor(train_df$y)
  valid_df$y <- as.factor(valid_df$y)

  feature_names <- setdiff(names(train_df), "y")

  # Preserve original row ids for tracking support vectors
  if (is.null(train_df$.row_id)) train_df$.row_id <- seq_len(nrow(train_df))

  # Initial active pool using localized_sampling
  initial_sample <- localized_sampling(
    data = train_df[, c(feature_names, "y", ".row_id"), drop = FALSE],
    y_var = "y",
    final_sample_size = initial_n,
    heterogeneous_prop = heterogeneous_prop
  )

  active_df <- initial_sample
  remaining_df <- dplyr::anti_join(
    train_df[, c(feature_names, "y", ".row_id"), drop = FALSE],
    active_df[, c(".row_id"), drop = FALSE],
    by = ".row_id"
  )

  n_val <- nrow(valid_df)

  # Storage structures
  history <- tibble::tibble(
    iteration = integer(0),
    active_size = integer(0),
    mean_accuracy = numeric(0),
    sd_accuracy = numeric(0),
    mean_train_time_ms = numeric(0),
    plateau_max_pred = numeric(0)
  )
  val_obs_metrics <- tibble::tibble(
    iteration = integer(0),
    val_row_id = integer(0),
    correct_count = integer(0),
    total_models = integer(0),
    mean_correct = numeric(0)
  )
  sv_counts <- tibble::tibble(
    iteration = integer(0),
    train_row_id = integer(0),
    sv_count = integer(0)
  )

  acc_history <- numeric(0)
  above_plateau_streak <- 0L

  # Helper: fit a single model on the whole active set and collect metrics
  fit_model <- function(active_df) {
    # Ensure only features + y for modeling
    active_model_df <- active_df[, c(feature_names, "y", ".row_id"), drop = FALSE]
    valid_model_df  <- valid_df[,  c(feature_names, "y"), drop = FALSE]

    # Train
    fit_b <- tryCatch({
      time_ms({
        result <- NULL
        capture.output({
          result <- kernlab::ksvm(
            y ~ .,
            data   = active_model_df[, c(feature_names, "y"), drop = FALSE],
            kernel = kernel,
            scaled = TRUE
          )
        })
        result  # <- devolve o modelo, não o texto
      })
    }, error = function(e) list(value = NULL, ms = NA_real_))

    # Predict on validation and compute accuracy
    correct_counts <- integer(nrow(valid_model_df))
    if (!is.null(fit_b$value)) {
      pred <- tryCatch({ predict(fit_b$value, newdata = valid_model_df) }, error = function(e) NULL)
      if (!is.null(pred)) {
        correct_counts <- as.integer(as.character(pred) == as.character(valid_model_df$y))
      }
    }

    # Support vectors in this iteration (relative to active_model_df)
    sv_counter <- integer(nrow(active_model_df))
    if (!is.null(fit_b$value)) {
      sv_idx_list <- tryCatch({ kernlab::alphaindex(fit_b$value) }, error = function(e) NULL)
      if (!is.null(sv_idx_list)) {
        sv_idx <- unique(unlist(sv_idx_list))
        if (length(sv_idx) > 0) {
          tab <- table(sv_idx)
          sv_counter[as.integer(names(tab))] <- sv_counter[as.integer(names(tab))] + 1L
        }
      }
    }

    mean_acc <- mean(correct_counts)
    sd_acc   <- NA_real_
    mean_time <- fit_b$ms

    list(
      mean_accuracy = mean_acc,
      sd_accuracy = sd_acc,
      mean_train_time_ms = mean_time,
      per_val = tibble::tibble(
        val_row_id = seq_len(nrow(valid_model_df)),
        correct_count = correct_counts,
        total_models = 1L,
        mean_correct = correct_counts
      ),
      per_sv = tibble::tibble(
        train_row_id = active_model_df$.row_id,
        sv_count = sv_counter
      )
    )
  }

  # Evaluate initial pool
  iter <- 0L
  res0 <- fit_model(active_df)
  # Plateau for initial iteration (tree over a single point -> fallback to observed max)
  acc_df0 <- tibble::tibble(iteration = iter, mean_accuracy = res0$mean_accuracy)
  tree_model0 <- tryCatch({
    rpart::rpart(
      mean_accuracy ~ iteration,
      data = acc_df0,
      control = rpart::rpart.control(maxdepth = 3)
    )
  }, error = function(e) NULL)
  plateau0 <- tryCatch({
    if (is.null(tree_model0)) {
      max(acc_df0$mean_accuracy, na.rm = TRUE)
    } else {
      preds0 <- predict(tree_model0, acc_df0)
      max(as.numeric(preds0), na.rm = TRUE)
    }
  }, error = function(e) max(acc_df0$mean_accuracy, na.rm = TRUE))

  history <- dplyr::bind_rows(history, tibble::tibble(
    iteration = iter,
    active_size = nrow(active_df),
    mean_accuracy = res0$mean_accuracy,
    sd_accuracy = res0$sd_accuracy,
    mean_train_time_ms = res0$mean_train_time_ms,
    plateau_max_pred = plateau0
  ))
  val_obs_metrics <- dplyr::bind_rows(val_obs_metrics, dplyr::mutate(res0$per_val, iteration = iter, .before = 1))
  sv_counts <- dplyr::bind_rows(sv_counts, dplyr::mutate(res0$per_sv, iteration = iter, .before = 1))
  acc_history <- c(acc_history, res0$mean_accuracy)

  # Console log for initial iteration
  if (verbose) {
    cat(sprintf("[ActiveLearning] Iteration %d | accuracy=%.4f | plateau=%.4f | decision=%s\n",
              iter, res0$mean_accuracy, plateau0, "continue"))
    flush.console()
  }

  if (is.null(max_additions)) max_additions <- nrow(remaining_df)

  no_improve_streak <- 0L

  # Iterative additions
  for (step in seq_len(max_additions)) {
    if (nrow(remaining_df) == 0) break

    # Pick one candidate from the remaining pool using nearest-enemy sampling
    cand <- nearest_enemy_sampling(
      data = remaining_df[, c(feature_names, "y", ".row_id"), drop = FALSE],
      y_var = "y",
      final_sample_size = 1,
      alpha = alpha
    )

    # Ensure not duplicated
    cand <- cand[1, , drop = FALSE]
    active_df <- dplyr::bind_rows(active_df, cand)
    remaining_df <- dplyr::anti_join(remaining_df, cand[, c(".row_id"), drop = FALSE], by = ".row_id")

    iter <- iter + 1L
    res_iter <- fit_model(active_df)

    # Build acc_df including the current iteration to compute plateau
    acc_df <- dplyr::bind_rows(
      history %>% dplyr::select(iteration, mean_accuracy) %>% dplyr::arrange(iteration),
      tibble::tibble(iteration = iter, mean_accuracy = res_iter$mean_accuracy)
    ) %>% dplyr::arrange(iteration)

    tree_model <- tryCatch({
      rpart::rpart(
        mean_accuracy ~ iteration,
        data = acc_df,
        control = rpart::rpart.control(maxdepth = 3)
      )
    }, error = function(e) NULL)

    plateau <- tryCatch({
      if (is.null(tree_model)) {
        max(acc_df$mean_accuracy, na.rm = TRUE)
      } else {
        preds <- predict(tree_model, acc_df)
        max(as.numeric(preds), na.rm = TRUE)
      }
    }, error = function(e) max(acc_df$mean_accuracy, na.rm = TRUE))

    history <- dplyr::bind_rows(history, tibble::tibble(
      iteration = iter,
      active_size = nrow(active_df),
      mean_accuracy = res_iter$mean_accuracy,
      sd_accuracy = res_iter$sd_accuracy,
      mean_train_time_ms = res_iter$mean_train_time_ms,
      plateau_max_pred = plateau
    ))
    val_obs_metrics <- dplyr::bind_rows(val_obs_metrics, dplyr::mutate(res_iter$per_val, iteration = iter, .before = 1))
    sv_counts <- dplyr::bind_rows(sv_counts, dplyr::mutate(res_iter$per_sv, iteration = iter, .before = 1))

    # Stopping rule: regression tree plateau (max predicted accuracy)
    acc_history <- c(acc_history, res_iter$mean_accuracy)
    if (!is.na(res_iter$mean_accuracy) && res_iter$mean_accuracy > plateau) {
      above_plateau_streak <- above_plateau_streak + 1L
    } else {
      above_plateau_streak <- 0L
    }

    # Console log for this iteration and decision
    will_stop <- above_plateau_streak >= stopping_patience
    decision_msg <- if (will_stop) {
      "stop"
    } else {
      if (above_plateau_streak == 0) {
        "continue"
      } else {
        "continue(*)"
      }
    }
    if (verbose) {
      cat(sprintf("[ActiveLearning] Iteration %d | accuracy=%.4f | plateau=%.4f | decision=%s\n",
                  iter, res_iter$mean_accuracy, plateau, decision_msg))
      flush.console()
    }

    if (will_stop) break
  }

  list(
    final_active = active_df,
    history = history,
    validation_metrics = val_obs_metrics,
    support_vector_counts = sv_counts
  )
}