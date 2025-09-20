require(dplyr)
require(tibble)
require(kernlab)

source("functions/utils/time_ms.R")

# Fit a single model on the active set and collect metrics
# - active_df: data.frame with features and target y
# - valid_df: validation data.frame for evaluation
# - feature_names: vector of feature column names
# - kernel: ksvm kernel parameter
fit_model <- function(active_df, valid_df, feature_names, kernel) {
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
