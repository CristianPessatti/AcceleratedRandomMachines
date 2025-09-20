## -----------------------------------------------------------------------------
## Active Learning module: pooled nearest-enemy selection with per-class best
##
## Summary
## - Initializes an active set via localized sampling to capture local structure
## - Iteratively samples a candidate pool (nearest-enemy sampling)
## - For each class present in the pool, evaluates candidates by validation
##   accuracy and adds the best-performing candidate for that class
## - Tracks history (accuracy, timing), per-validation correctness, and
##   support-vector counts; uses a shallow regression tree to estimate a
##   plateau and stops after a patience window
## - Console logs include the number of observations added each iteration
## -----------------------------------------------------------------------------
require(dplyr)
require(tibble)
require(kernlab)
require(rpart)
require(ggplot2)
utils::globalVariables(c(
  "iteration", "mean_accuracy", "plateau_max_pred",
  "x", "y", "y_cls", ".row_id"
))

source("functions/sampling/localized_sampling.R")
source("functions/sampling/nearest_enemy_sampling.R")
source("functions/utils/time_ms.R")
source("functions/metrics/metrics.R")
source("activeLearning/fit_model.R")

# Active Learning routine
# - train_df: data.frame with features x1, x2, ... and target y
# - valid_df: same structure as train_df (used only for evaluation)
# - initial_n: initial labeled pool size (selected with localized_sampling)
# - kernel: ksvm kernel (e.g., "rbfdot", "vanilladot", "polydot", ...)
# - alpha: weighting parameter for nearest_enemy_sampling
# - pool_size: number of candidates to sample per round before choosing best per class
# - heterogeneous_prop: pass-through to localized_sampling for initial pool
# - stopping_delta: min absolute improvement in mean accuracy to continue
# - stopping_patience: stop after this many consecutive small improvements
# - max_additions: optional cap on number of points to add
# - animate: if TRUE, draw plots during iterations (convergence and scatter)
activeLearning <- function(
  train_df,
  valid_df,
  initial_n,
  kernel = "rbfdot",
  alpha = 1,
  pool_size = 10,
  heterogeneous_prop = 0.8,
  stopping_delta = 1e-3,
  stopping_patience = 5,
  max_additions = NULL,
  verbose = TRUE,
  animate = TRUE
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

  # Precompute 2D coordinates for scatter animation
  plot_coords <- NULL
  if (animate) {
    if (length(feature_names) == 2) {
      plot_coords <- tibble::tibble(
        .row_id = train_df$.row_id,
        x = train_df[[feature_names[1]]],
        y = train_df[[feature_names[2]]],
        y_cls = train_df$y
      )
    } else if (length(feature_names) > 2) {
      X <- as.matrix(scale(train_df[, feature_names, drop = FALSE]))
      pr <- tryCatch(stats::prcomp(X, center = FALSE, scale. = FALSE), error = function(e) NULL)
      if (!is.null(pr)) {
        S <- pr$x
        if (ncol(S) == 1) S <- cbind(S, rep(0, nrow(S)))
        plot_coords <- tibble::tibble(
          .row_id = train_df$.row_id,
          x = S[, 1],
          y = S[, 2],
          y_cls = train_df$y
        )
      } else {
        # Fallback to first two features if PCA fails
        f2 <- head(feature_names, 2)
        plot_coords <- tibble::tibble(
          .row_id = train_df$.row_id,
          x = train_df[[f2[1]]],
          y = train_df[[f2[2]]],
          y_cls = train_df$y
        )
      }
    }
  }


  # Evaluate initial pool
  iter <- 0L
  res0 <- fit_model(active_df, valid_df, feature_names, kernel)
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
    cat(sprintf("[ActiveLearning] Iteration %d | accuracy=%.4f | plateau=%.4f | added=%d | decision=%s\n",
              iter, res0$mean_accuracy, plateau0, 0L, "continue"))
    flush.console()
  }

  # Draw initial animation frames
  if (animate) {
    # Convergence plot
    p <- ggplot(history, aes(x = iteration, y = mean_accuracy)) +
      geom_line(color = "steelblue", linewidth = 1) +
      geom_point(color = "steelblue", size = 1.5) +
      geom_line(aes(x = iteration, y = plateau_max_pred), color = "red", linewidth = 1) +
      ylim(0, 1) +
      theme_minimal() +
      labs(title = "Active Learning Convergence",
           x = "Iteration",
           y = "Mean Accuracy (validation)")
    # Scatter highlighting active samples if coords available
    if (!is.null(plot_coords)) {
      active_ids <- active_df$.row_id
      p_sc <- ggplot() +
        geom_point(data = plot_coords, aes(x = x, y = y), color = "gray50", alpha = 0.08, size = 1) +
        geom_point(data = dplyr::filter(plot_coords, .row_id %in% active_ids),
                   aes(x = x, y = y, color = y_cls), alpha = 1, size = 1.5) +
        theme_minimal() +
        labs(title = "Active Sample Highlighting",
             x = "Dim 1",
             y = "Dim 2",
             color = "Class")
      grid::grid.newpage()
      grid::pushViewport(grid::viewport(layout = grid::grid.layout(1, 2)))
      print(p, vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 1))
      print(p_sc, vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 2))
    } else {
      print(p)
    }
    flush.console()
  }

  if (is.null(max_additions)) max_additions <- nrow(remaining_df)

  no_improve_streak <- 0L

  # Iterative additions (pooled sampling and best-per-class selection)
  total_added <- 0L
  while (nrow(remaining_df) > 0 && total_added < max_additions) {
    remaining_quota <- max_additions - total_added
    if (remaining_quota <= 0L) break

    # Sample a pool of candidates using nearest-enemy sampling
    pool_n <- min(pool_size, nrow(remaining_df))
    cand_pool <- nearest_enemy_sampling(
      data = remaining_df[, c(feature_names, "y", ".row_id"), drop = FALSE],
      y_var = "y",
      final_sample_size = pool_n,
      alpha = alpha
    )

    # For each class in the pool, choose the best candidate (or the only one) by validation accuracy
    classes_in_pool <- unique(as.character(cand_pool$y))
    best_rows <- list()
    for (cls in classes_in_pool) {
      cand_cls <- cand_pool[as.character(cand_pool$y) == cls, , drop = FALSE]
      if (nrow(cand_cls) == 1) {
        best_rows[[length(best_rows) + 1L]] <- cand_cls[1, , drop = FALSE]
      } else {
        # Evaluate each candidate by temporarily adding to active set
        acc_values <- numeric(nrow(cand_cls))
        for (i in seq_len(nrow(cand_cls))) {
          tmp_active <- dplyr::bind_rows(active_df, cand_cls[i, , drop = FALSE])
          res_tmp <- fit_model(tmp_active, valid_df, feature_names, kernel)
          acc_values[i] <- res_tmp$mean_accuracy
        }
        best_idx <- which.max(acc_values)
        best_rows[[length(best_rows) + 1L]] <- cand_cls[best_idx, , drop = FALSE]
      }
    }

    # Enforce remaining quota if fewer additions allowed than classes present
    add_df <- dplyr::bind_rows(best_rows)
    if (nrow(add_df) > remaining_quota) {
      add_df <- add_df[seq_len(remaining_quota), , drop = FALSE]
    }

    # Apply additions
    added_this_iter <- nrow(add_df)
    if (added_this_iter == 0L) break
    active_df <- dplyr::bind_rows(active_df, add_df)
    remaining_df <- dplyr::anti_join(remaining_df, add_df[, c(".row_id"), drop = FALSE], by = ".row_id")
    total_added <- total_added + added_this_iter

    iter <- iter + 1L
    res_iter <- fit_model(active_df, valid_df, feature_names, kernel)

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
      cat(sprintf("[ActiveLearning] Iteration %d | accuracy=%.4f | plateau=%.4f | added=%d | decision=%s\n",
                  iter, res_iter$mean_accuracy, plateau, added_this_iter, decision_msg))
      flush.console()
    }

    # Update animation frames
    if (animate) {
      p <- ggplot(history, aes(x = iteration, y = mean_accuracy)) +
        geom_line(color = "steelblue", linewidth = 1) +
        geom_point(color = "steelblue", size = 1.5) +
        geom_line(aes(x = iteration, y = plateau_max_pred), color = "red", linewidth = 1) +
        ylim(0, 1) +
        theme_minimal() +
        labs(title = "Active Learning Convergence",
             x = "Iteration",
             y = "Mean Accuracy (validation)")
      if (!is.null(plot_coords)) {
        active_ids <- active_df$.row_id
        just_added_ids <- add_df$.row_id
        just_added_coords <- dplyr::filter(plot_coords, .row_id %in% just_added_ids)
        p_sc <- ggplot() +
          geom_point(data = plot_coords, aes(x = x, y = y), color = "gray50", alpha = 0.08, size = 1) +
          geom_point(data = dplyr::filter(plot_coords, .row_id %in% active_ids),
                     aes(x = x, y = y, color = y_cls), alpha = 1, size = 1.5) +
          geom_point(data = just_added_coords,
                     aes(x = x, y = y, color = y_cls), alpha = 1, size = 6) +
          theme_minimal() +
          labs(title = "Active Sample Highlighting",
               x = "Dim 1",
               y = "Dim 2",
               color = "Class")
        grid::grid.newpage()
        grid::pushViewport(grid::viewport(layout = grid::grid.layout(1, 2)))
        print(p, vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 1))
        print(p_sc, vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 2))
      } else {
        print(p)
      }
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