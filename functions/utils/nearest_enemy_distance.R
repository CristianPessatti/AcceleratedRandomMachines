nearest_enemy_distance <- function(df, y_var, scale_vars = TRUE) {
  stopifnot(is.data.frame(df))

  # Handle both string and symbol inputs
  if (is.character(y_var)) {
    y_name <- y_var
  } else {
    y_sym <- rlang::ensym(y_var)
    y_name <- rlang::as_string(y_sym)
  }
  stopifnot(y_name %in% names(df))

  yv <- df[[y_name]]

  # Use all columns except y_var as features
  feature_names <- setdiff(names(df), y_name)
  if (length(feature_names) == 0) stop("No features to compute distances on.")
  X <- as.matrix(df[, feature_names, drop = FALSE])
  if (scale_vars) X <- scale(X)

  n <- nrow(X)
  out_dist  <- rep(NA_real_, n)
  out_idx   <- rep(NA_integer_, n)
  out_class <- rep(NA_character_, n)

  classes <- unique(yv)

  for (cls in classes) {
    idx_c <- which(yv == cls)
    idx_o <- which(yv != cls)
    if (length(idx_o) == 0L || length(idx_c) == 0L) next
    A <- X[idx_c, , drop = FALSE]
    B <- X[idx_o, , drop = FALSE]
    aa <- rowSums(A^2)
    bb <- rowSums(B^2)
    D2 <- outer(aa, bb, "+") - 2 * A %*% t(B)
    D2[D2 < 0] <- 0
    min_j <- max.col(-D2, ties.method = "first")
    out_dist[idx_c]  <- sqrt(D2[cbind(seq_along(min_j), min_j)])
    out_idx[idx_c]   <- idx_o[min_j]
    out_class[idx_c] <- as.character(yv[out_idx[idx_c]])
  }

  return(out_dist)
}