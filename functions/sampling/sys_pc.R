# Systematic Sampling from Partition Centroid (SyS-PC) for a single partition
# data: tibble/data.frame already filtered to one partition
# y_var: response column to exclude from features (if NULL, use all numeric)
# n_total: desired sample size from the partition (optional)
# prop: sampling proportion (0<prop<=1) if n_total not provided (optional)
# scale_vars: scale variables before PCA
# seed: optional random seed
# return_idx: if TRUE return only row indices; otherwise return sampled rows

sys_pc <- function(data,
                   y_var = NULL,
                   n_total = NULL,
                   prop = NULL,
                   scale_vars = TRUE,
                   seed = NULL,
                   return_idx = FALSE) {
  stopifnot(is.data.frame(data))
  n <- nrow(data)
  if (n == 0) return(data[0, , drop = FALSE])
  if (!is.null(seed)) set.seed(seed)

  # define features (exclude y_var if specified)
  numeric_cols <- names(data)[vapply(data, is.numeric, logical(1))]
  if (is.null(y_var)) {
    features <- numeric_cols
  } else {
    features <- setdiff(numeric_cols, y_var)
  }
  if (length(features) == 0) stop("Sem features numéricas.")
  X <- data[, features, drop = FALSE]

  # define n_total
  if (is.null(n_total)) {
    if (is.null(prop)) stop("Informe n_total ou prop.")
    stopifnot(prop > 0, prop <= 1)
    n_total <- max(1L, round(n * prop))
  }
  n_total <- min(n_total, n)

  # padronização por partição (opcional)
  if (scale_vars) {
    X <- scale(X)
    X <- as.matrix(X)
  } else {
    X <- as.matrix(X)
  }

  # casos de 1D, 2D+ (PCA->2D)
  p <- ncol(X)
  if (p == 1) {
    # distância assinada ao centróide (média ~ 0 se scale=TRUE)
    scores1 <- as.numeric(X[, 1])
    ord <- order(scores1)
    angle <- atan2(0, scores1)  # apenas para manter coluna .angle
    angle[is.na(angle)] <- 0
    S <- cbind(pc1 = scores1, pc2 = rep(0, n))
  } else {
    # PCA para 2D
    pr <- tryCatch(prcomp(X, center = FALSE, scale. = FALSE), error = function(e) NULL)
    if (is.null(pr)) stop("Falha no PCA.")
    scores <- pr$x
    # garante 2 colunas
    if (ncol(scores) == 1) scores <- cbind(scores, rep(0, n))
    S <- scores[, 1:2, drop = FALSE]
    angle <- atan2(S[, 2], S[, 1])  # ângulo em torno do centróide (0 após centralizar)
    ord <- order(angle)
  }

  # ordena por ângulo e faz amostragem sistemática
  idx_ord <- seq_len(n)[ord]
  step <- n / n_total
  start <- runif(1, 0, step)
  pick_pos <- floor(start + (0:(n_total - 1)) * step) + 1
  pick_pos[pick_pos > n] <- n  # segurança
  idx_sys <- unique(idx_ord[pick_pos])

  # se duplicou por arredondamento, completa com vizinhos não usados
  if (length(idx_sys) < n_total) {
    faltam <- n_total - length(idx_sys)
    candidatos <- setdiff(idx_ord, idx_sys)
    if (faltam > 0 && length(candidatos) > 0) {
      idx_sys <- c(idx_sys, candidatos[seq_len(min(faltam, length(candidatos)))])
    }
  }

  # saída
  out_meta <- tibble::tibble(
    .row_id = seq_len(n),
    .pc1 = S[, 1],
    .pc2 = S[, 2],
    .angle = angle
  )

  if (return_idx) {
    return(sort(idx_sys))
  } else {
    res <- data[idx_sys, , drop = FALSE]
    res$.pc1 <- out_meta$.pc1[idx_sys]
    res$.pc2 <- out_meta$.pc2[idx_sys]
    res$.angle <- out_meta$.angle[idx_sys]
    res$.rank_sys <- match(idx_sys, idx_ord)
    return(res)
  }
}
