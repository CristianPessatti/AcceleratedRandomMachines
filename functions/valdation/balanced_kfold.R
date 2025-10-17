stratified_kfold <- function(df, K = 5, y) {
  stopifnot(is.data.frame(df), K >= 2, y %in% names(df))
  n <- nrow(df)
  yv <- df[[y]]
  all_idx <- seq_len(n)

  # helper: embaralha e reparte um vetor de índices em K partes (aprox. iguais)
  split_into_k <- function(idx, K) {
    idx <- sample(idx)  # embaralha
    groups <- split(idx, rep(1:K, length.out = length(idx)))
    # garante listas vazias quando a classe tem menos obs. que K
    groups[as.character(1:K)] <- lapply(1:K, function(k) groups[[as.character(k)]] %||% integer(0))
    groups
  }

  # operador "ou" para lidar com NULL
  `%||%` <- function(a, b) if (is.null(a)) b else a

  # para cada classe, cria K partições
  class_splits <- lapply(split(all_idx, yv), split_into_k, K = K)

  # monta folds: em cada k, junta as partes k de todas as classes
  folds <- vector("list", K)
  for (k in seq_len(K)) {
    test_idx <- unlist(lapply(class_splits, function(g) g[[as.character(k)]]), use.names = FALSE)
    train_idx <- setdiff(all_idx, test_idx)
    folds[[k]] <- list(train = train_idx, test = test_idx)
  }

  folds
}
