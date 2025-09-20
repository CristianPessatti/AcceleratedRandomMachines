## -----------------------------------------------------------------------------
## stratified_holdout
##
## Split indices into train/test using class-stratified sampling. Ensures each
## class contributes roughly train_prop of its examples to the training split.
## Returns a list with integer vectors 'train' and 'test'.
## -----------------------------------------------------------------------------
stratified_holdout <- function(y, train_prop, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  y <- as.factor(y)
  idx_train <- unlist(lapply(split(seq_along(y), y), function(idx_cls) {
    n_cls <- length(idx_cls)
    n_tr  <- max(1L, floor(n_cls * train_prop))
    sample(idx_cls, n_tr)
  }))
  idx_test <- setdiff(seq_along(y), idx_train)
  list(train = sort(idx_train), test = sort(idx_test))
}