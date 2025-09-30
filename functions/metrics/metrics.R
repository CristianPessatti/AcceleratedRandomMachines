#-------------------------------------------------------------------------------
# accuracy: simple exact-match accuracy for factor/character labels
#-------------------------------------------------------------------------------
accuracy <- function(truth, pred) {
  mean(as.character(truth) == as.character(pred))
}

#-------------------------------------------------------------------------------
# f1_score: macro-averaged F1 for multi-class; binary reduces to standard F1
#-------------------------------------------------------------------------------
f1_score <- function(truth, pred) {
  truth <- as.factor(truth)
  pred  <- as.factor(pred)
  classes <- union(levels(truth), levels(pred))
  truth <- factor(truth, levels = classes)
  pred  <- factor(pred, levels = classes)
  f1_per_class <- sapply(classes, function(cls) {
    tp <- sum(truth == cls & pred == cls)
    fp <- sum(truth != cls & pred == cls)
    fn <- sum(truth == cls & pred != cls)
    precision <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
    recall    <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
    if ((precision + recall) == 0) 0 else 2 * precision * recall / (precision + recall)
  })
  mean(f1_per_class)
}

#-------------------------------------------------------------------------------
# mcc: Matthews Correlation Coefficient (binary). For multi-class, compute
# one-vs-rest MCC per class and macro-average.
#-------------------------------------------------------------------------------
mcc <- function(truth, pred) {
  truth <- as.factor(truth)
  pred  <- as.factor(pred)
  classes <- union(levels(truth), levels(pred))
  truth <- factor(truth, levels = classes)
  pred  <- factor(pred, levels = classes)
  mcc_per_class <- sapply(classes, function(cls) {
    tp <- sum(truth == cls & pred == cls)
    tn <- sum(truth != cls & pred != cls)
    fp <- sum(truth != cls & pred == cls)
    fn <- sum(truth == cls & pred != cls)
    denom <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if (denom == 0) 0 else (tp * tn - fp * fn) / denom
  })
  mean(mcc_per_class)
}