#-------------------------------------------------------------------------------
# accuracy: simple exact-match accuracy for factor/character labels
#-------------------------------------------------------------------------------
accuracy <- function(truth, pred) {
  mean(as.character(truth) == as.character(pred))
}