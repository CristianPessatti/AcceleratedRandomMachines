class_entropy <- function(probs) {
  probs <- probs[probs > 0]
  return(-sum(probs * log2(probs)))
}