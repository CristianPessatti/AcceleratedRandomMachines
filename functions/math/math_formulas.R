class_entropy <- function(probs) {
  probs <- probs[probs > 0]
  -sum(probs * log2(probs))
}

scale_exponential <- function(x, alpha = 1) {
  exp(alpha * x) / sum(exp(alpha * x))
}