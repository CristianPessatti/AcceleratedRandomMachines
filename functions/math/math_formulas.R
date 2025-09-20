#-------------------------------------------------------------------------------
# class_entropy: Shannon entropy for a vector of class probabilities
# scale_exponential: softmax-like exponential scaling with temperature alpha
#-------------------------------------------------------------------------------
class_entropy <- function(probs) {
  probs <- probs[probs > 0]
  -sum(probs * log2(probs))
}

scale_exponential <- function(x, alpha = 1) {
  exp(alpha * x) / sum(exp(alpha * x))
}