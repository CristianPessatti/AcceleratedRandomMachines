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

log_normalize <- function(x) {
  eps <- 1e-8
  x <- pmin(pmax(x, eps), 1 - eps)  # Clamp x to (0,1)
  l <- log(x / (1 - x))
  l_min <- min(l)
  if (l_min < 0) {
    l <- l - l_min  # shift so minimum is 0
  }
  total <- sum(l)
  # If all x are forced to eps or (1-eps), l might be all zeros; in this case assign uniform weights
  if (total == 0) {
    return(rep(1 / length(x), length(x)))
  } else {
    return(l / total)
  }
}
