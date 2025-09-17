time_ms <- function(expr) {
  t <- system.time(val <- eval.parent(substitute(expr)))
  list(value = val, ms = unname(t["elapsed"]) * 1000)
}