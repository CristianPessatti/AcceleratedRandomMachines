#-------------------------------------------------------------------------------
# time_ms: Evaluate an expression and return elapsed wall time in milliseconds
#
# Usage
#   time_ms({
#     heavy_result <- do_something()
#     heavy_result
#   })
#
# Returns a list
#   - value: result of the evaluated expression
#   - ms: elapsed time in milliseconds (numeric)
#-------------------------------------------------------------------------------
time_ms <- function(expr) {
  t <- system.time(val <- eval.parent(substitute(expr)))
  list(value = val, ms = unname(t["elapsed"]) * 1000)
}