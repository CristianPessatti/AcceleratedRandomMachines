sample_bootstrap <- function(data, n = 100) {
  n_rows <- nrow(data)
  map(1:n, function(i) {
    indices <- sample(1:n_rows, size = n_rows, replace = TRUE)
    list(train = indices, test = setdiff(1:n_rows, indices))
  })
}