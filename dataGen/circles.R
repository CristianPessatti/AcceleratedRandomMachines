gen_circles_data <- function(n = 1000, outName = "circles") {
  d <- mlbench::mlbench.circle(n = n)
  d <- as.data.frame(d)
  colnames(d) <- c("x1", "x2", "y")
  d$y <- as.factor(d$y)
  arrow::write_parquet(d, paste0("datasets/", outName, ".parquet"))
  print(paste("Data written to", paste0("datasets/", outName, ".parquet")))
}
