gen_twonormals_data <- function(n = 1000, cl = 2, sd = 1, outName = "twonormals") {
  d <- mlbench::mlbench.2dnormals(n = n, cl = cl, sd = sd)
  d <- as.data.frame(d)
  colnames(d) <- c("x1", "x2", "y")
  d$y <- as.factor(d$y)
  arrow::write_parquet(d, paste0("datasets/", outName, ".parquet"))
  print(paste("Data written to", paste0("datasets/", outName, ".parquet")))
}