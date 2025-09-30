d <- read.csv("C:\\Users\\criss\\Downloads\\data.csv")

head(d)

colnames(d) <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13","y")

d$y <- factor(d$y)

head(d)

arrow::write_parquet(d, "datasets/cure_the_princess.parquet")
