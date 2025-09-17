gen_linear_split_data <- function(n = 2000, sd = 0.5, outName = "linear_split") {
    d <- expand.grid(
        seq(0,10, by=0.01),
        seq(0,10, by=0.01)
    ) %>% 
    sample_n(n) %>% 
    rename(x1 = 'Var1', x2 = 'Var2') %>%
    mutate(y = ifelse(x1 > x2, 0, 1)) %>%
    mutate(y = factor(y)) %>%
    mutate(
    x1 = x1 + rnorm(n(), mean = 0, sd = sd),
    x2 = x2 + rnorm(n(), mean = 0, sd = sd)
    ) %>% 
    as_tibble()

    arrow::write_parquet(d, paste0("datasets/", outName, ".parquet"))
    print(paste("Data written to", paste0("datasets/", outName, ".parquet")))
}
