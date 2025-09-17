require(tidyverse)

dados <- arrow::read_parquet("datasets/twonormals.parquet")

dados %>% 
  ggplot(aes(x = x1, y = x2, colour = y)) +
  geom_point() +
  theme_minimal()

dados_sample <- dados %>% 
  localized_sampling(y='y', ppc_target_size = 100)

dados_sample %>% 
  ggplot(aes(x = x1, y = x2, colour = y)) +
  geom_point() +
  theme_minimal()
