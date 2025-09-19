require(dplyr)
require(tidyr)
require(ggplot2)
require(tibble)
require(purrr)

source("functions/math/math_formulas.R")
source("functions/utils/create_partitions.R")
source("functions/utils/class_props.R")

source("functions/sampling/sys_pc.R")
source("functions/sampling/sys_o2.R")

localized_sampling <- function(data,
  y_var,
  final_sample_size = NULL,
  final_sample_prop = NULL,
  n_partitions = 6,
  heterogeneous_prop = 0.8,
  return_idx = FALSE
) {


  data$partition <- create_partitions(data, n_partitions = n_partitions)

  res_wide <- class_props(data, wide = TRUE)

  df_entropy <- res_wide %>%
    rowwise() %>%
    mutate(entropy = class_entropy(c_across(-partition))) %>%
    ungroup() %>%
    select(partition, entropy)

  heterogeneous_partitions <- df_entropy %>%
    filter(entropy > 0) %>%
    pull(partition)
  homogeneous_partitions <- df_entropy %>%
    filter(entropy == 0) %>%
    pull(partition)

  if (is.null(final_sample_size)) {
    final_sample_size <- nrow(data) * final_sample_prop
  }

  heterogeneous_total_size <- final_sample_size * heterogeneous_prop
  homogeneous_total_size <- final_sample_size * (1 - heterogeneous_prop)

  heterogeneous_samples_prop <- min(heterogeneous_total_size /
    sum(data$partition %in% heterogeneous_partitions), 1)
  homogeneous_samples_prop <- min(homogeneous_total_size /
    sum(data$partition %in% homogeneous_partitions), 1)

  heterogeneous_samples <- map_dfr(heterogeneous_partitions, function(prt) {
    sys_o2(data %>% filter(partition == prt),
           y_var = "y",
           prop = heterogeneous_samples_prop)
  })
  homogeneous_samples <- map_dfr(homogeneous_partitions, function(prt) {
    sys_o2(data %>% filter(partition == prt),
           y_var = "y",
           prop = homogeneous_samples_prop)
  })

  final_sample <- bind_rows(heterogeneous_samples, homogeneous_samples)
  return(final_sample)
}