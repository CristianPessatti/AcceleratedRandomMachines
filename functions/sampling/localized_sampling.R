## -----------------------------------------------------------------------------
## localized_sampling
##
## Purpose
## - Partition the feature space, estimate within-partition class heterogeneity,
##   and sample a final set that prioritizes heterogeneous partitions while also
##   covering homogeneous ones.
##
## Key steps
## - Create integer partitions over numeric features
## - Compute per-partition class proportions and entropy
## - Split the target sample size between heterogeneous and homogeneous
##   partitions, then draw samples from each using systematic samplers
## -----------------------------------------------------------------------------
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

  # Ensure at least one instance of every class appears in the final sample
  # Uses the provided y_var to determine classes and fills any missing class
  # by selecting one representative from the remaining pool.
  all_classes <- unique(as.character(data[[y_var]]))
  present_classes <- unique(as.character(final_sample[[y_var]]))
  missing_classes <- setdiff(all_classes, present_classes)

  if (length(missing_classes) > 0) {
    ensure_rows <- purrr::map_dfr(missing_classes, function(cls) {
      cands <- data[data[[y_var]] == cls, , drop = FALSE]
      by_cols <- intersect(names(cands), names(final_sample))
      # Prefer candidates not already selected
      if (length(by_cols) > 0 && nrow(final_sample) > 0) {
        cands2 <- dplyr::anti_join(cands, final_sample[, by_cols, drop = FALSE], by = by_cols)
      } else {
        cands2 <- cands
      }
      if (nrow(cands2) == 0) cands2 <- cands
      cands2[1, , drop = FALSE]
    })
    final_sample <- dplyr::bind_rows(final_sample, ensure_rows)
  }
  return(final_sample)
}