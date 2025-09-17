# Systematic Sampling Over Order (SyS-O2)

# data: data.frame/tibble
# y_var: string com o nome da coluna de classe (fator/char/int)
# n_total: tamanho amostral total desejado (opcional se usar `prop`)
# prop: proporção global (0<prop<=1) do total (opcional se usar `n_total`)
# order_by: nome de coluna para ordenar dentro de cada classe (opcional)
# decreasing: se TRUE, ordena decrescente por `order_by`
# within: "random" (default) ou "given" (usa a ordem atual das linhas)
# seed: semente opcional
# return_idx: se TRUE, retorna apenas índices das linhas amostradas

sys_o2 <- function(data,
                   y_var,
                   n_total = NULL,
                   prop = NULL,
                   order_by = NULL,
                   decreasing = FALSE,
                   within = c("random", "given"),
                   seed = NULL,
                   return_idx = FALSE) {

  stopifnot(is.data.frame(data))
  stopifnot(y_var %in% names(data))
  within <- match.arg(within)

  df <- tibble::as_tibble(data)
  if (!".row_id" %in% names(df)) {
    df$.row_id <- seq_len(nrow(df))
  }
  df$.class  <- df[[y_var]]

  if (!is.null(seed)) set.seed(seed)

  # Tamanhos por classe
  tab <- df |>
    dplyr::count(.class, name = "n_class") |>
    dplyr::arrange(.class)

  N <- sum(tab$n_class)

  # Decide tamanho amostral por classe
  if (!is.null(n_total)) {
    stopifnot(n_total > 0, n_total <= N)
    tab$m_raw <- n_total * tab$n_class / N
  } else {
    stopifnot(!is.null(prop), prop > 0, prop <= 1)
    tab$m_raw <- prop * tab$n_class
    n_total <- round(prop * N)
  }

  # Maiores restos + limites
  tab$m_class <- pmax(0L, pmin(tab$n_class, floor(tab$m_raw)))
  deficit <- n_total - sum(tab$m_class)
  if (deficit > 0) {
    residues <- tab$m_raw - tab$m_class
    add_idx <- order(residues, decreasing = TRUE)[seq_len(deficit)]
    tab$m_class[add_idx] <- pmin(tab$n_class[add_idx], tab$m_class[add_idx] + 1L)
  } else if (deficit < 0) {
    remove_idx <- order(tab$m_raw - tab$m_class, decreasing = FALSE)[seq_len(abs(deficit))]
    tab$m_class[remove_idx] <- pmax(0L, tab$m_class[remove_idx] - 1L)
  }

  # Junta a alocação ao df (UMA vez)
  df <- df |>
    dplyr::left_join(tab, by = dplyr::join_by(.class))

  # Sistemático por classe
  sys_one_class <- function(dfc, m) {
    n <- nrow(dfc)
    if (m <= 0 || n == 0) return(integer(0))
    if (m >= n) return(dfc$.row_id)

    if (!is.null(order_by)) {
      stopifnot(order_by %in% names(dfc))
      ord <- order(dfc[[order_by]], decreasing = decreasing, na.last = TRUE)
      dfc <- dfc[ord, , drop = FALSE]
    } else if (within == "random") {
      dfc <- dfc[sample.int(n), , drop = FALSE]
    } # within == "given": mantém ordem

    step <- n / m
    start <- stats::runif(1, min = 0, max = step)
    pos <- floor(start + (0:(m - 1)) * step) + 1L
    pos[pos > n] <- n
    picked <- unique(pos)

    if (length(picked) < m) {
      faltam <- m - length(picked)
      cand <- setdiff(seq_len(n), picked)
      if (faltam > 0 && length(cand) > 0) {
        picked <- c(picked, cand[seq_len(min(faltam, length(cand)))])
      }
    }

    dfc$.row_id[picked]
  }

  ids <- df |>
    dplyr::group_by(.class) |>
    dplyr::group_map(~ sys_one_class(.x, m = unique(.x$m_class)), .keep = TRUE) |>
    unlist(use.names = FALSE)

  if (return_idx) {
    return(sort(ids))
  }

  out <- df[df$.row_id %in% ids, , drop = FALSE]

  # Rank informativo (não recupera a ordem aleatória exata)
  rank_by_class <- df |>
    dplyr::group_by(.class) |>
    dplyr::group_modify(function(dfc, key) {
      n <- nrow(dfc)
      if (!is.null(order_by)) {
        ord <- order(dfc[[order_by]], decreasing = decreasing, na.last = TRUE)
        dfc <- dfc[ord, , drop = FALSE]
      }
      dfc$.rank_sys <- seq_len(n)
      dfc
    }) |>
    dplyr::ungroup() |>
    dplyr::select(.row_id, .class, .rank_sys)

  # ⚠️ FIX: NÃO fazer novo left_join(tab, ...) aqui
  out <- out |>
    dplyr::left_join(rank_by_class, by = dplyr::join_by(.row_id, .class)) |>
    dplyr::rename(.n_class = n_class, .m_class = m_class) |>
    dplyr::mutate(.step_class = .n_class / pmax(1L, .m_class)) |>
    dplyr::select(-dplyr::any_of("m_raw"))

  out
}