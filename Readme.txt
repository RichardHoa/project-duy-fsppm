ollama pull qwen2.5:7b
ollama pull nomic-embed-text
pip install ollama pandas openpyxl
pip install tqdm
pip install seaborn matplotlib scipy

cd "C:\Users\Duy Pham\Layer 2 - Persona Simulator"

cd "/Users/duypham/Documents/Layer 2 - Persona Simulator"


python src/step2_validation.py
python src/step2_validation.py --runs 5
python src/step3_validation.py 
python src/calibration_scientist.py
python src/step3_calibration.py 
python src/step4_vhlss.py 
python src/step4_Personas_IPFP.py 
python src/step4_Personas_IPFP_feedback.py 



MAE
Confusion matrix
Accuracy rate
MCC
Distribution gap

# ============================================================
# FINAL R SCRIPT (single-sheet Excel input, fixed column names)
# Metrics per factor:
#   - Confusion matrix
#   - Agreement rate (Accuracy)
#   - MCC (multi-class)
#   - Distribution gap (TVD)
#   - MAE on numeric scores (optional but recommended)
# Exports an Excel workbook with:
#   - Summary (Agreement, MCC, TVD, MAE)
#   - <FACTOR>_CM (confusion matrix)
#   - <FACTOR>_Dist (distribution + gaps)
# ============================================================

# install.packages(c("readxl","dplyr","stringr","tibble","writexl"))

library(readxl)
library(dplyr)
library(stringr)
library(tibble)
library(writexl)

# -----------------------------
# Settings
# -----------------------------
classes <- c("Oppose", "Neutral", "Support")
factors <- c("ENF1","ENF2","FAC1","FAC2","TRU1","TRU2","TRU3","TRU4","OUT1")

# Required columns per factor (you said you have aligned them):
#   AI_<FACTOR>_Category, Human_<FACTOR>_Category
#   AI_<FACTOR>_Score,    Human_<FACTOR>_Score   (for MAE)
#
# Example:
#   AI_ENF1_Category, Human_ENF1_Category
#   AI_ENF1_Score,    Human_ENF1_Score

# -----------------------------
# Helpers
# -----------------------------
normalize_cat <- function(x) {
  x <- as.character(x)
  x <- str_trim(x)
  str_to_title(x)
}

empty_cm <- function(classes) {
  m <- matrix(0, nrow = length(classes), ncol = length(classes),
              dimnames = list(classes, classes))
  as.table(m)
}

calc_multiclass_mcc_with_reason <- function(cm) {
  # cm: matrix rows=AI(pred), cols=Human(true)
  N <- sum(cm)
  if (N == 0) {
    return(list(mcc = NA_real_, reason = "N=0 after filtering (no valid Oppose/Neutral/Support rows)"))
  }

  c_trace <- sum(diag(cm))
  p_k <- rowSums(cm)
  t_k <- colSums(cm)

  num <- c_trace * N - sum(p_k * t_k)
  den_left  <- (N^2 - sum(p_k^2))
  den_right <- (N^2 - sum(t_k^2))
  den <- sqrt(den_left * den_right)

  if (den == 0) {
    pred_unique <- sum(p_k > 0)
    true_unique <- sum(t_k > 0)
    reason <- paste0(
      "Denominator=0 (degenerate). Pred classes used: ", pred_unique,
      ", True classes used: ", true_unique,
      ". Usually predictions or ground truth collapse to a single class."
    )
    return(list(mcc = NA_real_, reason = reason))
  }

  list(mcc = as.numeric(num / den), reason = "OK")
}

calc_distribution_gap_tvd <- function(ai_cat, human_cat, classes) {
  # TVD = 0.5 * sum_k |p_ai(k) - p_human(k)|
  if (length(ai_cat) == 0) {
    return(list(tvd = NA_real_, p_ai = rep(0, length(classes)),
                p_human = rep(0, length(classes)), gap = rep(0, length(classes))))
  }

  p_ai <- table(factor(ai_cat, levels = classes)) / length(ai_cat)
  p_hu <- table(factor(human_cat, levels = classes)) / length(human_cat)

  gap <- p_ai - p_hu
  tvd <- 0.5 * sum(abs(gap))

  list(
    tvd = as.numeric(tvd),
    p_ai = as.numeric(p_ai),
    p_human = as.numeric(p_hu),
    gap = as.numeric(gap)
  )
}

safe_numeric <- function(x) suppressWarnings(as.numeric(x))

calc_mae_with_reason <- function(df, ai_score_col, hu_score_col) {
  if (!(ai_score_col %in% names(df)) || !(hu_score_col %in% names(df))) {
    return(list(mae = NA_real_, reason = "Missing score columns"))
  }

  a <- safe_numeric(df[[ai_score_col]])
  h <- safe_numeric(df[[hu_score_col]])
  ok <- is.finite(a) & is.finite(h)

  if (sum(ok) == 0) {
    return(list(mae = NA_real_, reason = "No valid numeric score pairs"))
  }

  list(mae = mean(abs(a[ok] - h[ok])), reason = "OK")
}

run_one_factor <- function(df, fac, classes) {
  ai_cat_col <- paste0("AI_", fac, "_Category")
  hu_cat_col <- paste0("Human_", fac, "_Category")
  ai_score_col <- paste0("AI_", fac, "_Score")
  hu_score_col <- paste0("Human_", fac, "_Score")

  missing <- c()
  if (!(ai_cat_col %in% names(df))) missing <- c(missing, ai_cat_col)
  if (!(hu_cat_col %in% names(df))) missing <- c(missing, hu_cat_col)

  # If category columns missing, still return placeholders so exports never drop factors
  if (length(missing) > 0) {
    mae_obj <- calc_mae_with_reason(df, ai_score_col, hu_score_col)
    return(list(
      factor = fac,
      status = paste("MISSING CATEGORY COLUMNS:", paste(missing, collapse = ", ")),
      n_cat = 0,
      cm = empty_cm(classes),
      agreement = NA_real_,
      mcc = NA_real_,
      mcc_reason = "Missing category columns",
      tvd = NA_real_,
      mae = mae_obj$mae,
      mae_reason = mae_obj$reason,
      dist = list(p_ai = rep(0,3), p_human = rep(0,3), gap = rep(0,3))
    ))
  }

  dat <- df %>%
    transmute(
      AI = normalize_cat(.data[[ai_cat_col]]),
      Human = normalize_cat(.data[[hu_cat_col]])
    ) %>%
    filter(AI %in% classes, Human %in% classes)

  dat$AI <- factor(dat$AI, levels = classes)
  dat$Human <- factor(dat$Human, levels = classes)

  cm_raw <- table(dat$AI, dat$Human)

  # Force full 3x3
  cm_full <- empty_cm(classes)
  if (sum(cm_raw) > 0) {
    cm_full[rownames(cm_raw), colnames(cm_raw)] <- cm_raw
  }

  agreement <- if (sum(cm_full) == 0) NA_real_ else sum(diag(cm_full)) / sum(cm_full)
  mcc_obj <- calc_multiclass_mcc_with_reason(as.matrix(cm_full))
  dist_obj <- calc_distribution_gap_tvd(dat$AI, dat$Human, classes)

  mae_obj <- calc_mae_with_reason(df, ai_score_col, hu_score_col)

  list(
    factor = fac,
    status = "OK",
    n_cat = nrow(dat),
    cm = cm_full,
    agreement = agreement,
    mcc = mcc_obj$mcc,
    mcc_reason = mcc_obj$reason,
    tvd = dist_obj$tvd,
    mae = mae_obj$mae,
    mae_reason = mae_obj$reason,
    dist = list(
      p_ai = dist_obj$p_ai,
      p_human = dist_obj$p_human,
      gap = dist_obj$gap
    )
  )
}

# -----------------------------
# 1) Pick file, read data
# -----------------------------
file_path <- file.choose()
sheets <- excel_sheets(file_path)

# If your data is NOT in the first sheet, replace sheets[1] with the correct sheet name.
df <- read_excel(file_path, sheet = sheets[1])

# -----------------------------
# 2) Run all factors
# -----------------------------
results <- setNames(lapply(factors, function(fac) run_one_factor(df, fac, classes)), factors)

# -----------------------------
# 3) Summary table
# -----------------------------
summary_tbl <- tibble(
  Factor = factors,
  Status = sapply(results, \(x) x$status),
  N_CategoryRows = sapply(results, \(x) x$n_cat),
  Agreement = sapply(results, \(x) x$agreement),
  MCC = sapply(results, \(x) x$mcc),
  MCC_Reason = sapply(results, \(x) x$mcc_reason),
  DistributionGap_TVD = sapply(results, \(x) x$tvd),
  MAE_Score = sapply(results, \(x) x$mae),
  MAE_Reason = sapply(results, \(x) x$mae_reason)
)

print(summary_tbl)

# -----------------------------
# 4) Build output workbook sheets
# -----------------------------
sheets_out <- list()
sheets_out[["Summary"]] <- summary_tbl

for (fac in factors) {
  res <- results[[fac]]

  # Confusion matrix
  cm_df <- as.data.frame.matrix(res$cm) %>%
    rownames_to_column("AI_Pred") %>%
    as_tibble()
  sheets_out[[paste0(fac, "_CM")]] <- cm_df

  # Distribution details (AI vs Human, and gap)
  dist_df <- tibble(
    Class = classes,
    AI_Distribution = res$dist$p_ai,
    Human_Distribution = res$dist$p_human,
    Gap_AI_minus_Human = res$dist$gap
  )
  sheets_out[[paste0(fac, "_Dist")]] <- dist_df
}

# -----------------------------
# 5) Export to Excel
# -----------------------------
out_file <- paste0("metrics_output_", format(Sys.Date(), "%Y%m%d"), ".xlsx")
write_xlsx(sheets_out, path = out_file)

message("Exported: ", out_file)




