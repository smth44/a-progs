options(stringsAsFactors=FALSE)

script_dir_best <- function() {
  frames <- sys.frames()
  source_files <- unlist(lapply(frames, function(frame) {
    if (is.null(frame$ofile)) character(0) else as.character(frame$ofile)[1L]
  }), use.names=FALSE)
  source_files <- source_files[nzchar(source_files)]
  if (length(source_files) > 0L) {
    return(dirname(normalizePath(tail(source_files, 1L), winslash="/", mustWork=FALSE)))
  }
  file_arg <- grep("^--file=", commandArgs(trailingOnly=FALSE), value=TRUE)
  if (length(file_arg) > 0L) {
    script_path <- sub("^--file=", "", tail(file_arg, 1L))
    return(dirname(normalizePath(script_path, winslash="/", mustWork=FALSE)))
  }
  normalizePath(getwd(), winslash="/", mustWork=TRUE)
}

cli_arg_best <- function(name, default=NULL) {
  prefix <- paste0("--", name, "=")
  hits <- commandArgs(trailingOnly=TRUE)
  hits <- hits[startsWith(hits, prefix)]
  if (length(hits) == 0L) return(default)
  substring(tail(hits, 1L), nchar(prefix) + 1L)
}

env_cli_best <- function(env_name, cli_name, default) {
  value <- Sys.getenv(env_name, unset="")
  if (nzchar(value)) return(value)
  cli_arg_best(cli_name, default)
}

find_input_best <- function(path, pattern, label) {
  if (file.exists(path)) {
    return(normalizePath(path, winslash="/", mustWork=TRUE))
  }
  candidates <- list.files(
    SCRIPT_DIR_BEST, pattern=pattern, full.names=TRUE, ignore.case=TRUE
  )
  if (length(candidates) == 1L) {
    return(normalizePath(candidates, winslash="/", mustWork=TRUE))
  }
  stop(
    "Не найден ", label, ": ", path,
    ". Положите файл рядом со скриптом или передайте путь через --",
    tolower(label), "=PATH / ", toupper(label), "_FILE."
  )
}

SCRIPT_DIR_BEST <- script_dir_best()
TRAIN_FILE_BEST <- env_cli_best(
  "TRAIN_FILE",
  "train",
  file.path(SCRIPT_DIR_BEST, "train.csv")
)

TEST_FILE_BEST <- env_cli_best(
  "TEST_FILE",
  "test",
  file.path(SCRIPT_DIR_BEST, "test.csv")
)

OUTPUT_FILE_BEST <- env_cli_best(
  "OUTPUT_FILE",
  "output",
  file.path(
    SCRIPT_DIR_BEST,
    "submission_best_74564_76.csv"
  )
)
TRAIN_FILE_BEST <- find_input_best(
  TRAIN_FILE_BEST,
  "^train.*\\.csv$",
  "TRAIN"
)

TEST_FILE_BEST <- find_input_best(
  TEST_FILE_BEST,
  "^test.*\\.csv$",
  "TEST"
)

OUTPUT_FILE_BEST <- normalizePath(
  OUTPUT_FILE_BEST,
  winslash = "/",
  mustWork = FALSE
)
dir.create(dirname(OUTPUT_FILE_BEST), recursive=TRUE, showWarnings=FALSE)
VALIDATION_FILE_BEST <- file.path(SCRIPT_DIR_BEST, "validation_best_74564_76.csv")
required_packages_best <- c("tidyverse", "xgboost", "lightgbm")
missing_packages_best <- required_packages_best[!vapply(required_packages_best, requireNamespace, logical(1), quietly=TRUE)]
if (length(missing_packages_best) > 0L) stop("Не установлены пакеты: ", paste(missing_packages_best, collapse=", "))
cat("Script directory:", SCRIPT_DIR_BEST, "\n")
cat("Train:", normalizePath(TRAIN_FILE_BEST, mustWork=TRUE), "\n")
cat("Test:", normalizePath(TEST_FILE_BEST, mustWork=TRUE), "\n")
cat("Output:", OUTPUT_FILE_BEST, "\n")


library(tidyverse)
library(xgboost)
HAVE_LGB <- requireNamespace("lightgbm", quietly = TRUE)
library(lightgbm)



N_FOLDS <- 5
SEEDS <- c(42, 777)
MAX_ROUNDS <- 5000
EARLY_STOP <- 100
PRIOR_TE <- 200
SEED_CV <- 2026


params_xgb_log <- list(eta=0.01, max_depth=6, min_child_weight=33.5,
                       subsample=0.93, colsample_bytree=0.71, lambda=0.77, alpha=0.55,
                       gamma=2.22e-16, tree_method="hist")

params_xgb_mae <- list(eta=0.01, max_depth=10, min_child_weight=40,
                       subsample=0.71, colsample_bytree=0.89, lambda=8.40, alpha=3.68,
                       gamma=3.32, tree_method="hist")

params_xgb_logmae <- list(eta=0.01, max_depth=10, min_child_weight=34.8,
                          subsample=0.73, colsample_bytree=0.56, lambda=6.57, alpha=1.37,
                          gamma=2.24, tree_method="hist")

lgb_params <- list(objective="regression_l1", metric="l1", learning_rate=0.05,
                   num_leaves=127, min_data_in_leaf=80, feature_fraction=0.6,
                   bagging_fraction=0.9, bagging_freq=1, lambda_l2=0, verbosity=-1)

wmae <- function(y, p, w) sum(w * abs(y - p)) / sum(w)


header_row <- read.csv2(TRAIN_FILE_BEST, header=FALSE, nrows=1, stringsAsFactors=FALSE)
данные_обуч <- read.csv2(TRAIN_FILE_BEST, header=FALSE, skip=1, stringsAsFactors=FALSE,
                         na.strings=c("", "NA", "NaN", "None", "null", "NULL", "nan"))
names(данные_обуч) <- as.character(header_row[1, ])

header_row_test <- read.csv2(TEST_FILE_BEST, header=FALSE, nrows=1, stringsAsFactors=FALSE)
данные_тест <- read.csv2(TEST_FILE_BEST, header=FALSE, skip=1, stringsAsFactors=FALSE,
                         na.strings=c("", "NA", "NaN", "None", "null", "NULL", "nan"))
names(данные_тест) <- as.character(header_row_test[1, ])


cat_cols <- c("gender", "adminarea", "city_smart_name",
              "dp_ewb_last_employment_position", "dp_address_unique_regions", "addrref")

other_cols <- setdiff(names(данные_обуч), c("id", "dt", "target", "w", cat_cols))
for (col in other_cols) {
  if (is.character(данные_обуч[[col]])) {
    данные_обуч[[col]] <- suppressWarnings(as.numeric(gsub(",", ".", данные_обуч[[col]])))
    if (col %in% names(данные_тест))
      данные_тест[[col]] <- suppressWarnings(as.numeric(gsub(",", ".", данные_тест[[col]])))
  }
}

id_обуч <- данные_обуч$id
id_тест <- данные_тест$id
целевая <- as.numeric(данные_обуч$target)
веса <- as.numeric(данные_обуч$w)

признаки_обуч <- данные_обуч %>% select(-any_of(c("id", "dt", "target", "w")))
признаки_тест <- данные_тест %>% select(-any_of(c("id", "dt")))


valid <- !is.na(целевая) & !is.na(веса) & is.finite(целевая) & is.finite(веса)
if (any(!valid)) {
  cat("Удалено строк:", sum(!valid), "\n")
  признаки_обуч <- признаки_обуч[valid, ]
  целевая <- целевая[valid]
  веса <- веса[valid]
  id_обуч <- id_обуч[valid]
}

n_tr <- nrow(признаки_обуч)
y_log <- log1p(целевая)
cat("Baseline WMAE:", round(wmae(целевая, median(целевая), веса), 1), "\n")


num_cols <- names(признаки_обуч)[sapply(признаки_обуч, is.numeric)]
for (col in num_cols) {
  x <- признаки_обуч[[col]]
  if (sum(!is.na(x)) < 20) next
  qs <- quantile(x, c(0.001, 0.999), na.rm=TRUE)
  if (!all(is.finite(qs)) || qs[1] == qs[2]) next
  признаки_обуч[[col]] <- pmin(pmax(x, qs[1]), qs[2])
  if (col %in% names(признаки_тест))
    признаки_тест[[col]] <- pmin(pmax(признаки_тест[[col]], qs[1]), qs[2])
}


miss_perc <- colMeans(is.na(признаки_обуч[num_cols])) * 100
na_flags <- names(miss_perc[miss_perc > 20])

for (col in na_flags) {
  признаки_обуч[[paste0(col, "_isna")]] <- as.numeric(is.na(признаки_обуч[[col]]))
  if (col %in% names(признаки_тест))
    признаки_тест[[paste0(col, "_isna")]] <- as.numeric(is.na(признаки_тест[[col]]))
}

признаки_обуч$n_missing <- rowSums(is.na(признаки_обуч[num_cols]))
признаки_тест$n_missing <- rowSums(is.na(признаки_тест[intersect(num_cols, names(признаки_тест))]))
признаки_обуч$completeness <- rowMeans(!is.na(признаки_обуч[num_cols]))
признаки_тест$completeness <- rowMeans(!is.na(признаки_тест[intersect(num_cols, names(признаки_тест))]))


prefixes <- c("dp_", "hdb_", "turn_", "avg_", "bki_", "transaction", "by_category", "vert_")
for (p in prefixes) {
  cols <- grep(paste0("^", p), num_cols, value=TRUE)
  if (length(cols) >= 3) {
    nm <- paste0("nfill_", gsub("[^a-z]+$", "", p))
    признаки_обуч[[nm]] <- rowSums(!is.na(признаки_обуч[cols]))
    признаки_тест[[nm]] <- rowSums(!is.na(признаки_тест[intersect(cols, names(признаки_тест))]))
  }
}


add_position_flags <- function(df) {
  col <- "dp_ewb_last_employment_position"
  if (!col %in% names(df)) return(df)
  s <- tolower(as.character(df[[col]]))
  s[is.na(s)] <- ""
  pats <- c(pos_head="директор|руковод|начальник|заведующ|управляющ|президент",
            pos_manager="менеджер|администратор", pos_engineer="инженер|техник|программист|разработ|it",
            pos_spec="специалист|эксперт|аналитик|консультант", pos_driver="водител|машинист|курьер|экспедитор",
            pos_sales="продавец|кассир|торгов", pos_teacher="учител|преподават|воспитат|педагог",
            pos_med="врач|медицин|медсестр|фельдшер",
            pos_worker="рабоч|слесар|монтаж|сварщ|электрик|оператор|грузчик|уборщ|охран|повар",
            pos_finance="бухгалтер|экономист|финанс|юрист")
  for (nm in names(pats)) df[[nm]] <- as.numeric(grepl(pats[[nm]], s))
  df$position_nchar <- nchar(s)
  df
}

признаки_обуч <- add_position_flags(признаки_обуч)
признаки_тест <- add_position_flags(признаки_тест)


safe_div <- function(a, b) ifelse(is.na(a) | is.na(b) | b == 0, NA, a / b)
has <- function(...) all(c(...) %in% names(признаки_обуч))

if (has("hdb_outstand_sum", "hdb_bki_total_max_limit")) {
  признаки_обуч$credit_util <- safe_div(признаки_обуч$hdb_outstand_sum, признаки_обуч$hdb_bki_total_max_limit)
  признаки_тест$credit_util <- safe_div(признаки_тест$hdb_outstand_sum, признаки_тест$hdb_bki_total_max_limit)
}

if (has("avg_cur_db_turn", "turn_cur_db_avg_v2")) {
  признаки_обуч$trend_db <- safe_div(признаки_обуч$avg_cur_db_turn, признаки_обуч$turn_cur_db_avg_v2)
  признаки_тест$trend_db <- safe_div(признаки_тест$avg_cur_db_turn, признаки_тест$turn_cur_db_avg_v2)
}

if (has("turn_cur_db_sum_v2", "avg_6m_all")) {
  признаки_обуч$savings_rate <- safe_div(признаки_обуч$turn_cur_db_sum_v2 - признаки_обуч$avg_6m_all*12, признаки_обуч$turn_cur_db_sum_v2)
  признаки_тест$savings_rate <- safe_div(признаки_тест$turn_cur_db_sum_v2 - признаки_тест$avg_6m_all*12, признаки_тест$turn_cur_db_sum_v2)
}


get_income_proxy <- function(df, fallback) {
  proxy <- rep(NA_real_, nrow(df))
  step <- function(proxy, v, f=identity) {
    if (v %in% names(df)) ifelse(is.na(proxy) & !is.na(df[[v]]), f(df[[v]]), proxy) else proxy
  }
  proxy <- step(proxy, "salary_6to12m_avg")
  proxy <- step(proxy, "dp_payoutincomedata_payout_avg_6_month")
  proxy <- step(proxy, "dp_ils_paymentssum_avg_12m")
  proxy <- step(proxy, "turn_cur_db_sum_v2", function(x) x/12)
  proxy <- step(proxy, "avg_6m_all", function(x) x*1.5)
  ifelse(is.na(proxy), fallback, proxy)
}

proxy_fallback <- median(c(признаки_обуч$salary_6to12m_avg, признаки_обуч$incomeValue, признаки_обуч$dp_ils_paymentssum_avg_12m), na.rm=TRUE)
признаки_обуч$income_proxy <- get_income_proxy(признаки_обуч, proxy_fallback)
признаки_тест$income_proxy <- get_income_proxy(признаки_тест, proxy_fallback)

if (has("hdb_outstand_sum")) {
  признаки_обуч$debt_to_income <- safe_div(признаки_обуч$hdb_outstand_sum, признаки_обуч$income_proxy)
  признаки_тест$debt_to_income <- safe_div(признаки_тест$hdb_outstand_sum, признаки_тест$income_proxy)
}

if (has("hdb_bki_total_max_limit")) {
  признаки_обуч$limit_to_income <- safe_div(признаки_обуч$hdb_bki_total_max_limit, признаки_обуч$income_proxy)
  признаки_тест$limit_to_income <- safe_div(признаки_тест$hdb_bki_total_max_limit, признаки_тест$income_proxy)
}

if (has("per_capita_income_rur_amt")) {
  признаки_обуч$income_to_region <- safe_div(признаки_обуч$income_proxy, признаки_обуч$per_capita_income_rur_amt)
  признаки_тест$income_to_region <- safe_div(признаки_тест$income_proxy, признаки_тест$per_capita_income_rur_amt)
  if (has("turn_cur_cr_avg_act_v2")) {
    признаки_обуч$turn_to_region <- safe_div(признаки_обуч$turn_cur_cr_avg_act_v2, признаки_обуч$per_capita_income_rur_amt)
    признаки_тест$turn_to_region <- safe_div(признаки_тест$turn_cur_cr_avg_act_v2, признаки_тест$per_capita_income_rur_amt)
  }
}


q_inc <- quantile(признаки_обуч$income_proxy, c(0.2, 0.4, 0.6, 0.8), na.rm=TRUE)
seg_num <- function(x) as.numeric(cut(x, breaks=c(-Inf, q_inc, Inf), labels=FALSE))
признаки_обуч$segment_num <- seg_num(признаки_обуч$income_proxy)
признаки_тест$segment_num <- seg_num(признаки_тест$income_proxy)


set.seed(SEED_CV)
dec <- cut(y_log, breaks=unique(quantile(y_log, 0:10/10)), include.lowest=TRUE, labels=FALSE)
номер_фолда <- integer(n_tr)
for (d in unique(dec)) {
  ii <- which(dec == d)
  номер_фолда[ii] <- sample(rep(1:N_FOLDS, length.out=length(ii)))
}


add_cat_encodings <- function(train_df, test_df, cols, y, w, folds, prior) {
  global_m <- sum(w * y) / sum(w)
  n <- nrow(train_df)
  for (col in intersect(cols, names(train_df))) {
    tr_v <- as.character(train_df[[col]]); tr_v[is.na(tr_v)] <- "__NA__"
    te_v <- as.character(test_df[[col]]); te_v[is.na(te_v)] <- "__NA__"
    te_col <- rep(global_m, n)
    for (f in sort(unique(folds))) {
      out_i <- folds == f
      s <- tapply(w[!out_i] * y[!out_i], tr_v[!out_i], sum)
      sw <- tapply(w[!out_i], tr_v[!out_i], sum)
      enc <- (s + prior*global_m) / (sw + prior)
      v <- unname(enc[tr_v[out_i]]); v[is.na(v)] <- global_m
      te_col[out_i] <- v
    }
    s_f <- tapply(w*y, tr_v, sum); sw_f <- tapply(w, tr_v, sum)
    enc_f <- (s_f + prior*global_m) / (sw_f + prior)
    te_te <- unname(enc_f[te_v]); te_te[is.na(te_te)] <- global_m
    cnt <- table(tr_v)
    cnt_tr <- as.numeric(cnt[tr_v]); cnt_tr[is.na(cnt_tr)] <- 0
    cnt_te <- as.numeric(cnt[te_v]); cnt_te[is.na(cnt_te)] <- 0
    lv <- sort(unique(tr_v))
    train_df[[paste0("te_", col)]] <- te_col
    test_df[[paste0("te_", col)]] <- te_te
    train_df[[paste0("cnt_", col)]] <- cnt_tr
    test_df[[paste0("cnt_", col)]] <- cnt_te
    train_df[[paste0("le_", col)]] <- as.numeric(factor(tr_v, levels=lv))
    test_df[[paste0("le_", col)]] <- as.numeric(factor(te_v, levels=lv))
    train_df[[col]] <- NULL; test_df[[col]] <- NULL
  }
  list(train=train_df, test=test_df)
}

res <- add_cat_encodings(признаки_обуч, признаки_тест, cat_cols, y_log, веса, номер_фолда, PRIOR_TE)
признаки_обуч <- res$train
признаки_тест <- res$test


drop_nonportable_best <- c("first_salary_income", "first_salary_income_isna")
признаки_обуч <- признаки_обуч[, !names(признаки_обуч) %in% drop_nonportable_best, drop=FALSE]
признаки_тест <- признаки_тест[, !names(признаки_тест) %in% drop_nonportable_best, drop=FALSE]
pred_cols <- names(признаки_обуч)[sapply(признаки_обуч, is.numeric)]
for (c0 in setdiff(pred_cols, names(признаки_тест))) признаки_тест[[c0]] <- NA_real_

X_train <- as.matrix(признаки_обуч[pred_cols])
X_test <- as.matrix(признаки_тест[pred_cols])
comp_tr <- признаки_обуч$completeness
comp_te <- признаки_тест$completeness

cat("Матрица:", nrow(X_train), "x", ncol(X_train), "\n")
d_test_xgb <- xgb.DMatrix(X_test, missing=NA)


KINDS <- c("xgb_log", "xgb_mae", if (HAVE_LGB) "lgb_l1" else "xgb_logmae")
cat("\nМодели:", paste(KINDS, collapse=", "), "\n")

wmae_eval_raw <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label"); w <- getinfo(dtrain, "weight")
  list(metric="wmae", value=sum(w*abs(y-preds))/sum(w))
}
wmae_eval_log <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label"); w <- getinfo(dtrain, "weight")
  list(metric="wmae", value=sum(w*abs(expm1(y)-expm1(preds)))/sum(w))
}

kind_spec <- function(kind) {
  switch(kind,
         xgb_log=list(lab=y_log, obj="reg:squarederror", log=TRUE, fe=wmae_eval_log),
         xgb_mae=list(lab=целевая, obj="reg:absoluteerror", log=FALSE, fe=wmae_eval_raw),
         xgb_logmae=list(lab=y_log, obj="reg:absoluteerror", log=TRUE, fe=wmae_eval_log))
}

get_best_iter <- function(m, default) {
  for (b in list(
    tryCatch(as.integer(m$best_iteration), error=function(e) NA_integer_),
    tryCatch(as.integer(attr(m, "best_iteration")), error=function(e) NA_integer_),
    tryCatch(as.integer(xgb.attributes(m)$best_iteration), error=function(e) NA_integer_)
  )) {
    if (length(b) == 1 && !is.na(b) && b >= 1) return(b)
  }
  default
}

run_cv <- function(kind) {
  oof <- rep(NA_real_, n_tr)
  best_iters <- integer(0)
  test_bag <- rep(0, nrow(X_test))
  for (f in 1:N_FOLDS) {
    tr <- номер_фолда != f; va <- !tr
    if (kind == "lgb_l1") {
      dtr <- lgb.Dataset(X_train[tr,,drop=FALSE], label=целевая[tr], weight=веса[tr])
      dva <- tryCatch(lgb.Dataset.create.valid(dtr, data=X_train[va,,drop=FALSE],
                                               label=целевая[va], weight=веса[va]),
                      error=function(e) lgb.Dataset(X_train[va,,drop=FALSE], label=целевая[va], weight=веса[va]))
      m <- lgb.train(params=c(lgb_params, list(seed=100+f)), data=dtr, nrounds=MAX_ROUNDS,
                     valids=list(valid=dva), early_stopping_rounds=EARLY_STOP, verbose=-1)
      b <- m$best_iter; if (is.null(b) || b < 1) b <- MAX_ROUNDS
      p <- tryCatch(predict(m, X_train[va,,drop=FALSE], num_iteration=b),
                    error=function(e) predict(m, X_train[va,,drop=FALSE]))
      pt <- tryCatch(predict(m, X_test, num_iteration=b), error=function(e) predict(m, X_test))
    } else {
      sp <- kind_spec(kind)
      cur_params <- switch(kind, xgb_log=params_xgb_log, xgb_mae=params_xgb_mae, xgb_logmae=params_xgb_logmae)
      dtr <- xgb.DMatrix(X_train[tr,,drop=FALSE], label=sp$lab[tr], weight=веса[tr], missing=NA)
      dva <- xgb.DMatrix(X_train[va,,drop=FALSE], label=sp$lab[va], weight=веса[va], missing=NA)
      set.seed(100+f)
      m <- xgb.train(params=c(cur_params, list(objective=sp$obj)), data=dtr, nrounds=MAX_ROUNDS,
                     evals=list(valid=dva), custom_metric=sp$fe, maximize=FALSE,
                     early_stopping_rounds=EARLY_STOP, verbose=0)
      b <- get_best_iter(m, MAX_ROUNDS)
      p <- tryCatch(predict(m, dva, iterationrange=c(1,b)), error=function(e) predict(m, dva))
      pt <- tryCatch(predict(m, d_test_xgb, iterationrange=c(1,b)), error=function(e) predict(m, d_test_xgb))
      if (sp$log) { p <- expm1(p); pt <- expm1(pt) }
    }
    best_iters <- c(best_iters, b)
    oof[va] <- p
    test_bag <- test_bag + pt/N_FOLDS
    cat(sprintf("  [%s] fold %d: best_iter=%d, WMAE=%.1f\n",
                kind, f, b, wmae(целевая[va], p, веса[va])))
  }
  list(oof=oof, best_iters=best_iters, test_bag=test_bag)
}

результаты <- list()
for (k in KINDS) {
  cat("\n=== CV:", k, "===\n")
  результаты[[k]] <- run_cv(k)
  cat("OOF WMAE (", k, "):", round(wmae(целевая, результаты[[k]]$oof, веса), 1), "\n")
}


oof_mat <- do.call(cbind, lapply(результаты, `[[`, "oof"))
K <- ncol(oof_mat)

set.seed(7)
W <- rbind(diag(K), matrix(1/K, 1, K),
           { M <- matrix(rexp(4000*K), ncol=K); M/rowSums(M) })

sc <- numeric(nrow(W))
for (i in seq(1, nrow(W), by=200)) {
  j <- i:min(i+199, nrow(W))
  P <- oof_mat %*% t(W[j,,drop=FALSE])
  sc[j] <- colSums(веса*abs(P-целевая))/sum(веса)
}

w_best <- W[which.min(sc), ]
oof_blend <- as.vector(oof_mat %*% w_best)
cat(sprintf("\nВеса ансамбля: %s -> OOF WMAE=%.1f\n",
            paste(round(w_best, 3), collapse=" / "), min(sc)))


cals <- seq(0.88, 1.12, by=0.005)
sc_c <- sapply(cals, function(cc) wmae(целевая, cc*oof_blend, веса))
c_global <- cals[which.min(sc_c)]

qs_comp <- unique(quantile(comp_tr, c(0.25, 0.5, 0.75)))
grp_of <- function(x) findInterval(x, qs_comp) + 1
g_tr <- grp_of(comp_tr)
g_te <- grp_of(comp_te)
n_grp <- max(g_tr)
c_grp <- rep(c_global, n_grp)

for (g in 1:n_grp) {
  ii <- which(g_tr == g)
  if (length(ii) < 3000) next
  sc_g <- sapply(cals, function(cc) wmae(целевая[ii], cc*oof_blend[ii], веса[ii]))
  c_grp[g] <- cals[which.min(sc_g)]
}

oof_final <- oof_blend * c_grp[g_tr]
cat("Коэфф. глоб:", c_global, "| по группам:", paste(round(c_grp, 3), collapse=" / "), "\n")
cat("OOF WMAE (финал):", round(wmae(целевая, oof_final, веса), 1), "\n")


full_predict <- function(kind, nrounds_k) {
  acc <- rep(0, nrow(X_test))
  for (s in SEEDS) {
    if (kind == "lgb_l1") {
      dfull <- lgb.Dataset(X_train, label=целевая, weight=веса)
      m <- lgb.train(params=c(lgb_params, list(seed=s, bagging_seed=s,
                                               feature_fraction_seed=s, data_random_seed=s)), data=dfull, nrounds=nrounds_k, verbose=-1)
      p <- tryCatch(predict(m, X_test, num_iteration=nrounds_k), error=function(e) predict(m, X_test))
    } else {
      sp <- kind_spec(kind)
      cur_params <- switch(kind, xgb_log=params_xgb_log, xgb_mae=params_xgb_mae, xgb_logmae=params_xgb_logmae)
      d <- xgb.DMatrix(X_train, label=sp$lab, weight=веса, missing=NA)
      set.seed(s)
      m <- xgb.train(params=c(cur_params, list(objective=sp$obj, seed=s)), data=d, nrounds=nrounds_k, verbose=0)
      p <- predict(m, d_test_xgb)
      if (sp$log) p <- expm1(p)
      if (kind == "xgb_log" && s == SEEDS[1]) final_log_model <<- m
    }
    acc <- acc + p/length(SEEDS)
  }
  acc
}

final_log_model <- NULL
test_mat <- matrix(0, nrow(X_test), K, dimnames=list(NULL, KINDS))

for (k in KINDS) {
  nr <- max(200, ceiling(1.05*mean(результаты[[k]]$best_iters)))
  cat("Retrain:", k, "| nrounds=", nr, "\n")
  full <- full_predict(k, nr)
  test_mat[, k] <- 0.5*результаты[[k]]$test_bag + 0.5*full
}

test_pred <- as.vector(test_mat %*% w_best) * c_grp[g_te]
lo <- min(целевая)
hi <- unname(quantile(целевая, 0.999))
test_pred <- pmin(pmax(test_pred, lo), hi)



cat("Финальная OOF WMAE:", round(wmae(целевая, oof_final, веса), 1), "\n")



cat("\n FROZEN V11 RESIDUAL — BEST 74564.76\n")
base_pred_tr_best <- as.numeric(oof_blend)
base_pred_te_best <- as.numeric(test_mat %*% w_best)
y_best <- as.numeric(целевая)
weights_best <- as.numeric(веса)
dt_train_best <- as.Date(данные_обуч$dt)
dt_test_best <- as.Date(данные_тест$dt)

wmae_best <- function(y, p, w) sum(w * abs(y - p)) / sum(w)
safe_row_median_best <- function(m) {
  as.numeric(apply(m, 1L, function(z) {
    z <- z[is.finite(z) & z > 0]
    if (length(z) == 0L) NA_real_ else stats::median(z)
  }))
}

make_residual_features_best <- function(df, base_pred, component_pred, dates) {
  salary_sources <- intersect(c(
    "salary_6to12m_avg", "incomeValue", "dp_ils_avg_salary_1y",
    "dp_ils_avg_salary_2y", "dp_ils_avg_salary_3y",
    "dp_payoutincomedata_payout_avg_6_month",
    "dp_payoutincomedata_payout_avg_3_month",
    "dp_ils_paymentssum_avg_12m"
  ), names(df))
  salary_matrix <- as.matrix(df[,salary_sources,drop=FALSE])
  storage.mode(salary_matrix) <- "double"
  salary_matrix[!is.finite(salary_matrix) | salary_matrix <= 0] <- NA_real_
  salary_matrix <- pmin(salary_matrix, 5e6)
  salary_consensus <- safe_row_median_best(salary_matrix)
  salary_count <- rowSums(is.finite(salary_matrix))
  log_salary <- log1p(salary_matrix)
  salary_disagreement <- apply(log_salary, 1L, function(z) {
    z <- z[is.finite(z)]
    if (length(z) <= 1L) 0 else stats::median(abs(z - stats::median(z)))
  })
  salary_confidence <- pmin(salary_count / 4, 1) * exp(-salary_disagreement)
  salary_confidence[!is.finite(salary_confidence)] <- 0

  component_pred <- as.matrix(component_pred)
  component_sd <- apply(component_pred, 1L, stats::sd, na.rm=TRUE)
  component_min <- apply(component_pred, 1L, min, na.rm=TRUE)
  component_max <- apply(component_pred, 1L, max, na.rm=TRUE)
  compact <- intersect(c(
    "salary_6to12m_avg", "incomeValue", "dp_ils_avg_salary_1y",
    "dp_ils_avg_salary_2y", "dp_ils_avg_salary_3y",
    "dp_payoutincomedata_payout_avg_6_month",
    "dp_payoutincomedata_payout_avg_3_month",
    "dp_ils_paymentssum_avg_12m",
    "hdb_bki_total_max_limit", "hdb_bki_total_cc_max_limit",
    "hdb_bki_total_pil_max_limit", "hdb_outstand_sum",
    "turn_cur_cr_avg_act_v2", "turn_cur_db_avg_act_v2",
    "turn_cur_cr_sum_v2", "turn_cur_db_sum_v2",
    "avg_cur_cr_turn", "avg_cur_db_turn", "avg_6m_all",
    "total_rur_amt_cm_avg", "per_capita_income_rur_amt",
    "credit_util", "debt_to_income", "limit_to_income",
    "income_to_region", "trend_db", "savings_rate",
    "income_proxy", "completeness", "n_missing", "segment_num"
  ), names(df))
  result <- data.frame(df[,compact,drop=FALSE], check.names=FALSE)
  result$meta_base_pred <- base_pred
  result$meta_log_base <- log1p(pmax(base_pred, 0))
  result$meta_component_sd <- component_sd
  result$meta_component_range <- component_max - component_min
  result$proxy_consensus <- salary_consensus
  result$proxy_log_consensus <- log1p(salary_consensus)
  result$proxy_source_count <- salary_count
  result$proxy_disagreement <- salary_disagreement
  result$proxy_confidence <- salary_confidence
  result$proxy_log_gap <- salary_confidence * pmin(
    pmax(log1p(salary_consensus) - log1p(pmax(base_pred, 0)), -2.5), 2.5
  )
  result$proxy_to_base <- salary_consensus / pmax(base_pred, 1000)
  result$month_index <- as.numeric(format(dates, "%Y")) * 12 +
    as.numeric(format(dates, "%m"))
  result$is_post_april <- as.numeric(dates >= as.Date("2024-05-01"))
  for (thr in c(60, 100, 150, 250, 400, 600)) {
    result[[paste0("base_ge_", thr, "k")]] <- as.numeric(base_pred >= thr * 1000)
  }
  for (col in salary_sources) {
    result[[paste0(col, "_missing")]] <- as.numeric(is.na(df[[col]]))
  }
  for (col in names(result)) {
    result[[col]] <- suppressWarnings(as.numeric(result[[col]]))
    result[[col]][!is.finite(result[[col]])] <- NA_real_
  }
  result
}

res_features_tr_best <- make_residual_features_best(
  признаки_обуч, base_pred_tr_best, oof_mat, dt_train_best
)
res_features_te_best <- make_residual_features_best(
  признаки_тест, base_pred_te_best, test_mat, dt_test_best
)
common_res_best <- intersect(
  names(res_features_tr_best), names(res_features_te_best)
)
X_res_tr_best <- as.matrix(res_features_tr_best[,common_res_best,drop=FALSE])
X_res_te_best <- as.matrix(res_features_te_best[,common_res_best,drop=FALSE])
residual_target_best <- y_best - base_pred_tr_best
cat("Residual features:", ncol(X_res_tr_best), "\n")

params_dev_best <- list(
  objective="reg:absoluteerror", eval_metric="mae",
  eta=0.0183904354896039, max_depth=7L,
  min_child_weight=125.491457254664,
  subsample=0.796883089572657,
  colsample_bytree=0.707180834189057,
  lambda=74.2853858412068, alpha=7.979771181,
  gamma=0.4436061, tree_method="hist", seed=11119L
)
params_final_best <- params_dev_best
params_final_best$seed <- 11211L
iter_april_best <- 368L
iter_may_best <- 271L
iter_june_best <- 320L
iter_full_best <- 346L
strength_best <- 1.1
correction_min_best <- -40000
correction_max_best <- 80000

predict_fixed_best <- function(model, matrix, iterations) {
  d <- xgb.DMatrix(matrix, missing=NA)
  tryCatch(
    predict(model, d, iterationrange=c(1, iterations)),
    error=function(e) predict(model, d)
  )
}
apply_correction_best <- function(base_prediction, raw_correction) {
  correction <- pmin(
    pmax(raw_correction, correction_min_best), correction_max_best
  )
  pmax(base_prediction + strength_best * correction, 0)
}

idx_train_april_best <- which(dt_train_best < as.Date("2024-04-01"))
idx_valid_april_best <- which(
  dt_train_best >= as.Date("2024-04-01") &
    dt_train_best < as.Date("2024-05-01")
)
idx_train_may_best <- which(dt_train_best < as.Date("2024-05-01"))
idx_valid_may_best <- which(
  dt_train_best >= as.Date("2024-05-01") &
    dt_train_best < as.Date("2024-06-01")
)
idx_train_june_best <- which(dt_train_best < as.Date("2024-06-01"))
idx_valid_june_best <- which(dt_train_best >= as.Date("2024-06-01"))

train_predict_best <- function(train_idx, valid_idx, params, iterations) {
  dtrain <- xgb.DMatrix(
    X_res_tr_best[train_idx,,drop=FALSE],
    label=residual_target_best[train_idx], weight=weights_best[train_idx],
    missing=NA
  )
  model <- xgb.train(
    params=params, data=dtrain, nrounds=iterations, verbose=0
  )
  raw <- predict_fixed_best(
    model, X_res_tr_best[valid_idx,,drop=FALSE], iterations
  )
  rm(model, dtrain)
  raw
}

raw_april_best <- train_predict_best(
  idx_train_april_best, idx_valid_april_best,
  params_dev_best, iter_april_best
)
raw_may_best <- train_predict_best(
  idx_train_may_best, idx_valid_may_best,
  params_dev_best, iter_may_best
)
raw_june_best <- train_predict_best(
  idx_train_june_best, idx_valid_june_best,
  params_final_best, iter_june_best
)
pred_april_best <- apply_correction_best(
  base_pred_tr_best[idx_valid_april_best], raw_april_best
)
pred_may_best <- apply_correction_best(
  base_pred_tr_best[idx_valid_may_best], raw_may_best
)
pred_june_best <- apply_correction_best(
  base_pred_tr_best[idx_valid_june_best], raw_june_best
)

dfull_best <- xgb.DMatrix(
  X_res_tr_best, label=residual_target_best, weight=weights_best, missing=NA
)
model_full_best <- xgb.train(
  params=params_final_best, data=dfull_best,
  nrounds=iter_full_best, verbose=0
)
raw_test_best <- predict_fixed_best(
  model_full_best, X_res_te_best, iter_full_best
)
pred_test_candidate_best <- apply_correction_best(
  base_pred_te_best, raw_test_best
)
pred_test_candidate_best <- pmin(
  pmax(pred_test_candidate_best, min(y_best, na.rm=TRUE)),
  unname(quantile(y_best, 0.9997, na.rm=TRUE, names=FALSE))
)


base_april_wmae_best <- wmae_best(y_best[idx_valid_april_best], base_pred_tr_best[idx_valid_april_best], weights_best[idx_valid_april_best])
base_may_wmae_best <- wmae_best(y_best[idx_valid_may_best], base_pred_tr_best[idx_valid_may_best], weights_best[idx_valid_may_best])
base_june_wmae_best <- wmae_best(y_best[idx_valid_june_best], base_pred_tr_best[idx_valid_june_best], weights_best[idx_valid_june_best])
april_wmae_best <- wmae_best(y_best[idx_valid_april_best], pred_april_best, weights_best[idx_valid_april_best])
may_wmae_best <- wmae_best(y_best[idx_valid_may_best], pred_may_best, weights_best[idx_valid_may_best])
june_wmae_best <- wmae_best(y_best[idx_valid_june_best], pred_june_best, weights_best[idx_valid_june_best])

cat("Residual features:", ncol(X_res_tr_best), "\n")
cat("Fixed iterations April/May/June/full: 368 / 271 / 320 / 346\n")
cat("Fixed post: strength=1.1 clip=[-40000, 80000]\n")
cat("April base/final WMAE:", round(base_april_wmae_best, 2), "/", round(april_wmae_best, 2), "\n")
cat("May base/final WMAE:", round(base_may_wmae_best, 2), "/", round(may_wmae_best, 2), "\n")
cat("June LOCKED base/final WMAE:", round(base_june_wmae_best, 2), "/", round(june_wmae_best, 2), "\n")

validation_best <- data.frame(
  split=c("April development", "May development", "June locked"),
  base_wmae=c(base_april_wmae_best, base_may_wmae_best, base_june_wmae_best),
  final_wmae=c(april_wmae_best, may_wmae_best, june_wmae_best),
  gain=c(base_april_wmae_best-april_wmae_best, base_may_wmae_best-may_wmae_best, base_june_wmae_best-june_wmae_best)
)
write.csv(validation_best, VALIDATION_FILE_BEST, row.names=FALSE)
submission_best <- data.frame(id=id_тест, predict=pred_test_candidate_best)
write.table(submission_best, OUTPUT_FILE_BEST, sep=";", dec=",", row.names=FALSE, quote=FALSE)
cat("Saved final submission:", OUTPUT_FILE_BEST, "\n")
cat("Saved validation:", VALIDATION_FILE_BEST, "\n")
cat("Rows:", nrow(submission_best), "| finite predictions:", all(is.finite(submission_best$predict)), "\n")
sessionInfo()
