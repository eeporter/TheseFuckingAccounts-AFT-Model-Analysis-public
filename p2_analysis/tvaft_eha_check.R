# Cross-check of tvaft.TimeVaryingAFT against R eha::aftreg (Brostrom), which accumulates an AFT clock across spells
# linked by `id`. With constant covariates the two agree exactly. With time-varying covariates eha 2.12.0's event term
# uses (gamma - 1) * (log(t) - alpha + z.beta), the current-covariate scaled time, instead of (gamma - 1) * log(clock),
# so its density does not integrate to 1. eha_aftreg_tv_fix.patch changes that one term; set EHA_LIB to a library
# holding the patched build to use it (tvaft_validation.ipynb builds one in a temporary directory).
# Usage: Rscript tvaft_eha_check.R <episodes.csv> <subjects.csv> <dist> <out.csv>
#   episodes.csv: id, start, stop, covariates...   subjects.csv: id, event[, entry]
# Writes one row per parameter: term, coef, se, loglik. Coefficients use param = "lifeExp" (exp(coef) = time ratio,
# same sign as tvaft); baseline terms are log(scale) = mu and log(shape) = -log_sigma.
# Reads: the two CSVs (same layout as tvaft's fit inputs, subjects with id as a column). Writes: <out.csv>, also printed.
# <dist> is passed to aftreg as is (e.g. "weibull", "lognormal", "loglogistic"). Model-based SEs from fit$var.

# Without EHA_LIB this loads the eha installed in the user library (unpatched unless patched there).
eha_lib <- Sys.getenv("EHA_LIB", Sys.getenv("R_LIBS_USER"))
suppressPackageStartupMessages(library(eha, lib.loc = eha_lib))
args <- commandArgs(trailingOnly = TRUE)
ep <- read.csv(args[1])
sb <- read.csv(args[2])
dist <- args[3]

# One spell per row, ordered within id. The event indicator goes on each subject's last spell only; other spells are 0.
ep <- ep[order(ep$id, ep$start), ]
last <- !duplicated(ep$id, fromLast = TRUE)
ep$event <- 0
ep$event[last] <- sb$event[match(ep$id[last], sb$id)]
if ("entry" %in% names(sb)) {                       # delayed entry: drop the spells before entry
  # keeps spells starting at or after entry (1e-9 float tolerance), so a spell boundary must fall on entry (tvaft.split_at)
  entry <- sb$entry[match(ep$id, sb$id)]
  ep <- ep[ep$start >= entry - 1e-9, ]
}
# every column other than id/start/stop/event is used as a covariate
covs <- setdiff(names(ep), c("id", "start", "stop", "event"))
f <- as.formula(paste("Surv(start, stop, event) ~", paste(covs, collapse = " + ")))
fit <- aftreg(f, data = ep, id = id, dist = dist, param = "lifeExp")
# fit$loglik[2] is the log-likelihood at the fitted values (loglik[1] is at the start values)
out <- data.frame(term = names(fit$coefficients), coef = unname(fit$coefficients),
                  se = sqrt(diag(fit$var)), loglik = fit$loglik[2])
write.csv(out, args[4], row.names = FALSE)
print(out)
