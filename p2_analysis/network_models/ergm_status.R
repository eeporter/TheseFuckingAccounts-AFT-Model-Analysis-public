# Step 3: ERGMs on the case-study alter-alter networks (ego removed). Exploratory, like the notebooks they read.
# Question: do suspended (and deleted) alters tie to each other more than chance, net of activity?
# Usage: Rscript ergm_status.R <ego>   (reads network_models_output/step3_inputs/, writes network_models_output/ and
# p2_figures/network_models_step3/; refuses to overwrite)
suppressPackageStartupMessages({library(network); library(ergm)})
args <- commandArgs(trailingOnly = TRUE)
ego <- args[1]
# Folders are found relative to this script: p2 = p2_analysis/.
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
p2 <- dirname(here)
out <- file.path(p2, "network_models_output")
fig <- file.path(dirname(p2), "p2_figures", "network_models_step3")
dir.create(fig, showWarnings = FALSE, recursive = TRUE)
# Path inside dir; stops if the file already exists.
new_path <- function(dir, name) { p <- file.path(dir, name); if (file.exists(p)) stop("exists, not overwritten: ", p); p }

nodes <- read.csv(file.path(out, "step3_inputs", paste0("nodes_", ego, ".csv")), stringsAsFactors = FALSE)
ties <- read.csv(file.path(out, "step3_inputs", paste0("ties_", ego, ".csv")), stringsAsFactors = FALSE)
# Everything printed from here on also goes to the log file.
log_path <- new_path(out, paste0("step3_ergm_", ego, ".log"))
sink(log_path, split = TRUE)
set.seed(2026)
cat("ego", ego, " nodes", nrow(nodes), " ties", nrow(ties), "\n")
# Undirected alter-alter network with status and activity as node attributes.
net <- network(ties[, c("a", "b")], vertices = nodes, directed = FALSE, matrix.type = "edgelist")
print(net)
cat("status mix:\n"); print(table(net %v% "status"))
cat("mixing matrix (ties by status pair):\n"); print(mixingmatrix(net, "status"))

# Terms: edges; status main effects (Active is the reference); same-status ties per status; activity as a node covariate.
lev <- sort(unique(net %v% "status"))
base <- ~ edges + nodefactor("status", levels = -which(lev == "Active")) +
  nodematch("status", diff = TRUE) + nodecov("activity")
# MCMC settings for the dyad-dependent model.
ctrl <- control.ergm(MCMC.samplesize = 4096, MCMC.interval = 2048, MCMC.burnin = 100000,
                     MCMLE.maxit = 40, parallel = 4, parallel.type = "PSOCK", seed = 2026)

cat("\n==== M0: dyad-independent (exact logit by MPLE) ====\n")
m0 <- ergm(update(base, net ~ .))
print(summary(m0))

cat("\n==== M1: + gwesp(0.25, fixed) ====\n")
m1 <- tryCatch(ergm(update(base, net ~ . + gwesp(0.25, fixed = TRUE)), control = ctrl),
               error = function(e) { cat("M1 failed:", conditionMessage(e), "\n"); NULL })
res <- list(M0 = m0)
if (!is.null(m1)) {
  print(summary(m1))
  res$M1 <- m1
  pdf(new_path(fig, paste0("mcmc_diagnostics_", ego, ".pdf")))
  cat("\n---- MCMC diagnostics ----\n")
  print(mcmc.diagnostics(m1))
  dev.off()
}
# Goodness of fit on degree and edgewise shared partners, with plots.
for (nm in names(res)) {
  g <- gof(res[[nm]], GOF = ~ degree + espartners + model)
  cat("\n---- GOF", nm, "----\n"); print(g)
  pdf(new_path(fig, paste0("gof_", nm, "_", ego, ".pdf"))); par(mfrow = c(2, 2)); plot(g); dev.off()
}
# Coefficient table for every model that fitted, with odds ratios and AIC.
co <- do.call(rbind, lapply(names(res), function(nm) {
  s <- summary(res[[nm]])$coefficients
  data.frame(ego = ego, model = nm, term = rownames(s), estimate = as.numeric(s[, 1]), se = as.numeric(s[, 2]), p = as.numeric(s[, ncol(s)]),
             odds_ratio = exp(as.numeric(s[, 1])), AIC = as.numeric(AIC(res[[nm]]))[1], row.names = NULL)
}))
write.csv(co, new_path(out, paste0("step3_ergm_coefs_", ego, ".csv")), row.names = FALSE)
saveRDS(res, new_path(out, paste0("step3_ergm_fits_", ego, ".rds")))
sink()
