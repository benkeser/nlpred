% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/auc_functions.R
\name{.get_predictions}
\alias{.get_predictions}
\title{Worker function for fitting prediction functions (possibly in parallel)}
\usage{
.get_predictions(
  learner,
  Y,
  X,
  K = 10,
  folds,
  parallel,
  nested_cv = FALSE,
  nested_K = K - 1
)
}
\arguments{
\item{learner}{The wrapper to use}

\item{Y}{The outcome}

\item{X}{The predictors}

\item{K}{The number of folds}

\item{folds}{Vector of CV fold assignments}

\item{parallel}{Whether to compute things in parallel using future}

\item{nested_cv}{Is nested CV being used?}

\item{nested_K}{How many folds of nested CV?}
}
\value{
A list of the result of the wrapper executed in each fold
}
\description{
Worker function for fitting prediction functions (possibly in parallel)
}
