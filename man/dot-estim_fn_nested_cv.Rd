% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/auc_functions.R
\name{.estim_fn_nested_cv}
\alias{.estim_fn_nested_cv}
\title{An estimating function for cvAUC with initial estimates generated via 
nested cross-validation}
\usage{
.estim_fn_nested_cv(auc = 0.5, prediction_list, folds, gn, K)
}
\arguments{
\item{auc}{The value of auc to find root for}

\item{prediction_list}{Entry in prediction_list}

\item{folds}{Cross-validation folds}

\item{gn}{Marginal probability of outcome}

\item{K}{Number of CV folds}
}
\value{
A numeric value of the estimating function evaluated at current
\code{auc} estimate.
}
\description{
An estimating function for cvAUC with initial estimates generated via 
nested cross-validation
}
