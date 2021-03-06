% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/auc_functions.R
\name{.get_auc}
\alias{.get_auc}
\title{Compute the AUC given the cdf and pdf of psi}
\usage{
.get_auc(dist_y0, dist_y1)
}
\arguments{
\item{dist_y0}{Distribution of psi given Y = 0}

\item{dist_y1}{Distribution of psi given Y = 1}
}
\value{
Numeric value of AUC
}
\description{
See \code{?.get_psi_distribution} to understand expected input format
}
