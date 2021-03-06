% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/auc_functions.R
\name{one_boot_auc}
\alias{one_boot_auc}
\title{Internal function used to perform one bootstrap sample. The function
\code{try}s to fit \code{learner} on a bootstrap sample. If for some reason
(e.g., the bootstrap sample contains no observations with \code{Y = 1}) 
the learner fails, then the function returns \code{NA}. These \code{NA}s 
are ignored later when computing the bootstrap corrected estimate.}
\usage{
one_boot_auc(Y, X, n, correct632, learner)
}
\arguments{
\item{Y}{A numeric binary outcome}

\item{X}{A \code{data.frame} of variables for prediction.}

\item{n}{Number of observations}

\item{correct632}{A boolean indicating whether to use the .632 correction.}

\item{learner}{A wrapper that implements the desired method for building a 
prediction algorithm. See \code{?glm_wrapper} or read the package vignette
for more information on formatting \code{learner}s.}
}
\value{
If \code{learner} executes successfully, a numeric estimate of AUC
on this bootstrap sample. Otherwise the function returns \code{NA}.
}
\description{
Internal function used to perform one bootstrap sample. The function
\code{try}s to fit \code{learner} on a bootstrap sample. If for some reason
(e.g., the bootstrap sample contains no observations with \code{Y = 1}) 
the learner fails, then the function returns \code{NA}. These \code{NA}s 
are ignored later when computing the bootstrap corrected estimate.
}
