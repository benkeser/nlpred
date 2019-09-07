
<!-- README.md is generated from README.Rmd. Please edit that file -->

# R/`nlpred`

[![Travis-CI Build
Status](https://travis-ci.org/benkeser/nlpred.svg?branch=master)](https://travis-ci.org/benkeser/nlpred)
[![AppVeyor Build
Status](https://ci.appveyor.com/api/projects/status/github/benkeser/nlpred?branch=master&svg=true)](https://ci.appveyor.com/project/benkeser/nlpred)
[![Coverage
Status](https://img.shields.io/codecov/c/github/benkeser/nlpred/master.svg)](https://codecov.io/github/benkeser/nlpred?branch=master)
<!-- [![CRAN](http://www.r-pkg.org/badges/version/nlpred)](http://www.r-pkg.org/pkg/nlpred) -->
<!-- [![CRAN downloads](https://cranlogs.r-pkg.org/badges/nlpred)](https://CRAN.R-project.org/package=nlpred) -->
[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![MIT
license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.835868.svg)](https://doi.org/10.5281/zenodo.835868) -->

> Small-sample optimized estimators of cross-validated prediction
> metrics

[David Benkeser](https://www.benkeserstatistics.com/)

-----

## Description

`nlpred` is an R package for computing estimates of cross-validated
prediction metrics. These estimates are tailored for superior
performance in small samples. Several estimators are available including
ones based cross-validated targeted minimum loss-based estimation,
estimating equations, and one-step estimation.

-----

## Installation

For standard use, we recommend installing the package from
[CRAN](https://cran.r-project.org/) via

``` r
install.packages("nlpred")
```

You can install the current release of `nlpred` from GitHub via
[`devtools`](https://www.rstudio.com/products/rpackages/devtools/) with:

``` r
devtools::install_github("benkeser/nlpred")
```

-----

## Usage

The main functions in the package are `cv_auc` and `cv_scrnp`, which are
used to compute, respectively, the `K`-fold [cross-validated area under
the receiver operating characteristics
curve](http://projecteuclid.org/euclid.ejs/1437742107) (CVAUC) and the
`K`-fold cross-validated [sensitivity constrained rate of negative
prediction](https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.7296).
However, rather than using standard cross-validation estimators (where
prediction algorithms are developed in a training sample and AUC/SCRNP
estimated using the validation sample), we instead use techniques from
efficiency theory to estimate these quantities. This allows us to use
the training data both to develop the prediction algorithm, *as well as*
key nuisance parameters needed to evaluate AUC/SCRNP. By reserving more
data for estimation of these key parameters, we obtain improved
performance in small samples.

``` r
# load package
library(nlpred)
#> Loading required package: data.table

# simulate data
n <- 200
p <- 10
X <- data.frame(matrix(rnorm(n*p), nrow = n, ncol = p))
Y <- rbinom(n, 1, plogis(X[,1] + X[,10]))

# get cv auc estimates for logistic regression
logistic_cv_auc_ests <- cv_auc(Y = Y, X = X, K = 5, learner = "glm_wrapper")
logistic_cv_auc_ests
#>                est         se       cil       ciu
#> cvtmle   0.7900437 0.03133407 0.7286300 0.8514573
#> onestep  0.7904006 0.03187774 0.7279214 0.8528798
#> esteq    0.7837634 0.03187774 0.7212842 0.8462427
#> standard 0.7860652 0.03210325 0.7231440 0.8489864

# get cv auc estimates for random forest using nested 
# cross-validation for nuisance parameter estimation. nested
# cross-validation is unfortunately necessary when aggressive learners 
# are used. 
rf_cv_auc_ests <- cv_auc(Y = Y, X = X, K = 5, 
                         learner = "randomforest_wrapper", 
                         nested_cv = TRUE)
rf_cv_auc_ests
#>                est         se       cil       ciu
#> cvtmle   0.7346812 0.03619140 0.6637474 0.8056151
#> onestep  0.7349126 0.03650895 0.6633564 0.8064689
#> esteq    0.7320209 0.03650895 0.6604646 0.8035771
#> standard 0.7370301 0.03468265 0.6690533 0.8050068

# same examples for scrnp
logistic_cv_scrnp_ests <- cv_scrnp(Y = Y, X = X, K = 5, learner = "glm_wrapper")
#> 
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   
logistic_cv_scrnp_ests
#>                 est         se         cil       ciu
#> cvtmle   0.08467894 0.02983346 0.026206429 0.1431514
#> onestep  0.06639411 0.03004345 0.007510036 0.1252782
#> esteq    0.06639411 0.03004345 0.007510036 0.1252782
#> standard 0.14294586 0.02782372 0.088412385 0.1974793


rf_cv_scrnp_ests <- cv_scrnp(Y = Y, X = X, K = 5, 
                         learner = "randomforest_wrapper", 
                         nested_cv = TRUE)
#> 
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   

Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 |
Multistart 1 of 1 /
Multistart 1 of 1 |
Multistart 1 of 1 |
                   
rf_cv_scrnp_ests
#>                 est        se         cil       ciu
#> cvtmle   0.08611932 0.0323499  0.02271469 0.1495240
#> onestep  0.04881878 0.0327503 -0.01537063 0.1130082
#> esteq    0.04881878 0.0327503 -0.01537063 0.1130082
#> standard 0.06968534 0.0301356  0.01062064 0.1287500
```

-----

## Issues

If you encounter any bugs or have any specific feature requests, please
[file an issue](https://github.com/benkeser/nlpred/issues).

-----

## Contributions

Interested contributors can consult our [`contribution
guidelines`](https://github.com/benkeser/nlpred/blob/master/CONTRIBUTING.md)
prior to submitting a pull request.

-----

## Citation

## License

© 2019 [David Benkeser](http://www.benkeserstatistics.com)

The contents of this repository are distributed under the MIT license.
See below for details:

    The MIT License (MIT)
    
    Copyright (c) 2019 David C. Benkeser
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
