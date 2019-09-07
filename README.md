
<!-- README.md is generated from README.Rmd. Please edit that file -->

# R/`nlpred`

[![Travis-CI Build
Status](https://travis-ci.org/benkeser/nlpred.svg?branch=master)](https://travis-ci.org/benkeser/nlpred)
[![AppVeyor Build
Status](https://ci.appveyor.com/api/projects/status/github/benkeser/nlpred?branch=master&svg=true)](https://ci.appveyor.com/project/benkeser/nlpred)
[![Coverage
Status](https://img.shields.io/codecov/c/github/benkeser/nlpred/master.svg)](https://codecov.io/github/benkeser/nlpred?branch=master)
[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![MIT
license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)
<!-- [![CRAN](http://www.r-pkg.org/badges/version/nlpred)](http://www.r-pkg.org/pkg/nlpred) -->
<!-- [![CRAN downloads](https://cranlogs.r-pkg.org/badges/nlpred)](https://CRAN.R-project.org/package=nlpred) -->
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

# turn off messages from np package
options(np.messages=FALSE)

# simulate data
n <- 200
p <- 10
X <- data.frame(matrix(rnorm(n*p), nrow = n, ncol = p))
Y <- rbinom(n, 1, plogis(X[,1] + X[,10]))

# get cv auc estimates for logistic regression
logistic_cv_auc_ests <- cv_auc(Y = Y, X = X, K = 5, learner = "glm_wrapper")
logistic_cv_auc_ests
#>                est         se       cil       ciu
#> cvtmle   0.7112994 0.03566441 0.6413985 0.7812004
#> onestep  0.7127992 0.03621594 0.6418173 0.7837811
#> esteq    0.7000396 0.03621594 0.6290577 0.7710216
#> standard 0.7000213 0.03728206 0.6269498 0.7730928

# get cv auc estimates for random forest using nested 
# cross-validation for nuisance parameter estimation. nested
# cross-validation is unfortunately necessary when aggressive learners 
# are used. 
rf_cv_auc_ests <- cv_auc(Y = Y, X = X, K = 5, 
                         learner = "randomforest_wrapper", 
                         nested_cv = TRUE)
rf_cv_auc_ests
#>                est         se       cil       ciu
#> cvtmle   0.7317074 0.03628587 0.6605884 0.8028264
#> onestep  0.7327262 0.03685109 0.6604994 0.8049530
#> esteq    0.7249168 0.03685109 0.6526900 0.7971436
#> standard 0.7351662 0.03603827 0.6645325 0.8057999

# same examples for scrnp
logistic_cv_scrnp_ests <- cv_scrnp(Y = Y, X = X, K = 5, learner = "glm_wrapper")
logistic_cv_scrnp_ests
#>                est         se        cil       ciu
#> cvtmle   0.1200194 0.03234325 0.05662778 0.1834110
#> onestep  0.1106784 0.03230671 0.04735844 0.1739984
#> esteq    0.1106784 0.03230671 0.04735844 0.1739984
#> standard 0.1353408 0.05376489 0.02996354 0.2407180


rf_cv_scrnp_ests <- cv_scrnp(Y = Y, X = X, K = 5, 
                         learner = "randomforest_wrapper", 
                         nested_cv = TRUE)
rf_cv_scrnp_ests
#>                est         se        cil       ciu
#> cvtmle   0.1312718 0.04157305 0.04979013 0.2127535
#> onestep  0.1538148 0.04112673 0.07320788 0.2344217
#> esteq    0.1538148 0.04112673 0.07320788 0.2344217
#> standard 0.2008037 0.03481675 0.13256413 0.2690433
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
