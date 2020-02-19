
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
#> Registered S3 methods overwritten by 'ggplot2':
#>   method         from 
#>   [.quosures     rlang
#>   c.quosures     rlang
#>   print.quosures rlang

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
#> cvtmle   0.8204338 0.02730337 0.7669202 0.8739474
#> onestep  0.8212162 0.02810503 0.7661314 0.8763010
#> esteq    0.8118607 0.02810503 0.7567758 0.8669455
#> standard 0.8110276 0.03052524 0.7511992 0.8708559

# get cv auc estimates for random forest using nested 
# cross-validation for nuisance parameter estimation. nested
# cross-validation is unfortunately necessary when aggressive learners 
# are used. 
rf_cv_auc_ests <- cv_auc(Y = Y, X = X, K = 5, 
                         learner = "randomforest_wrapper", 
                         nested_cv = TRUE)
rf_cv_auc_ests
#>                est         se       cil       ciu
#> cvtmle   0.8001042 0.03250963 0.7363865 0.8638219
#> onestep  0.8007034 0.03289988 0.7362208 0.8651860
#> esteq    0.7959679 0.03289988 0.7314853 0.8604504
#> standard 0.8035088 0.03169928 0.7413793 0.8656382

# same examples for scrnp
logistic_cv_scrnp_ests <- cv_scrnp(Y = Y, X = X, K = 5, learner = "glm_wrapper")
logistic_cv_scrnp_ests
#>                est         se        cil      ciu
#> cvtmle   0.1790416 0.04025296 0.10014728 0.257936
#> onestep  0.1915809 0.04008089 0.11302376 0.270138
#> esteq    0.1915809 0.04008089 0.11302376 0.270138
#> standard 0.2050000 0.06089600 0.08564602 0.324354


rf_cv_scrnp_ests <- cv_scrnp(Y = Y, X = X, K = 5, 
                         learner = "randomforest_wrapper", 
                         nested_cv = TRUE)
rf_cv_scrnp_ests
#>                 est         se         cil       ciu
#> cvtmle   0.16549372 0.05384936  0.05995091 0.2710365
#> onestep  0.08616216 0.05446186 -0.02058113 0.1929055
#> esteq    0.08616216 0.05446186 -0.02058113 0.1929055
#> standard 0.21000000 0.04172416  0.12822216 0.2917778
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

After using the `nlpred` package, please cite the following:

    @Manual{nlpredpackage,
      title = {nlpred: Estimators of Non-Linear Cross-Validated Risks Optimized for Small Samples},
      author = {David Benkeser},
      note = {R package version 1.0.1}
    }
    
    @article{benkeser2019improved,
      year  = {2019},
      author = {Benkeser, David C and Petersen, Maya and van der Laan, Mark J},
      title = {Improved Small-Sample Estimation of Nonlinear Cross-Validated Prediction Metrics},
      journal = {Journal of the American Statistical Association},
      doi = {10.1080/01621459.2019.1668794}
    }

## License

© 2019- David Benkeser

The contents of this repository are distributed under the MIT license.
See below for details:

    The MIT License (MIT)
    
    Copyright (c) 2019- David C. Benkeser
    
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
