
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
#> cvtmle   0.6967176 0.03520323 0.6277206 0.7657147
#> onestep  0.6975034 0.03564613 0.6276383 0.7673685
#> esteq    0.6878086 0.03564613 0.6179434 0.7576737
#> standard 0.7016291 0.03710328 0.6289080 0.7743502

# get cv auc estimates for random forest using nested 
# cross-validation for nuisance parameter estimation. nested
# cross-validation is unfortunately necessary when aggressive learners 
# are used. 
rf_cv_auc_ests <- cv_auc(Y = Y, X = X, K = 5, 
                         learner = "randomforest_wrapper", 
                         nested_cv = TRUE)
rf_cv_auc_ests
#>                est         se       cil       ciu
#> cvtmle   0.7141555 0.03783280 0.6400045 0.7883064
#> onestep  0.7165059 0.03850217 0.6410430 0.7919688
#> esteq    0.7023838 0.03850217 0.6269210 0.7778467
#> standard 0.7351003 0.03584855 0.6648384 0.8053621

# same examples for scrnp
logistic_cv_scrnp_ests <- cv_scrnp(Y = Y, X = X, K = 5, learner = "glm_wrapper")
logistic_cv_scrnp_ests
#>                est         se        cil       ciu
#> cvtmle   0.1348865 0.02860080 0.07882994 0.1909430
#> onestep  0.1364033 0.02858968 0.08036861 0.1924381
#> esteq    0.1364033 0.02858968 0.08036861 0.1924381
#> standard 0.1508590 0.03365578 0.08489491 0.2168231


rf_cv_scrnp_ests <- cv_scrnp(Y = Y, X = X, K = 5, 
                         learner = "randomforest_wrapper", 
                         nested_cv = TRUE)
rf_cv_scrnp_ests
#>                 est         se        cil       ciu
#> cvtmle   0.09638631 0.03279161 0.03211594 0.1606567
#> onestep  0.09737640 0.03278857 0.03311199 0.1616408
#> esteq    0.09737640 0.03278857 0.03311199 0.1616408
#> standard 0.13018271 0.02878232 0.07377039 0.1865950
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
