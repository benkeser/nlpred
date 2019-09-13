## Test environments
* Local: macOS High Sierra, R 3.5.3, 3.6.0
* Travis-CI: Ubuntu 16.04, R 3.6.1
* Appveyor: Windows Server 2012 R2 x64, R 3.6.1
* RHub: Windows Server 2008 R2 SP1, R-devel, 32/64 bit
* RHub: Ubuntu Linux 16.04 LTS, R-release, GCC 
* RHub: Fedora Linux, R-devel, clang, gfortran
* Win-builder: devel and release

## Downstream dependencies
Nothing to report.

## Additional Notes
Re-submission of package, version 1.0.0

### Comments from initial submission
> Please do not use return(NA) but rather stop() with a message f.i. in
	boot_scrnp(). Either adapt the documentation of \value because there is a case
	where the return value is not a list, or adapt the code.

I would prefer to allow these functions sometimes return NA. To correct, I have
moved the functions that can return NA out of `boot_scrnp` and `boot_auc`
and into their own internal functions. I have updated documentation to provide
justification for this. 

> Some code lines in examples are commented out. Please never do that. Ideally
	find toy examples that can be regularly executed and checked. Lengthy examples 
	(> 5 sec), can be wrapped in \donttest{}. f.i.: cv_auc.Rd

Commented code has been removed. Lengthy examples are now wrapped in \donttest{}.

> Please always make sure to reset to user's options, wd or par after you
	changed it in examples and vignettes. 

I could not find anywhere in the R/ directory where I modified par. Perhaps it
was happening in .Rapp.history. I have added this file to .Rbuildignore, which
I hope will resolve the issue. 