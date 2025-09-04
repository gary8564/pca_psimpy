v1.0.0
------
* Removed the `simulator/ravaflow24` module.

* Replaced deprecated `scipy.interpolate.interp2d` with `scipy.interpolate.RegularGridInterpolator` to maintain compatibility with newer versions of SciPy.
  
* Added citation information to the README.

* Added repeated k-fold validation for `emulator/robustgasp` module.

v0.2.1
------
* Set R_HOME automatically for conda environment to ensure seamless installation.

* Updated the installation guide by using conda-forge::r-robustgasp package.

* Improved precision for ScalarGaSP and PPGaSP classes.

v0.2.0
------
* Introduced the `simulator/ravaflow3G` module.

* Updated tests and docs accordingly.

* Removed upper bound version constraints for dependencies. This release requires `Python 3.9` or later.


v0.1.2
------

* Elaborate on installation. Use pip from conda-forge channel to install `PSimPy`

* Add info about how tests can be run

v0.1.1
------

* Add info about installation

* Change np.float128 to float to avoid issues in windows system

* Update metadata

v0.1.0
------

* First release
