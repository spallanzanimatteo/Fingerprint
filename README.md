# The *fingerprint* method

The *fingerprint* method is a technique to classify bags of mixed-type data measurements.

This repository hosts the MATLAB, R and Python code used to run the experiments whose results are presented in the paper ["A *fingerprint* of a heterogeneous data set"]().
If you use this code in your research project, please cite
```
@article{Spallanzani2021fingerprint,
    author = {Matteo Spallanzani and Gueorgui Mihaylov and Marco Prato and Roberto Fontana},
    title = {A \emph{fingerprint} of a heterogeneous data set},
    journal= {Advances in Data Analysis and Classification},
    volume= {},
    issue = {},
    pages = {},
    year = {2021},
    publisher = {Springer}
}
```


## Installation

To guide you through the installation process, I will use the following conventional symbols:

* `~`, your home directory;
* `/`, filesystem paths separator;
* `$`, shell prompt.

I assume that you will clone the repo into your home directory:

```
$ cd ~
$ git clone https://github.com/spallanzanimatteo/Fingerprint
$ cd Fingerprint
$ git submodule update --init
```


### Create an Anaconda environment and install other third-party software

You can install most of the R and Python dependencies using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html):

```
$ conda env create -f fingerprint.yml
```

After creating the environment, it is time to install third-party packages.
To make them accessible from the newly-created Anaconda environment, be sure to activate it first:

```
$ conda activate fingerprint
```

To install the *mixture composer* (MixtComp) R package, launch an interactive R interpreter and download the package from a CRAN mirror:

```
$ (fingerprint) R
$ > install.packages('RMixtComp')
$ q()
```

To install the *multiple-instance support vector machine* (MI-SVM) Python package, navigate to the `misvm` sub-module and run the Python setup script contained therein:

```
$ (fingerprint) cd Python/misvm
$ (fingerprint) python setup.py install
$ (fingerprint) cd ../..
```


### Verify the installation

To verify that the installation was performed correctly, try to run the experiments *suites*:

```
$ (fingerprint) cd R
$ (fingerprint) Rscript suite.R --data_set=Toy1 --data_type=mixed
$ ...
$ (fingerprint) cd ..
$ (fingerprint) cd Python
$ (fingerprint) python suite.py --data_set=Toy1 --data_type=mixed
$ ...
$ (fingerprint) cd ..
```


### SVM experiments

To run the SVM experiments, you will need [MATLAB&copy;](https://www.mathworks.com/products/matlab.html).
In particular, we used the [R2018b release](https://www.mathworks.com/products/new_products/release2018b.html).

Then, you will need to install two executables in the `MATLAB` folder, named `gpdt` and `classify` (respectively `gpdt.exe` and `classify.exe` on Windows systems).
These executables are the programs performing the training and classification steps of the SVM algorithm.
In particular, the training program benefits from the *gradient projection-based decomposition technique* (GPDT) for numerical optimisation.
You can find archives containing pre-compiled binaries for `gpdt` (for both Windows and UNIX systems) at the [GPDT page](http://dm.unife.it/gpdt/).
You can find archives containing pre-compiled binaries for the `classify` (for both Windows and UNIX systems) at the [SVM light page]().
After extracting the `svm_classify` (`svm_classify.exe` if you are on Windows) binary into the `MATLAB` folder, rename it to `classify` (`classify.exe`).
For compatibility, be sure that you download the GPDT files released at the beginning of 2007, and the version 6.01 of SVM light, that was released in 2004.

**Bonus note**.
In case you feel entitled to an explanation for this peculiar installation procedure, I will outline its reasons.
As a researcher, you might be familiar with that situation where a magical, never-breaking binary is passed from user to user until its source code and compilation process becomes history, from history they become legend, from legend they become myth.
Well, this is more or less what happened with our SVM experiments.
We had these Windows binaries `gpdt.exe` and `classify.exe` that had been compiled from a different version of GPDT than the one published on the GDPT page that we were using over and over.
We decided to use them also for our experiments, but when we got to the preparation of the code release we realised that we could not really allow users to reproduce our experiments on non-Windows machines.
I took a couple of days to reconstruct the process, and it seems that the combination outlined above (the one using publicly-available versions of GPDT and SVM light) can produce the same results that we report in our paper.
In case your machine is running Windows, we also ship the magical binaries with the current repository.


### Compatibility remarks

We developed this project on a Windows system.
I suppose that the installation procedure described in this section could work on UNIX systems as well, but since I have not yet found the time to test it thoroughly, I can not guarantee it.
Therefore, the least that I can do is helping you in case you encounter issues when trying to install the repository on UNIX systems: feel free to write me an email.


## Notice

### Licensing information

The code in the current repository is released under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).
If you plan to integrate this code into your products, remember to consider also the licenses under which the dependencies are released.
In particular:

* `RMixtComp` is released under the [GNU Affero General Public License, version 3 (AGPLv3)](https://github.com/modal-inria/MixtComp/blob/master/LICENCE.md);
* `misvm` is released under the [3-Clause BSD License](https://github.com/garydoranjr/misvm/blob/master/LICENSE);
* GPDT is released under the [GPU General Public License, version 2 (GNU GPLv2)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html);
* to probe the possibility of including SVM light in commercial products, you should contact its author at <a href="mailto:svm-light@ls8.cs.uni-dortmund.de"><svm-light@ls8.cs.uni-dortmund.de><a>;
* MATLAB&copy; is proprietary software licensed by [MathWorks, Inc.](https://www.mathworks.com/)


### Author

Matteo Spallanzani <a href="mailto:spmatteo@ethz.ch"><spmatteo@ethz.ch></a>