iPyCLES project is a Large Eddy Simulation model (PyCLES) coupled with stable water isotope tracer components, developed by the [Jonathon S. Wright group](https://jonathonwright.github.io/group.html) in the Department of Earth System Science, Tsinghua University. Now the isotopic components adopted under the branch `isotopetracer` with the model version is iPyCLES v1.0.

## Installation of ipycles in Linux (tested with ubuntu, Debian and Centos):
Important system environments needed to be installed:

`$ sudo apt-get install gcc gfortran-8 csh` for Debian based distributions
or 
`$ sudo yum install gcc csh` for Centos based distributions: like Fedora, Alamlinux, etc.

Here the **gfortran version** should be lower than 9 (the lastest version until May, 2021), or some files can't be compiled.

We recommend using [conda](https://docs.conda.io/en/latest/) as the package management system and environment management system for python environment settings. Miniconda can be downloaded using **wget**:

`$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`

then install by:

`$ bash Miniconda3-latest-Linux-x86_64.sh`

**Becareful with the python version**, which can't be higher then 3.8, because cython doesn't support python 3.9 until now (May, 2021).

Install packages needed to compile ipycles:

`$ conda install numpy scipy netcdf4 mpi4py matplotlib cython gcc_linux-64`

Complie ipycles by doing following steps:

1. `$ cd ipycles`

2. `$ python generate_paramters.py`

3. `$ CC=mpicc python setup.py build_ext --inplace`

Besides, if there is no **openmpi** environemnt, you can also try to install 'mpich-mpicc' using conda like:
`$ conda install -c conda-forge mpich-mpicc`
and then run though the complie processes above 

 More details about installation in defferent platforms and erros can be found : [install.rst](https://github.com/huzizhan/ipycles/blob/master/docs/source/install.rst)
## Run test cases
Run test cases of ipycles follows: [running.rst](https://github.com/huzizhan/ipycles/blob/master/docs/source/running.rst)

More information about PyCLES can be found in [pycles](https://github.com/pressel/pycles)