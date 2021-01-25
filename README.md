# fiberpolytope

A collection of [sage](https://www.sagemath.org/) routines to study fibrations of reflexive polytopes and their F-theroy compactifications. This package originated from work done with M. Larfors, M. Magill and P-K. Oehlmann. The symbolic computations use singular under the hood and the rest utilizes many functions of [LatticePolytope](https://doc.sagemath.org/html/en/reference/discrete_geometry/sage/geometry/lattice_polytope.html) and [PointCollection](https://doc.sagemath.org/html/en/reference/discrete_geometry/sage/geometry/point_collection.html). It contains an implementation of the [algorithm](http://ctp.lns.mit.edu/wati/data/fibers/fibers.jl) by Huang and Taylor for finding 2d subpolytopes in 4d reflexive polytopes. It also contains routines to find and work with enhancement diagrams as presented in [1910.02963](https://arxiv.org/abs/1910.02963).

## Set up

Go to the sage shell

```console
sage -sh
```

and install with pip

```console
pip install git+https://github.com/robin-schneider/fiberpolytope
```

this however might not always work. You can also just go to the [fiberpolytope.sage](https://github.com/robin-schneider/fiberpolytope) file, put it in your working directory, and use:

```python
load('fiberpolytope.sage')
```

that should always work. In particular also when you want to use the package in the cloud with [cocalc](https://cocalc.com/).

## Notation

When finding f, g and Delta of an F-theory compatification it uses results and notation from [1408.4808](https://arxiv.org/abs/1408.4808).

## Bugs and Features

The package probably contains some (many?) bugs. In the end it's just a collection of routines and we didn't come around wrapping up the whole project.

## Tutorial

For a brief tutorial and non exhaustive list of all the functions check the examples folder.
