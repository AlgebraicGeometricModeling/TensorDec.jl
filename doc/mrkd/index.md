# TensorDec

Package for the decomposition of tensors and polynomial-exponential series.

## Introduction

The package `TensorDec.jl`provides tools for the following decomposition problems:

### Symmetric tensor decomposition
For symmetric tensors or multivariate homogeneous polynomials ``\sigma(\mathbf{x}) = \sum_{|\alpha|=d} \sigma_{\alpha} {d \choose \alpha} \mathbf{x}^{\alpha}``, we consider their Waring decomposition:
```math
    \sigma(\mathbf{x}) = \sum_{i=1}^r \omega_i\, (\xi_{i,1} x_1+ \cdots + \xi_{i,n} x_n)^d
```
with `r`minimal.
    
### Multilinear tensor decomposition    
For multilinear tensors, ``\sigma=(\sigma_{i,j,k})\in E_1 \otimes E_2 \otimes E_3``
we consider the decomposition:
```math
    \sigma = \sum_{i=1}^r \omega_i\, U_i^1 \otimes U_i^2 \otimes U_i^3
```    
with ``U_i^j \in E_j `` vectors and `r` minimal.

### Polynomial-exponential decomposition        
For sequences ``(\sigma_{\alpha})_{\alpha} \in \mathbb{K}^{\mathbb{N}^{n}}`` or series 
```math
\sigma(y) = \sum_{\alpha \in \mathbb{K}^{\mathbb{N}^{n}}} \sigma_{\alpha} \frac{y^{\alpha}}{\alpha!}
```
which can be decomposed as polynomial-exponential series 
```math
\sum_{i=1}^r \omega_i(y) e^{\xi_{i,1} y_1+ \cdots + \xi_{i,n} y_n}
```
with polynomials ``\omega_{i}(y)`` and points ``\xi_{i}= (\xi_{i,1}, \ldots, \xi_{i,n})\in \mathbb{K}^{n}``, we compute the weights ``\omega_i`` and the frequencies ``\xi_i``.


These types of decompositions appear in many problems (see [Examples](@ref sec_examples)). 

The package `TensorDec` provides functions to manipulate (truncated) series, to construct truncated Hankel matrices, and to compute such a decomposition from these Hankel matrices.

## [Examples](@id sec_examples)

```@contents
Pages = map(file -> joinpath("expl", file), filter(x ->endswith(x, "md"), readdir("expl")))
```


## Functions and types

```@contents
Pages = map(file -> joinpath("code", file), filter(x ->endswith(x, "md"), readdir("code"))) 
```

## Installation

The package is available at [https://gitlab.inria.fr/AlgebraicGeometricModeling/TensorDec.jl](https://gitlab.inria.fr/AlgebraicGeometricModeling/TensorDec.jl.git).


To install it from Julia:
```julia
Pkg.clone("https://gitlab.inria.fr/AlgebraicGeometricModeling/TensorDec.jl.git")
```
It can then be used as follows:
```julia
using TensorDec
```
See the [Examples](@ref sec_examples) for more details.


## Dependencies

The package `TensorDec` depends on the following packages:

- `DynamicPolynomials` package on multivariate polynomials represented as lists of monomials.
- `MultivariatePolynomials` generic interface package for multivariate polynomials.

        
