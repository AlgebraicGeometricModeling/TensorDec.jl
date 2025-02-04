# TensorDec

Package for the decomposition of tensors and polynomial-exponential series.

## Introduction

The package `TensorDec.jl` provides tools for the following decomposition problems:

### Symmetric tensor decomposition
For symmetric tensors or multivariate homogeneous polynomials ``\sigma(\mathbf{x}) = \sum_{|\alpha|=d} \sigma_{\alpha} {d \choose \alpha} \mathbf{x}^{\alpha}``, we consider their Waring decomposition:
```math
    \sigma(\mathbf{x}) = \sum_{i=1}^r \omega_i\, (\xi_{i,1} x_1+ \cdots + \xi_{i,n} x_n)^d
```
with `r` minimal.
    
### Multilinear tensor decomposition    
For multilinear tensors, ``\sigma=(\sigma_{i,j,k})\in E_1 \otimes E_2 \otimes E_3``
we consider the decomposition:
```math
    \sigma = \sum_{i=1}^r \omega_i\, U_i^1 \otimes U_i^2 \otimes U_i^3
```    
with ``U_i^j \in E_j `` vectors and `r` minimal.

    
## [Tutorials](@id sec_examples)

```@contents
Pages = map(file -> joinpath("expl", file), filter(x ->endswith(x, "md"), readdir("expl")))
```


## Manual

```@contents
Pages = map(file -> joinpath("code", file), filter(x ->endswith(x, "md"), readdir("code"))) 
```

## [Installation](@id sec_installation)

The package is available at [https://github.com/AlgebraicGeometricModeling/TensorDec.jl](https://github.com/AlgebraicGeometricModeling/TensorDec.jl.git).


To install it from Julia:
```julia
] add https://github.com/AlgebraicGeometricModeling/TensorDec.jl
```
It can then be used as follows:
```julia
using TensorDec
```
For more details, see the [tutorials](@ref sec_examples).


## Dependencies

The package `TensorDec` depends on the following packages:

- `LinearAlgebra` standard package for linear algebra.
- `DynamicPolynomials` package on multivariate polynomials represented as lists of monomials.
- `MultivariateSeries` for duality on multivariate polynomials.

These packages are installed with `TensorDec`  (see [installation](@ref sec_installation)).

        
