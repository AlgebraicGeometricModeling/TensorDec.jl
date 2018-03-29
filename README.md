The package `TensorDec.jl` is for tensor decompositions.

## Installation

To install the package within julia:

```julia
Pkg.clone("https://gitlab.inria.fr/AlgebraicGeometricModeling/TensorDec.jl.git")
```

## Example

```julia
using TensorDec

X = @ring x0 x1 x2 
n = length(X)
d = 4
r = 4

# Symmetric tensor of degree d and rank r:
Xi = rand(n,r)
w = fill(1.0,r)
F = tensor(w,Xi,X, d)

k = 2
H = hankel(F,k)

P = perp(F,k)

decompose(F)

# Multilinear tensor
A = rand(3,2)
B = rand(3,2)
C = rand(3,2)
w = rand(2)

T = tensor(w,A,B,C)

w0, A0, B0, C0 = decompose(T)

T-tensor(w0,A0,B0,C0)

```

## Documentation

- [LATEST](http://www-sop.inria.fr/members/Bernard.Mourrain/software/TensorDec/index.html)
- For more information on [Julia](https://julialang.org/)
