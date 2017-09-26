The package `TensorDec.jl` is for tensor decompositions.

To install the package within julia:

```julia
Pkg.clone("https://gitlab.inria.fr/AlgebraicGeometricModeling/TensorDec.jl.git")
Pkg.build("TensorDec")
```

To use it within julia:

```julia
using TensorDec

X = @ring x0 x1 x2 
n = length(X)
d = 4
r = 4

println("Symmetric tensor: dim ", n, "  degree ",d, "  rank ",r)
Xi = rand(r,n)
w = fill(1.0,r)
T = tensor(w,Xi,X, d)

println("Hilbert fct: ", hilbert(T))
k = 2
H = hankel(T,k)

P = perp(T,k)
```

## Documentation

- More on the package [TensorDec.jl](https://gitlab.inria.fr/AlgebraicGeometricModeling/TensorDec.jl) comming.
- For more information on [Julia](https://julialang.org/)
