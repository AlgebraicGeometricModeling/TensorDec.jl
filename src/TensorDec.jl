module TensorDec

info() = "TensorDec", "0.1.1", "https://gitlab.inria.fr/AlgebraicGeometricModeling/TensorDec.jl"

using Reexport
@reexport using MultivariateSeries
using LinearAlgebra
using MultivariatePolynomials
using DynamicPolynomials

include("symmetric.jl")
include("multilinear.jl")
include("apolar.jl")
include("decompose.jl")
include("simdiag.jl")
include("approximate.jl")

end # module
