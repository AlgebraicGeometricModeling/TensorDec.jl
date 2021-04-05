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

#include("newton.jl")

include("ahp.jl")
include("prelim.jl")
include("RNE_N_TR.jl")
include("RGN_V_TR.jl")
include("symr.jl")



end # module
