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

include("ahp.jl")
include("prelim.jl")
include("RNS.jl")
include("RNS_TR.jl")
include("RN_Mat.jl")
include("RQN_Ver.jl")
include("RQN_Ver_C.jl")
include("symr.jl")

#include("weierstrass.jl")

end # module
