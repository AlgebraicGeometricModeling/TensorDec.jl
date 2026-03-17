module TensorDec

#info() = "TensorDec", "0.1.1", "https://gitlab.inria.fr/AlgebraicGeometricModeling/TensorDec.jl"

#using Reexport
using LinearAlgebra
#using MultivariatePolynomials
using DynamicPolynomials
using AlgebraicSolvers

include("symmetric.jl")
include("multilinear.jl")
include("apolar.jl")
include("decompose.jl")
include("rcg_decompose.jl")
include("simdiag.jl")
include("rcg_simdiag.jl")
include("approximate.jl")
include("moments.jl")
include("gad_decompose.jl")

end # module
