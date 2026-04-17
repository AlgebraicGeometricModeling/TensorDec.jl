module TensorDec

using LinearAlgebra
using DynamicPolynomials
using AlgebraicSolvers

include("symmetric.jl")
include("multilinear.jl")
include("apolar.jl")
include("extension.jl")
include("decompose.jl")
include("rcg_decompose.jl")
include("simdiag.jl")
include("rcg_simdiag.jl")
include("approximate.jl")
include("moments.jl")
include("gad_decompose.jl")

end # module
