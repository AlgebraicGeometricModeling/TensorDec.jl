module TensorDec

info() = "TensorDec", "0.1.0", "https://gitlab.inria.fr/AlgebraicGeometricModeling/TensorDec.jl"

using Reexport
@reexport using MultivariateSeries
using LinearAlgebra
using MultivariatePolynomials
using DynamicPolynomials

#include("polynomials.jl")
#include("series.jl")
#include("moments.jl")
#include("hankel.jl")
include("symmetric.jl")
include("multilinear.jl")
include("apolar.jl")
#include("apolar_prod.jl")
include("decompose.jl")

#include("newton.jl")
include("ahp.jl")
include("prelim.jl")
include("RNS.jl")
include("RNS_TR.jl")
include("rns_tr_mat.jl")
include("RQN_Ver.jl")
include("symr.jl")
include("weierstrass.jl")

end # module
