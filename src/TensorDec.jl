module TensorDec

using LinearAlgebra
using MultivariatePolynomials
using DynamicPolynomials

include("polynomials.jl")
include("series.jl")
include("moments.jl")
include("hankel.jl")
include("newton.jl")
include("symmetric.jl")
include("multilinear.jl")
include("apolar.jl")
include("decompose.jl")
include("apolar_prod.jl")
include("ahp.jl")
include("prelim.jl")
include("RNS.jl")
include("TR_RNS.jl")
end

# using LinearAlgebra
# using DynamicPolynomials
# using MultivariatePolynomials
