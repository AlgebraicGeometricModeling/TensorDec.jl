module TensorDec

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

end

using DynamicPolynomials
using MultivariatePolynomials
