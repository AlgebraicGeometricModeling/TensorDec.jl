F0 = filter(x -> startswith(x,"ex") && endswith(x, ".jl") && !startswith(x,"runexpls"), readdir(pwd()))

E = String[]
T = Float64[]
_s = 0.0
for f in F0
    try
        local _t = @elapsed include(f)
        @info "\033[96m$f\033[0m   $_t(s)"
        global _s += _t
    catch
        @warn "problem with $f"
        push!(E,f)
    end
end

if length(E)>0
    @warn "problems with $E"
end

println()
@info "total time: $_s(s)"

#=

include("ex0_apolar.jl")
include("ex1_symmetric.jl")	
include("ex2_multilinear.jl")	
include("ex3_multiprony.jl")	
include("ex4_moments.jl")	
include("ex5_decompose.jl")
#include("ex6_perproots.jl")
#include("ex7_newton.jl")
#include("ex7.1_RNS.jl")

=#
