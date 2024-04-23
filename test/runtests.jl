
F0 = filter(x -> startswith(x,"test") && endswith(x, ".jl") && !startswith(x,"runtests"), readdir(pwd()))

E = String[]
T = Float64[]
_s = 0.0
for f in F0
    try
        local _t = @elapsed include(f)
        @info "\033[96m$f\033[0m   $t(s)"
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

