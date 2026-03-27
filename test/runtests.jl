
F0 = filter(x -> startswith(x,"test_") && endswith(x, ".jl") && !startswith(x,"runtests"), readdir(pwd()))
println("Testing ", F0)
_E = String[]
T = Float64[]
_s = 0.0
for _f in F0
    try
        local _t = @elapsed include(_f)
        @info "\033[96m$_f\033[0m   $_t(s)"
        global _s += _t
    catch
        @warn "problem with $_f"
        push!(_E,_f)
    end
end

if length(_E)>0
    @warn "problems with $_E"
end

println()
@info "total time: $_s(s)"

