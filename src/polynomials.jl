export @ring, deg
import DynamicPolynomials: maxdegree, monomials

function buildpolvar(::Type{PV}, arg, var) where PV
    :($(esc(arg)) = $var)
end

"""
```
@ring args...
```
Defines the arguments as variables and output their array.

Example
-------
```
X = @ring x1 x2
```
"""
macro ring(args...)
    X = DynamicPolynomials.PolyVar{true}[DynamicPolynomials.PolyVar{true}(string(arg)) for arg in args]
    V = [buildpolvar(PolyVar{true}, args[i], X[i]) for i in 1:length(X)]
    push!(V, :(TMP = $X) )
    reduce((x,y) -> :($x; $y), :(), V)
end

"""
```
deg(p:Polynomial) -> Int64
```
Degree of a polynomial
"""
function deg(p::Polynomial{B,T}) where {B,T}
    maxdegree(p.x)
end
