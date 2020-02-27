export apolarpro


using LinearAlgebra
using MultivariatePolynomials
using TensorDec
using DynamicPolynomials

#-----------------------------------------------------------------------
"""
```
apolarpro(P,Q) -> Float64
```
The apolar product of two homogeneous polynomials P=∑_{|α|=d} binom{d}{α} p_α x^α and
Q=∑_{|α|=d} binom{d}{α} q_α  x^α, of degree d in n variables is given by
⟨P,Q⟩_d=∑_{|α|=d} binom{d}{α}̅p_αq_α.


Example
-------
```jldoctest
julia> X=@ring x1 x2
2-element Array{PolyVar{true},1}:
 x1
 x2

julia> P=x1^2+2*im*x1*x2+x2^2
x1² + (0 + 2im)x1x2 + x2²

julia> Q=2*x1^2+3*x1*x2+6x2^2
2x1² + 3x1x2 + 6x2²

julia> apolarpro(P,Q)
8.0 - 3.0im
```
"""
function apolarpro(P,Q)
    X=variables(P)
    n=size(X,1)
    d=maxdegree(P)
    t1=terms(P)
    t2=terms(Q)
    s=size(t1,1)
    t3=0.0
    for i in 1:s
        a1=coefficient(t1[i])
        a2=coefficient(t2[i])
        b=monomial(t1[i])
        e=exponent(b)
        p=prod(factorial(e[j]) for j in 1:n)
        alpha=factorial(d)/p
        t3=t3+dot(a1,a2)/alpha
    end
    return t3
end
