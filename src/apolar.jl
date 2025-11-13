using MultivariatePolynomials
import MultivariateSeries: hankel, dual

export hilbert, perp, apolar, norm_apolar, dual, catalecticant

#include("symmetric.jl")

"""
```
catalecticant(p::DynamicPolynomials.Polynomial, d1:: Int64, d2::Int64)
```
 Compute the Catalecticant matrix of the homogeneous polynomial `p` in bidegree `(d1, d2)`.
 One should have `maxdegree(p)=d1+d2`.
""" 
function catalecticant(p::DynamicPolynomials.Polynomial, d1:: Int64, d2::Int64)
    MultivariateSeries.hankel(p,d1,d2)
end

function MultivariateSeries.hankel(p::DynamicPolynomials.Polynomial, d1:: Int64, d2::Int64)
    d = maxdegree(p)
    @assert( d == d1+d2)
    X = variables(p)
    L1 = monomials(X,d1)
    L2 = monomials(X,d2)
    hankel(dual(p, maxdegree(p)), L1, L2)
end


function MultivariateSeries.hankel(p::DynamicPolynomials.Polynomial, L1::AbstractVector, L2::AbstractVector) 
    hankel(dual(p, maxdegree(p)), L1, L2)
end
"""
```
hankel(F::DynamicPolynomials.Polynomial, k::Int64)
```
Compute the Hankel matrix (a.k.a. Catalecticant matrix) in degree (d-k,k) of the
symmetric tensor or form F, where d=maxdegree(F).

The rows are indexed by the monomials of degree d-k and the colmuns by the monomials of degree k, sorted in the reverse of the lexicographic order
"""
function MultivariateSeries.hankel(F::DynamicPolynomials.Polynomial, k::Int64, X = variables(F))
    L0 = reverse(monomials(X, maxdegree(F)-k))
    L1 = reverse(monomials(X, k))
    hankel(dual(F,maxdegree(F)),L0,L1)
end

"""
Sequence of dimension of ``S/(F^⟂)`` or of the kernels of the Hankel matrix in degree i
for i in 1:maxdegree(F).
"""
function hilbert(F)
    H = [1]
    for i in 1:maxdegree(F)-1
        N = nullspace(hankel(F,i))
        push!(H,size(N,1)-size(N,2))
    end
    push!(H,1)
    H
end

"""
Compute the kernel of the Hankel matrix in degree d of the symmetric tensor F.
"""
function perp(F,d)
    X = variables(F)
    L0 = monomials(X, d)
    L1 = monomials(X, maxdegree(F)-d)
    H = hankel(F,L1,L0)
    N = nullspace(H)
    N'*L0
end



#-----------------------------------------------------------------------
"""
```
apolar(P,Q) -> ComplexF64
```
The apolar product of two homogeneous polynomials P=∑_{|α|=d} binom{d}{α} p_α x^α and
Q=∑_{|α|=d} binom{d}{α} q_α  x^α, of degree d in n variables is given by
⟨P,Q⟩_d=∑_{|α|=d} binom{d}{α}̅conj(p_α) q_α.


Example
-------
```jldoctest
julia> X= @polyvar x1 x2
2-element Array{PolyVar{true},1}:
 x1
 x2

julia> P=x1^2+2*im*x1*x2+x2^2
x1² + (0 + 2im)x1x2 + x2²

julia> Q=2*x1^2+3*x1*x2+6x2^2
2x1² + 3x1x2 + 6x2²

julia> apolar(P,Q)
8.0 - 3.0im
```
"""
function apolar(P,Q)

    #assert()
    L = monomials(P)
    E = exponents.(L)
    c = coefficients(P)
    d = maxdegree(P)
    return sum( c[i]*conj(coefficient(Q,L[i]))/binomial(d,E[i]) for i in 1:length(L))

end
"""
```
norm_apolar(P) -> Float64
```
Gives the apolar norm of a homogeneous polynomial P=∑_{|α|=d} binom{d}{α} p_α x^α
as: norm_apolar(P)=∑_{|α|=d} binom{d}{α} p_α*̄p_α.



```
"""
function norm_apolar(P)
    X=variables(P)
    n=size(X,1)
    d=maxdegree(P)

    nrm=0.0
    for t in terms(P)
        c=coefficient(t)
        nrm += abs(c)^2/binomial(d,exponents(t))
    end
    return sqrt(nrm)
end
