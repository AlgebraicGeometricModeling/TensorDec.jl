export hilbert, perp, apolarpro, norm_apolar
#include("symmetric.jl")
import MultivariateSeries: hankel, dual
"""
```
dual(p::Polynomial, d:: Int64) -> Series{T}
```
Compute the series associated to the tensor p of degree d.
T is the type of the coefficients of the polynomial p.
"""
function MultivariateSeries.dual(p::Polynomial{true,T}, d:: Int64) where T
    s = Series{T, Monomial{true}}()
    for t in p
	s[t.x] = t.α/binomial(d,exponent(t.x))
    end
    return s
end

# """
# ```
# dual(p::Polynomial) -> Series{T}
# ```
# Compute the series associated to the polynomial p, replacing
# the variables xi by its dual dxi. T is the type of the coefficients of the polynomial p.
# """
# function MultivariateSeries.dual(p::Polynomial{true,T}) where T
#     s = Series{T, DynamicPolynomials.Monomial{true}}()
#     for t in p
# 	       s[t.x] = t.α
#     end
#     return s
# end

# function dual(t::Term{true,T}) where T
#     s = Series{T, Monomial{true}}()
#     s[t.x] = t.α
#     return s
# end


function MultivariateSeries.hankel(p::Polynomial{true,T}, L1::AbstractVector, L2::AbstractVector) where T
    hankel(dual(p, deg(p)), L1, L2)
end

"""
Compute the Hankel matrix (a.k.a. Catalecticant matrix) in degree d of the
symmetric tensor F.
"""
function MultivariateSeries.hankel(F::Polynomial{true,T}, d::Int64, X = variables(F)) where T
    L0 = monomials(X, deg(F)-d)
    L1 = monomials(X, d)
    hankel(dual(F,deg(F)),L0,L1)
end

"""
Sequence of dimension of ``S/(F^⟂)`` or of the kernels of the Hankel matrix in degree i
for i in 1:deg(F).
"""
function hilbert(F)
    H = [1]
    for i in 1:deg(F)-1
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
    L1 = monomials(X, deg(F)-d)
    H = hankel(F,L1,L0)
    N = nullspace(H)
    N'*L0
end



#-----------------------------------------------------------------------
"""
```
apolarpro(P,Q) -> ComplexF64
```
The apolar product of two homogeneous polynomials P=∑_{|α|=d} binom{d}{α} p_α x^α and
Q=∑_{|α|=d} binom{d}{α} q_α  x^α, of degree d in n variables is given by
⟨P,Q⟩_d=∑_{|α|=d} binom{d}{α}̅conj(p_α) q_α.


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
function apolarpro(P,Q,X)
    n=size(X,1)
    d=maxdegree(P)
    T=(transpose(ones(n))*X)^d
	t=terms(T)
	s=size(t,1)
    t3=0.0
    for i in 1:s
		b=monomial(t[i])
        a1=coefficient(P,b)
        a2=coefficient(Q,b)
        e=exponent(b)
        p=prod(factorial(e[j]) for j in 1:n)
        alpha=factorial(d)/p
        t3=t3+dot(a1,a2)/alpha
    end
    return t3
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
