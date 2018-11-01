export dual, hankel, hilbert, perp


"""
```
dual(p::Polynomial, d:: Int64) -> Series{T}
```
Compute the series associated to the tensor p of degree d.
T is the type of the coefficients of the polynomial p.
"""
function dual(p::Polynomial{true,T}, d:: Int64) where T
    s = Series{T, Monomial{true}}()
    for t in p
	s[t.x] = t.α/binomial(d,exponent(t.x))
    end
    return s
end

"""
```
dual(p::Polynomial) -> Series{T}
```
Compute the series associated to the polynomial p, replacing
the variables xi by its dual dxi. T is the type of the coefficients of the polynomial p.
"""
function dual(p::Polynomial{true,T}) where T
    s = Series{T, DynamicPolynomials.Monomial{true}}()
    for t in p
	       s[t.x] = t.α
    end
    return s
end

function dual(t::Term{true,T}) where T
    s = Series{T, Monomial{true}}()
    s[t.x] = t.α
    return s
end

function hankel(p, L1::AbstractVector, L2::AbstractVector)
    hankel(dual(p, deg(p)), L1, L2)
end

"""
Compute the Hankel matrix (a.k.a. Catalecticant matrix) in degree d of the
symmetric tensor F.
"""
function hankel(F, d::Int64, X = variables(F))
    L0 = monomials(X, deg(F)-d)
    L1 = monomials(X, d)
    hankel(F,L0,L1)
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
