import PolyExp: hankel, dual, Series
export hilbert, perp, weights, solve


"""                                                                                                                                  ```                                                                                                                                  dual(p::Polynomial, d:: Int64) -> Series{T}                                                                                          ```                                                                                                                                  Compute the series associated to the tensor p of degree d.                                                                           T is the type of the coefficients of the polynomial p.                                                                               """
function PolyExp.dual{T}( p::Polynomial{true,T}, d:: Int64 = deg(p))
    s = Dict{Monomial{true},T}()
    for t in p
	s[t.x] = t.α/binomial(d,exponent(t.x))
    end
    return PolyExp.Series(s)
end

function PolyExp.hankel(T, L1::AbstractVector, L2::AbstractVector)
    PolyExp.hankel(dual(T,deg(T)), L1, L2)
end

function PolyExp.hankel(T, d::Int64, X = variables(T))
    L0 = monomials(X, deg(T)-d)
    L1 = monomials(X, d)
    hankel(T,L0,L1)
end
    
function hilbert(T)
    H = [1]
    for i in 1:deg(T)-1
        N = nullspace(hankel(T,i))
        push!(H,size(N,1)-size(N,2))   
    end
    push!(H,1)
    H
end

function perp(T,d)
    X = variables(T)
    L0 = monomials(X, d)
    L1 = monomials(X, deg(T)-d)
    H = hankel(T,L1,L0)
    N = nullspace(H)
    N'*L0
end

