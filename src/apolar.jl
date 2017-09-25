import PolyExp: hankel
export hilbert, perp, weights, solve

function PolyExp.hankel(T, L1::AbstractVector, L2::AbstractVector)
    PolyExp.hankel(series(T), L1, L2)
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

