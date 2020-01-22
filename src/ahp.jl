export ahp
using LinearAlgebra
using MultivariatePolynomials
using TensorDec
using DynamicPolynomials
function ahp(T::Array,X)
    n=size(X,1)
    v=size(T)
    d=size(v,1)
    S=(sum(X[i] for i in 1:n))^d
    t=terms(S)
    s=size(t,1)
    t1=fill((0.0+0.0im)*x1,s)
    for i in 1:s
        c=coefficient(t[i])
        m=monomial(t[i])
        a=fill(0.0,n)
        for j in 1:n
            a[j]=degree(m,X[j])
        end
         a = convert(Vector{Int64}, a)
        Ids=vcat([fill(k,a[k]) for k in 1:n]...)
        t1[i]=(c*T[Ids...])*m
    end
    P=sum(t1[i] for i in 1:s)

    return P
end
