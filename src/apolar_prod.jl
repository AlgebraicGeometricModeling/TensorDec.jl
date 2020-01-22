export apolarpro
using LinearAlgebra
using MultivariatePolynomials
using TensorDec
using DynamicPolynomials
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
