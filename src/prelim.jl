export gradeval, hessianeval, hpol, op
using LinearAlgebra
using MultivariatePolynomials
using TensorDec
using DynamicPolynomials

function gradeval(F,X,a)
        [DynamicPolynomials.differentiate(F, X[i])(a) for i in 1:length(X)]
end

function hessianeval(F,X,a)
    n=length(a)
    v=fill(0.0+0.0im, (n,n))
    for i in 1:n
        p=DynamicPolynomials.differentiate(F, X[i])
        v[i,:]=transpose([DynamicPolynomials.differentiate(p,X[j])(a) for j in 1:n])
    end
    v
end

function hpol(W,A,X,d)
    P = sum(W[i]*(transpose(A[:,i])*X)^d for i in 1:length(W))
end

function op(W::Vector, V::Matrix,P)
    d=maxdegree(P)
    r=size(W,1)
    A=fill(0.0+0.0im,r,r)
    B=fill(0.0+0.0im,r)
    for i in 1:r
        A[i,:]=[conj(W[i])*W[j]*(dot(V[:,i],V[:,j]))^d for j in 1:r]
    end
    for i in 1:r
        B[i]=conj(W[i])*P(conj(V[:,i]))
    end
    C=A\B
end
