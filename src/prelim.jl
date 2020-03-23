export gradeval, hessianeval, hpol, op, Delta, solve
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
function Delta(P,W::Vector)
    W0=real(W)
    r=size(W0,1)
    d=maxdegree(P)
    delta1=(1/10)*sqrt((d/r)*sum(W0[i]^2 for i in 1:r))
    delta2=(1/2)*(norm(P))
    delta=min(delta1,delta2)

end

function solve(a::Float64,b::Float64,c::Float64)
    D=b^2-4*a*c
    x1=(-b+sqrt(D))/(2*a)
    x2=(-b-sqrt(D))/(2*a)
    if x1>=1 && x1<=2
        x=x1
    else
        x=x2
    end
    x
end
