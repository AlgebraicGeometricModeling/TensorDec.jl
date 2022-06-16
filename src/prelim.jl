using LinearAlgebra
using MultivariatePolynomials
using DynamicPolynomials

export ahp
export gradeval, hessianeval, hpol, op, Delta, solve

#-----------------------------------------------------------------------
"""
```
ahp(T::symmetric Tensor, X=@ring x1...xn)-> 'P' Associated homogeneous polynomial
```
The associated homogeneous polynomials of degree d in n variables of a symmetric tensor of order d and dimension n.


Example
-------
```jldoctest
julia> n=2
2

julia> d=3
3

julia> T
2×2×2 Array{Float64,3}:
[:, :, 1] =
 -3.0  -1.5
 -1.5   0.0

[:, :, 2] =
 -1.5  0.0
  0.0  1.5

julia> X=@ring x1 x2
2-element Array{PolyVar{true},1}:
 x1
 x2

 julia> P=ahp(T,X)
 (-3.0 + 0.0im)x1³ + (-4.5 + 0.0im)x1²x2 + (1.5 + 0.0im)x2³
```
"""
function ahp(T::Array,X)
    n=size(X,1)
    v=size(T)
    d=size(v,1)
    S=(sum(X[i] for i in 1:n))^d
    t=terms(S)
    s=size(t,1)
    t1=fill((0.0+0.0im)*X[1],s)
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



"""
```
gradeval(F,X,a) ⤍ Vector
```
Compute the evaluation of the gradient vector of the polynomial F with n variables X=@ring x1...xn (i.e. (∂F/∂xi)_{1≤i≤n}) at vector 'a'.

```
"""
function gradeval(F,X,a)
    [DynamicPolynomials.differentiate(F, X[i])(a) for i in 1:length(X)]
end
"""
```
hessianeval(F,X,a) ⤍ Matrix
```
Compute the evaluation of the Hessian matrix of the polynomial F with n variables X=@ring x1...xn (i.e.(∂^2(F)/∂xi∂xj)_{1≤i,j≤n}) at vector 'a'.

```
"""

function hessianeval(F,X,a)
    n=length(a)
    v=fill(0.0+0.0im, (n,n))
    for i in 1:n
        p=DynamicPolynomials.differentiate(F, X[i])
        v[i,:]=transpose([DynamicPolynomials.differentiate(p,X[j])(a) for j in 1:n])
    end
    v
end
"""
```
hpol(W,A,X,d) ⤍ Homogeneous polynomial
```
This function gives the homogeneous polynomial associated to the symmetric decomposition W,A.

```
"""
function hpol(W,A,X,d)
    #P = sum(W[i]*(transpose(A[:,i])*X)^d for i in 1:length(W))
    r = length(W)
    P=sum( W[i]*dot(X,A[:,i])^d for i in 1:r)
end
"""
```
op(W,V,P) ⤍ Vector
```
This function solves the linear least square problem: 1/2 min_{α1,...,αr} ||∑αiW[i](V[:,i]'x)^d-P||^2.

```
"""

function op(W::Vector, V::Matrix,P)
    r=size(W,1)
    d=maxdegree(P)
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
"""
```
Delta(P,W) ⤍ Float64
```
This function gives the initial radius of the initial sphere in a trust region method for the symmetric tensor rank approximation problem.

```
"""
function Delta(P,W::Vector)
    W0=real(W)
    r=size(W0,1)
    d=maxdegree(P)
    delta1=(1/10)*sqrt((d/r)*sum(W0[i]^(2) for i in 1:r))
    delta2=(1/2)*(norm_apolar(P))
    delta=min(delta1,delta2)

end

function Delta1(P,V::Matrix)
    r=size(V,2)
    d=maxdegree(P)
    delta1=(1/10)*sqrt((d/r)*sum((norm(V[:,i]))^(2*d) for i in 1:r))
    delta2=(1/2)*(norm_apolar(P))
    delta=min(delta1,delta2)
end

function solve(a::Float64,b::Float64,c::Float64)
    D=b^2-4*a*c
    if D < 0
        D = - D
    end
    x1=(-b+sqrt(D))/(2*a)
    x2=(-b-sqrt(D))/(2*a)
    if x1>=1 && x1<=2
        x=x1
    else
        x=x2
    end
    x
end
