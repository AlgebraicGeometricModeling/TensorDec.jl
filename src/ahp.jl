export ahp

using LinearAlgebra
using MultivariatePolynomials
using TensorDec
using DynamicPolynomials

#-----------------------------------------------------------------------
"""
```
ahp(T::symmetric Tensor, X=@polyvar x1...xn)-> 'P' Associated homogeneous polynomial
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

julia> X=@polyvar x1 x2
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
