using MultivariateSeries

import Base: binomial
export tensor, generic_rank



function Base.binomial(d, alpha::Vector{Int64})
  r = binomial(d, alpha[1])
  for i in 2:length(alpha)
      d -= alpha[i-1]
      r *= binomial(d, alpha[i])
  end
  r
end

"""
    generic_rank(n::Int64,d::Int64; verbose = false)

Genereic rank of a form of degree d in n variables (i.e. in S^d(kk^n)).
In the exceptional cases, the generic rank is `r = Int64(ceil(binomial(n-1+d,d)/n)) + 1`
If verbose = true then there is a warning for these exceptional cases.

"""
function generic_rank(n::Int64, d::Int64; verbose = true)
    if d == 2
        return n
    end
    r = Int64(ceil(binomial(n-1+d,d)/n))
    if (d == 4 && (n in [3,4,5])) || (d==5 && n == 5)
        verbose && @warn "Exception of Alexander-Hirschowitz theorem"
        return r+1
    end
    return r 
end
"""
```
tensor(w, Xi, V, d) -> Polynomial{true,T} 
```
Compute ``∑ wᵢ (ξ_{i,1} V₁ + ... + ξ_{i,n} Vₙ)ᵈ`` where 

- `Xi` is a column-wise matrix of r points, 
- `V`  is a vector of variables,
- `d`  is a degree.

## Example

```
using TensorDec, DynamicPolynomials
X = @polyvar x0 x1 x2
w = rand(5)
Xi = rand(3,5)
tensor(w,Xi,X,4)
```
"""
function tensor(w::AbstractVector{T}, Xi::AbstractMatrix{U}, V, d::Int64) where {T,U}
    r = length(w)
    p = sum( w[i]* dot(V,Xi[:,i])^d for i in 1:r)
end
"""
```
tensor(w, Xi, V, d) -> MultivariatePolynomial
```
Compute ``∑ wᵢ Π_j(ξ_{i,j,1} V[j][1] + ... + ξ_{i,j,n_j} V[j][n_j])^d[j]`` where 
- `Xi` is a vector of matrices of points, 
- `V`  is a vector of vectors of variables,
- `d`  is a vector of degrees.

## Example

```
using TensorDec, DynamicPolynomials
X = @polyvar x0 x1 x2
Y = @polyvar y0 y1
w = rand(5)
Xi0 = rand(3,5)
Xi1 = rand(2,5)
tensor(w,[Xi0,Xi1],[X,Y],[4,2])
```
"""
function tensor(w::Vector{T}, Xi::Vector{AbstractMatrix{U}}, V::Vector, d::Vector{Int64}) where {T,U}
    r = length(w)
    sum( w[i]*prod( dot(Xi[j][:,i],V[j])^d[j] for j in 1:length(d)) for i in 1:r)
end


"""
```
tensor(H, L1, L2) -> MultivariatePolynomial
```
Compute the symmetric tensor or homogeneous polynomial which Hankel matrix in the bases L1, L2 is H.
"""
function tensor(H::Array{C,2} , L1::AbstractVector{M}, L2::AbstractVector{M}) where {C, M}
    res = zero(L1[1])
    dict = Dict{M,Bool}()
    d = maxdegree(L1)+ maxdegree(L2)

    i = 1
    for i in 1:length(L1)
        m1=L1[i]
        for j in 1:length(L2)
            m2 = L2[j]
            m = m1*m2
            if !get(dict,m,false)
	        res = res + H[i,j]*m*TensorDec.binomial(d,exponents(m))
                dict[m]=true
	    end
        end
    end
   res
end



function _monomial(X, E)
    prod(X[i]^E[i] for i in 1:length(E))
end

"""
```
tensor(s, X, d) -> MultivariatePolynomial
```
Compute the symmetric tensor or homogeneous polynomial in the variables `X` corresponding to the series s.
The coefficients ``s_{\\alpha}`` are multiplied by ``binomial(d,\\alpha)``. The monomials are homogenised in degree d with respect to the **first variable** of X.

Example:
========
    X = @polyvar x1 x2
    s = dual(1 - x1*x2 + x2^2)
    @polyvar x0
    F = tensor(s, [x0, x1, x2], 3)

gives

    x0³ + 3x2²x0 - 6x1x2x0

"""
function tensor(s::MultivariateSeries.Series{T}, X, d = maxdegree(s)) where {T}

    P = zero(T) #Polynomial{true,T})
    for (m,c) in s
        alpha = exponent(m)
        alpha = cat([d-sum(alpha)],alpha; dims =1)
        P += c*binomial(d, alpha)*_monomial(X,alpha)
    end
    return P
end


import MultivariateSeries:series, hankel

function MultivariateSeries.series(F, X, d::Int64 = maxdegree(F))
    P = zero(F) #Polynomial{true,T})
    for (c,t) in zip(coefficients(F),monomials(F))
        m = exponents(t)
        alpha = m[2:end]
        c = coefficient(t)
        P += c/binomial(d, m)*_monomial(X,alpha)
    end
    return MultivariateSeries.dual(P)
end

function MultivariateSeries.hankel(F, k)
    X = variables(F)
    d = maxdegree(F)
    L0 = reverse(monomials(X,d-k))
    L1 = reverse(monomials(X,k))
    s = MultivariateSeries.dual(F,d)
    MultivariateSeries.hankel(s,L0,L1)
end
