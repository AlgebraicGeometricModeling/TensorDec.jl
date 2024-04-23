using MultivariateSeries

import Base: binomial
export tensor



function Base.binomial(d, alpha::Vector{Int64})
  r = binomial(d, alpha[1])
  for i in 2:length(alpha)
      d -= alpha[i-1]
      r *= binomial(d, alpha[i])
  end
  r
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
    res = zero(DynamicPolynomials.Polynomial{true,C})
    dict = Dict{M,Bool}()
    d = maxdegree(L1)+ maxdegree(L2)

    i = 1
    for m1  in L1
        j=1
        for m2 in L2
            m = m1*m2
            if !get(dict,m,false)
	        res = res + H[i,j]*m*TensorDec.binomial(d,exponents(m))
                dict[m]=true
	    end
            j+=1
        end
        i+=1
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
The coefficients ``s_{\\alpha}`` are multiplied by ``binomial(d,\\alpha)``. The monomials are homogenised in degree d with respect to the **last variable** of X0.
"""
function tensor(s::MultivariateSeries.Series{T}, X, d = maxdegree(s)) where {T}

    P = zero(T) #Polynomial{true,T})
    for (m,c) in s
        alpha = exponent(m)
        alpha = cat(alpha,[d-sum(alpha)]; dims =1)
        P += c*binomial(d, alpha)*_monomial(X,alpha)
    end
    return P
end
