export moment, exp, log, series, moment, sparse_pol, sparse_decompose;


#----------------------------------------------------------------------
function Base.exp(A::Matrix{U}, pt::Vector{T}) where {U,T}
    N = size(A)
    R = fill(zero(eltype(pt)), N[1], N[2])
    for i  in 1:N[1]
        for j in 1:N[2]
            R[i,j] = pt[j]^A[i,j]
        end
    end
    R
end

#----------------------------------------------------------------------
function Base.log(A::Array{U}, pt::Vector{T}) where {U,T}
    N = size(A)
    R = fill(zero(eltype(pt)), N[1], N[2])
    for i  in 1:N[1]
        for j in 1:N[2]
            R[i,j] = real(log(A[i,j])/log(pt[j]))
        end
    end
    R
end

#----------------------------------------------------------------------
"""
```
moment(w::Vector{T}, P::Matrix{T}) -> Vector{Int64} -> T
```
Compute the moment function ``α -> ∑_{i} ω_{i} P_{i}^α`` 
associated to the sequence P of r points of dimension n, which is a matrix 
of size r*n and the weights w.
"""
moment(w, P) = function(α)
  res = 0
  for i in 1:size(P,1)
      m = 1
      for j in 1:length(α)
        m *= P[i,j]^α[j]
      end
      res+=m*w[i];
  end
  res
end

#----------------------------------------------------------------------
"""
```
moment(p::Polynomial, zeta::Vector{T}) -> Vector{Int64} -> T
```
Compute the moment function ``α \\rightarrow p(ζ^α)``.
"""
moment(p::Polynomial, zeta::Vector) =  function(V::Vector{Int})
    U = [zeta[i]^V[i] for i in 1:length(V)]
    p(U)
end
    
#----------------------------------------------------------------------
"""
```
series(w:: Vector{T}, P::Matrix{T}, L::Vector{M}) -> Series{T}
```
Compute the series of the moment sequence ``∑_{i} ω_{i} P_{i}^α`` for ``α \\in L``.
"""
function series(w:: Vector{T}, P, L::Vector{M}) where {T,M}
   series(moment(w,P), L) 
end
    
#----------------------------------------------------------------------
"""
```
series(w:: Vector{T}, P::Matrix{T}, X, d::Int64) -> Series{T}
```
Compute the series of the moment sequence ``∑_i ω_{i} P_{i}^α`` for ``|α| \\leq d``.
"""
function series(w::Vector{T}, P::Matrix{T}, X, d::Int64) where T
    h = moment(w,P)
    L = monoms(X,d)
   series(h,L)
end


#----------------------------------------------------------------------
"""
```
series(p::Polynomial, zeta, X, d::Int64) -> Series{T}
```
Compute the series of moments ``p(ζ^α)`` for ``|α| \\leq d``.
"""
function series(p::Polynomial, zeta, X, d::Int64)
    h = moment(p,zeta)
    L = monoms(X,d)
    series(h,L)
end

#----------------------------------------------------------------------
"""
```
sparse_pol(w, E, X) -> Polynomial{true,T}
```
Compute the polynomial ``∑ ωᵢ X^E[i,:]`` with coefficients ``ωᵢ`` and monomial exponents ``E[i,:]``.
"""
function sparse_pol(w::Vector, E::Matrix{Int64}, X)
    sum(w[j]*prod(X[i]^E[j,i] for i in 1:size(E,2)) for j in 1:length(w))
end



#------------------------------------------------------------------------
function sparse_decompose(f, zeta, X, d:: Int64)
    sigma = series(moment(f,zeta), monoms(X,d))
    w, Xi = decompose(sigma)
    w, log(Xi,zeta)
end
