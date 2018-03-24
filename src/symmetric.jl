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
Compute ``∑ wᵢ (ξi1 X₁ + ... + ξin Xₙ)ᵈ``.
"""
function tensor(w::Vector{T}, Xi::Matrix{U}, X, d) where {T,U}
    r = length(w)
    p = sum( w[i]*(sum(Xi[j,i]*X[j] for j in 1:length(X)))^d for i in 1:r)
end
