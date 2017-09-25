export series, tensor, homog, svd_decompose

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
# function tensor(w, Xi, X, d)
#   r = length(w);
#   t1 = sum( w[i]*(sum(Xi[i,j]*X[j] for j in 1:length(X)))^d for i in 1:r)
# end

function tensor{T,U}(w::Vector{T}, Xi::Matrix{U}, X, d)
    r = length(w)
    p = sum( w[i]*(sum(Xi[i,j]*X[j] for j in 1:length(X)))^d for i in 1:r)
end

function homog(w, Xi, d, v = 1)
    Xi0 = cat(2,Xi[:,1:v-1], fill(one(w[1]), size(Xi,1)), Xi[:,v:size(Xi,2)])
    w0 = copy(w)
    for i in 1:size(Xi0,1)
        n = norm(Xi0[i,:])
        for j in 1:size(Xi0,2)
            Xi0[i,j] /=n
        end
         w0[i] *= n^d
    end
    w0, Xi0
end
