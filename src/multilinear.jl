export tensor

"""
```
tensor(w, Xi, V, d) -> Polynomial{true,T} 
```
Compute ``∑ wᵢ \Pro_j(ξ_{i,j,1} X[j][1] + ... + ξ_{i,j,n_j} X[j][n_j])^d[j]``.
"""
function tensor(w::Vector{T}, Xi::Matrix{U}, Xl::Vector, dl::Vector{Int64}) where {T,U}
    r = length(w)
    I = []; s=0;
    for k in 1:length(dl)
        push!(I,s+1:s+length(Xl[k]))
        s+=length(Xl[k])
    end
    sum( w[i]*prod( dot(Xi[i,I[j]],Xl[j])^dl[j] for j in 1:length(dl)) for i in 1:r)
end


function tensor(A::Matrix, B::Matrix, C::Matrix)
    d1 = size(A,1)
    d2 = size(B,1)
    d3 = size(C,1)
    r = size(A,2)
    reshape(sum(A[:,i]*reshape(B[:,i]*C[:,i]',1,d2*d3) for i in 1:r),d1,d2,d3)
end
function tensor(w::Vector, A::Matrix, B::Matrix, C::Matrix)
    d1 = size(A,1)
    d2 = size(B,1)
    d3 = size(C,1)
    r = size(A,2)
    reshape(sum(w[i]*A[:,i]*reshape(B[:,i]*C[:,i]',1,d2*d3) for i in 1:r),d1,d2,d3)
end

function Base.norm(T::Array{C,3}) where C
    n = size(T)
    r = zero(0)
    for i in 1:n[1]
        for j in 1:n[2]
            for k in 1:n[3]
                r += abs(T[i,j,k])
            end
        end
    end
    return sqrt(r)
end
