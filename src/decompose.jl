import PolyExp: svd_decompose
export svd_decompose

#------------------------------------------------------------------------
"""
```
svd_decompose(p :: Polynomial{true,T},  x, eps :: Float64 =1.e-6)
```
Decompose the polynomial ``pol`` as ``∑ ωi (ξi1 x1 + ... + ξin xn)ᵈ `` where 
``d`` is the degree of ``p``.

The variable ``x`` is the variable, which is substituted by 1 for the affine decomposition.

The optional argument ``eps`` is the precision used to compute the numerical rank.
Its default value is ``1.e-6``.
"""
function PolyExp.svd_decompose{T}(pol :: Polynomial{true,T},  x, eps :: Float64 = 1.e-6)
    d  = deg(pol)
    sigma = series(pol, x, d)
    X = variables(pol)
    v = 0
    for i in 1:length(X)
        if X[i] == x
            v=i; break
        end
    end
    w, Xi = svd_decompose(sigma, eps)
    homog(w,Xi,d,v)
end

#------------------------------------------------------------------------
"""
```
svd_decompose(p :: Polynomial{true,T}, x, r::Int64)
```
Decompose the polynomial ``pol`` as ``∑ ωi (ξi1 x1 + ... + ξin xn)ᵈ `` where 
``d`` is the degree of ``p`` assuming the rank is ``r``.

The variable ``x`` is the variable, which is substituted by 1 for the affine decomposition.
"""
function PolyExp.svd_decompose{T}(pol :: Polynomial{true,T}, x, r :: Int64)
      d  = deg(pol)
    sigma = series(pol, x, d)
    X = variables(pol)
    v = 0
    for i in 1:length(X)
        if X[i] == x
            v=i; break
        end
    end
    w, Xi = svd_decompose(sigma, r)
    homog(w,Xi,d,v)
end


#------------------------------------------------------------------------
# function weights(T, Xi)
#     X = variables(T)
#     d = deg(T)
#     L = monomials(X,d)
#     I = idx(L)
#     A = fill(0.0, length(L), size(Xi,1))
#     for i in 1:size(Xi,1)
#         p = (sum(Xi[i,j]*X[j] for j in 1:length(X)))^d
#         for t in p
#             j = get(I,t.x,0)
#             if j != 0
#                 A[j,i] = t.α/PolyExp.binom(d,exponent(t.x))
#             end
#         end
#     end
#     b = [t[2] for t in terms(T)]
#     A\b
# end

function normalize(M,i)
    diagm([1/M[j,i] for j in 1:size(M,1)])*M
end
