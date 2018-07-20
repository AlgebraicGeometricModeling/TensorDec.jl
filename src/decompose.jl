export decompose, cst_rkf, eps_rkf, weights, normlz

#------------------------------------------------------------------------
eps_rkf = eps::Float64 -> function (S)
  i :: Int = 1;
  while i<= length(S) && S[i]/S[1] > eps
    i+= 1;
  end
  i-1;
end

cst_rkf = r::Int64 -> function (S) return r end


# Decomposition of the pencil of matrices
function decompose(H::Vector{Matrix{C}}, rkf::Function ) where C

    U, S, V = svd(H[1])       # H0= U*diagm(S)*V'
    r = rkf(S)

    Sr = S[1:r]
    Sinv = diagm([one(C)/S[i] for i in 1:r])

    M = []
    for i in 2:length(H)
    	push!(M, Sinv*(ctranspose(U[:,1:r])*H[i]*V[:,1:r]))
    end

    n  = length(M)
    M0 = sum(M[i]*rand(Float64) for i in 1:n)

    E = eigvecs(M0)

    Xi = fill(zero(E[1,1]),n+1,r)
    for i in 1:r
        Xi[1,i]=1
    end
    for i in 1:r
    	for j in 1:n
	    Xi[j+1,i] = (E[:,i]\(M[j]*E[:,i]))[1]
	end
    end

    X = (U[:,1:r].* Sr')*E
    Y = (E \ V[:,1:r]')'

    return Xi, X, Y
end

#------------------------------------------------------------------------
"""
```
decompose(p :: Polynomial{true,T},  rkf :: Function)
```
Decompose the homogeneous polynomial ``p`` as ``∑ ω_i (ξ_{i1} x_1 + ... + ξ_{in} x_n)ᵈ `` where ``d`` is the degree of ``p``.

The optional argument `rkf` is the rank function used to determine the numerical rank from the vector S of singular values. Its default value `eps_rkf(1.e-6)` determines the rank as the first i s.t. S[i+1]/S[i]< 1.e-6 where S is the vector of singular values.

If the rank function `cst_rkf(r)` is used, the SVD is truncated at rank r.
"""
function decompose(pol::Polynomial{true,C}, rkf::Function=eps_rkf(1.e-6)) where C
    d  = deg(pol)
    X = variables(pol)
    sigma = dual(pol,d)

    B0 = monomials(X, d-1-div(d-1,2))
    B1 = monomials(X, div(d-1,2))

    H = Matrix{C}[]
    for x in X
        push!(H, hankel(sigma, B0, [b*x for b in B1]))
    end
    Xi, X, Y = decompose(H, rkf)
    r = size(Xi,2)
    n = size(Xi,1)
    w = fill(one(C),r)
    for i in 1:r
        #w[i]*= norm(Xi[:,i]);  Xi[:,i] /= norm(Xi[:,i])
        w[i] = Xi[1,i]; Xi[:,i]/= Xi[1,i]
        w[i]*= X[1,i]
        w[i]*= Y[1,i]
    end
    return w, Xi
end
#------------------------------------------------------------------------
"""
```
decompose(T :: Array{C,3},  rkf :: Function)
```
Decompose the multilinear tensor `T` of order 3 as a weighted sum of tensor products of vectors of norm 1.

The optional argument `rkf` is the rank function used to determine the numerical rank from the vector S of singular values. Its default value `eps_rkf(1.e-6)` determines the rank as the first i s.t. S[i+1]/S[i]< 1.e-6 where S is the vector of singular values.

If the rank function cst_rkf(r) is used, the SVD is truncated at rank r.

Slices along the mode m=1 of the tensor (i.e. `T[i,:,:]`) are used by default to compute the decomposition. The optional argument `mode = m` can be used to specify the sliced mode.
```
decompose(T, mode=2)
decompose(T, eps_rkf(1.e-10), mode=3)
```
"""

function decompose(T::Array{R,3}, rkf::Function = eps_rkf(1.e-6); mode=1) where R

    H = Matrix{R}[]
    for i in 1:size(T,mode)
        push!(H, slicedim(T,mode,i))
    end

    A, B, C = decompose(H, rkf)
    r = size(A,2)
    w = fill(one(R),r)

    for i in 1:r
        nm = norm(A[:,i])
        A[:,i]./=nm
        w[i] = nm

        nm = norm(B[:,i])
        B[:,i] ./= nm
        w[i] *= nm

        nm = norm(C[:,i])
        C[:,i] ./= nm
        w[i] *= nm
    end

    if mode==1
        return w, A, B, C
    elseif mode==2
        return w, B, A, C
    else
        return w, B, C, A
    end
end

#------------------------------------------------------------------------
"""
```
decompose(σ :: Series{T}, rkf :: Function)
```
Decompose the series ``σ`` as a weighted sum of exponentials.
Return ``ω``, ``Ξ`` where
 - ``ω`` is the vector of weights,
 - ``Ξ`` is the matrix of frequency points, stored per row.
The list of monomials of degree ``\\leq {d-1 \\over 2}`` are used to construct
the Hankel matrix, where ``d`` is the maximal degree of the moments in ``σ``.

The optional argument `rkf` is the rank function used to determine the numerical rank from the vector S of singular values. Its default value `eps_rkf(1.e-6)` determines the rank as the first i s.t. S[i+1]/S[i]< 1.e-6 where S is the vector of singular values.

If the rank function cst_rkf(r) is used, the SVD is truncated at rank r.
"""
function decompose(sigma::Series{R,M}, rkf::Function = eps_rkf(1.e-6)) where {R, M}
    d  = maxdegree(sigma)
    X = variables(sigma)

    B0 = monoms(X, div(d-1,2))
    B1 = monoms(X, div(d-1,2))

    H = Matrix{R}[hankel(sigma, B0, B1)]
    for x in X
        push!(H, hankel(sigma, B0, [b*x for b in B1]))
    end

    Xi, X, Y = decompose(H, rkf)

    r = size(Xi,2)
    w = fill(one(eltype(Xi)),r)

    for i in 1:r
        w[i] = Xi[1,i]
        Xi[:,i]/= Xi[1,i]
        w[i]*= X[1,i]
        w[i]*= Y[1,i]
    end

    return w, Xi[2:end,:]

end

#------------------------------------------------------------------------
"""
```
weights(T, Xi::Matrix) -> Vector
```
Compute the weight vector in the decomposition of the homogeneous polynomial `T`
as a weighted sum of powers of the linear forms associated to the
columns of `Xi`.
"""
function weights(T::Polynomial{true,C}, Xi::Matrix) where C
    X = variables(T)
    d = deg(T)
    L = monomials(X,d)
    I = Dict{Monomial{true},Int64}()
    for (m, i) in zip(L,1:length(L))
        I[m] = i
    end
    A = fill(zero(Xi[1,1]), length(L), size(Xi,2))
    for i in 1:size(Xi,2)
        p = dot(Xi[:,i],X)^d
        for t in p
            j = get(I,t.x,0)
            if j != 0
                A[j,i] = t.α
            end
        end
    end

    b = fill(zero(C), length(L))
    for t in T
        j = get(I,t.x,0)
        if j != 0
            b[j] = t.α
        end
    end
    b = [t.α for t in T]
    A\b
end

function normlz(M,i)
    diagm([1/M[i,j] for j in 1:size(M,1)])*M
end
