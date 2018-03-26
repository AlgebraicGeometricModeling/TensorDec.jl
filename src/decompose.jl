export decompose, cst_rkf, eps_rkf

#------------------------------------------------------------------------
eps_rkf = eps::Float64 -> function (S)
  i :: Int = 1;
  while i<= length(S) && S[i]/S[1] > eps
    i+= 1;
  end
  i-1;
end

cst_rkf = r::Int64 -> function (S) return r end

function decompose(H::Vector{Matrix{C}}, rkf::Function ) where C

    H0 = H[1]
    U, S, V = svd(H0)       #H= U diag(S) V'
    r = rkf(S)
    
    #Un0 = transpose(U[1,1:r])
    #Un1 = V[1,1:r]

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

    for i in 1:r Xi[1,i]=1 end

    #w = E \ Un1
    w = fill(one(E[1,1]),r)
    
    for i in 1:r
    	for j in 1:n
	    Xi[j+1,i] = (E[:,i]\(M[j]*E[:,i]))[1]
	end
        nm = norm(Xi[:,i])
        Xi[:,i]./=nm
        w[i] = nm
    end

    #D = Un0 .* Sr'
    # for i in 1:r
    # 	w[i] *=  (D*E[:,i])[1]
    # end

    X = (U[:,1:r].* Sr')*E
    for i in 1:r
        nm = norm(X[:,i])
        X[:,i] ./= nm
        w[i]*=nm
    end
    
    Y = V[:,1:r]*inv(E')
    for i in 1:r
        nm = norm(Y[:,i])
        Y[:,i] ./= nm
        w[i]*=nm
    end

    return w, Xi, X, Y 
end


#------------------------------------------------------------------------
"""
```
decompose(p :: Polynomial{true,T},  rkf :: Function)
```
Decompose the homogeneous polynomial ``pol`` as ``∑ ωi (ξi1 x1 + ... + ξin xn)ᵈ `` where ``d`` is the degree of ``p``.

The optional argument ``rkf`` is the rank function used to determine the numerical rank. Its default value ``eps_rkf(1.e-6)`` determines the rank as the first i s.t. S[i+1]/S[i]< 1.e-6 where S is the vector of singular values.

If the rank function cst_rkf(r) is used, the SVD is truncated at rank r.
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
    w, Xi, X, Y = decompose(H, rkf)
    return w, Xi
end
#------------------------------------------------------------------------
"""
```
decompose(T :: Array{C,3},  rkf :: Function)
```
Decompose the multilinear tensor of order 3 T  as a weighted sum of tensor products of vectors of norm 1.

The optional argument ``rkf`` is the rank function used to determine the numerical rank. Its default value ``eps_rkf(1.e-6)`` determines the rank as the first i s.t. S[i+1]/S[i]< 1.e-6 where S is the vector of singular values.

If the rank function cst_rkf(r) is used, the SVD is truncated at rank r.
"""

function decompose(T::Array{C,3}, rkf::Function = eps_rkf(1.e-6)) where C
    H = Matrix{C}[]
    for i in 1:size(T,3)
        push!(H,T[i,:,:])
    end
    return decompose(H, rkf)
end

#------------------------------------------------------------------------
function decompose(sigma::Series{C,M}, rkf::Function = eps_rkf(1.e-6)) where {C, M}
    d  = deg(sigma)
    X = variables(sigma)
    
    B0 = monomials(X, d-1-div(d-1,2))
    B1 = monomials(X, div(d-1,2))

    H = Matrix{C}[hankel(sigma, B0, B1)]
    for x in X
        push!(H, hankel(sigma, B0, [b*x for b in B1]))
    end
    H = Matrix{C}[]
    for i in 1:size(T,3)
        push!(H,T[i,:,:])
    end
    w, Xi, X, Y = decompose(H, rkf)
    return w, Xi
    
end


#------------------------------------------------------------------------
function weights(T, Xi)
    X = variables(T)
    d = deg(T)
    L = monomials(X,d)
    I = Dict{Monomial{true},Int64}()
    for (m, i) in zip(L,1:length(L)) I[m] = i end
    A = fill(zero(Xi[1,1]), length(L), size(Xi,1))
    for i in 1:size(Xi,1)
        p = (sum(Xi[i,j]*X[j] for j in 1:length(X)))^d
        for t in p
            j = get(I,t.x,0)
            if j != 0
                A[j,i] = t.α
            end
        end
    end
    b = [t.α for t in T]
    A\b
end

function normalize(M,i)
    diagm([1/M[j,i] for j in 1:size(M,1)])*M
end


