export decompose, decompose_qr, weights, affine
import MultivariateSeries: diagonalization, decompose
import LinearAlgebra: diagm
using DynamicPolynomials

#------------------------------------------------------------------------
function power_vec(d, L, pt)
 [m(pt)*binomial(d,exponents(m)) for m in L]
end

#------------------------------------------------------------------------
"""
```
decompose(p :: DynamicPolynomials.Polynomial,  rkf :: Function)
```
Decompose the homogeneous polynomial ``p`` as ``∑ ω_i (ξ_{i1} x_1 + ... + ξ_{in} x_n)ᵈ `` where ``d`` is the degree of ``p``.

The optional argument `rkf` is the rank function used to determine the numerical rank from the vector S of singular values. Its default value `eps_rkf(1.e-6)` determines the rank as the first i s.t. S[i+1]/S[i]< 1.e-6 where S is the vector of singular values.

If the rank function `cst_rkf(r)` is used, the SVD is truncated at rank r.
"""
function decompose(pol::DynamicPolynomials.Polynomial, rkf::Function=eps_rkf(1.e-6), lbd = :Random  ) 
    d  = maxdegree(pol)
    X = variables(pol)
    n = length(X)
    sigma = dual(pol,d)

    d0 = div(d-1,2); d1 = d-1-d0
    B0 = monomials(X, d0)
    B1 = monomials(X, d1)

    H = Matrix[]
    for x in X
        push!(H, hankel(sigma, B0, [b*x for b in B1]))
    end
    N = length(H)
    
    lambda = randn(N);

    H0 = sum(H[i]*lambda[i] for i in 1:N)

    U, S, V = svd(H0)       # H0= U*diag(S)*V'
    r = rkf(S)

    Sr  = S[1:r]
    Sri = LinearAlgebra.diagm([one(Sr[1])/S[i] for i in 1:r])

    M = Matrix{typeof(Sr[1])}[]
    for i in 1:length(H)
    	push!(M, Sri*(U[:,1:r]')*H[i]*(V[:,1:r]))
    end

    if r > 1
        Xi, E, DiagInfo = diagonalization(M)
    else
        Xi = fill(zero(C),n,1)
        for i in 1:n
            Xi[i,1] = M[i][1,1]
        end
        E  = fill(1.0,1,1)
        DiagInfo = Dict{String,Any}( "case" => "1x1" )
    end

    Uxi = (U[:,1:r].*Sr')*E
    Vxi = (E\ V[:,1:r]')

    w0 = power_vec(d0, B0, lambda)'*Uxi
    w1 = Vxi*power_vec(d1,B1,lambda)

    w  = [w0[i]*w1[i] for i in 1:r]

    # normalize the vectors Xi
    for i in 1:r
        w[i] *= norm(Xi[:,i])^d;
        Xi[:,i] /= norm(Xi[:,i])
    end

    return w, Xi, DiagInfo
end


function decompose(pol::DynamicPolynomials.Polynomial, r::Int64)
    return decompose(pol, cst_rkf(r))
end

function decompose(pol::DynamicPolynomials.Polynomial, eps::Float64)
    return decompose(pol, eps_rkf(eps))
end

#----------------------------------------------------------------------
function dec_mat(pol::DynamicPolynomials.Polynomial, d0 = div(maxdegree(pol)-1,2))
    d     = maxdegree(pol)
    X     = variables(pol)
    sigma = dual(pol,d)

    B0 = monomials(X, d0)
    B1 = monomials(X, d-d0)

    H = hankel(sigma, B0, B1)
    return H, B1
end

function dec_red(H, rkf::Function=MultivariateSeries.eps_rkf(1.e-6))

    U, S, V = svd(H)       # H0= U*diag(S)*V'
    r = rkf(S)
    L = V[:,1:r]'
    return L

end

function dec_qrbasis(D, L, X)

    Idx = Dict{DynamicPolynomials.Monomial{true},Int64}()
    i = 1;
    for m in L
        Idx[m] = i
        i+=1
    end

    L0 = Any[]
    for m in L
        if degree(m,X[1])>0 push!(L0,m) end
    end

    D0 = fill(zero(D[1,1]), size(D,1),length(L0))
    for i in 1:length(L0)
        for j in 1:size(D,1)
            D0[j,i]= D[j,get(Idx,L0[i],0)]
        end
    end

    F = qr(D0,Val(true))

    B = DynamicPolynomials.Monomial{true}[]
    for i in 1:size(D0,1)
        m = copy(L0[F.p[i]])
        m.z[1]-=1
        push!(B, m)
    end

    B, F.Q'*D, Idx
end

function dec_pencil(D, Idx, B, X)
    r = length(B)

    R = []
    for v in X
        H = fill(0.0, r, r )
        for i in 1:length(B)
            m = B[i]
            k = get(Idx, m*v, 0)
            #println(m, " ", m*v, " ", k)
            if k != 0
                H[:,i] = D[:,k]
            end
        end
        push!(R,H)
    end
    R
end

function dec_eigen(M)
    n = length(M)
    r = size(M[1],1)
    I0 = inv(M[1])
    M0 = sum(I0*M[i]*rand(Float64) for i in 2:n);
    E = eigvecs(M0)
    Xi = fill(zero(E[1,1]),n,r)
    for i in 1:r
    	for j in 1:n
	    Xi[j,i] = (E[:,i]\(M[j]*E[:,i]))[1]
	end
    end
    Xi
end

function decompose_qr(pol::DynamicPolynomials.Polynomial, rkf::Function=eps_rkf(1.e-6))

    d = maxdegree(pol)
    X = variables(pol)
    d0 = div(d-1,2)

    H, L = dec_mat(pol, d0)
    D    = dec_red(H, rkf)

    B, Rr, Idx = dec_qrbasis(D, L, X)

    H  = dec_pencil(D, Idx, B, X)

    I0 = inv(H[1])
    for i in 1:length(H) H[i]*= I0 end

    Xi = dec_eigen(H)
    w = weights(pol,Xi);
    w, Xi
end

#------------------------------------------------------------------------
"""
```
decompose(T :: Array{C,3},  rkf :: Function)
```
Decompose the multilinear tensor `T` of order 3 as a weighted sum of tensor products of vectors of norm 1.

The optional argument `rkf` is the rank function used to determine the numerical rank from the vector S of singular values. Its default value `eps_rkf(1.e-6)` determines the rank as the first i s.t. S[i+1]/S[i]< 1.e-6 where S is the vector of singular values.

If the rank function cst_rkf(r) is used, the SVD is truncated at rank r.

Slices along the mode with minimal dimension  (eg. `T[i,:,:]` if mode=1) are used by default to compute the decomposition. The optional argument `mode = m` can be used to specify the sliced mode.

```
decompose(T, mode=2)
decompose(T, eps_rkf(1.e-10), mode=3)
```
"""
function decompose(T::Array{R,3}, rkf::Function = eps_rkf(1.e-6); mode=findmin(size(T))[2]) where R

    H = Matrix{R}[]
    for i in 1:size(T,mode)
        push!(H, selectdim(T,mode,i))
    end

    A, B, C = simdiag(H, [1.0], rkf)
    C = C'

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
decompose(T :: Array{C,3},  r:Int; mode = m)
```
Approximate the multilinear tensor `T` of order 3 as a weighted sum of r tensor products of vectors of norm 1.

The SVD is used to truncate at rank r.

Slices along the mode with minimal dimension  (eg. `T[i,:,:]` if mode=1) are used by default to compute the decomposition. The optional argument `mode = m` can be used to specify the sliced mode.
```
decompose(T, 3, mode=2)
```
"""
function decompose(T::Array{R,3}, r::Int; mode=findmin(size(T))[2]) where R
    decompose(T, cst_rkf(r))
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
function weights(T::DynamicPolynomials.Polynomial, Xi::AbstractMatrix)
    X = variables(T)
    d = maxdegree(T)
    P = Xi'*[X[i] for i in 1:length(X)]
    L = monomials(X,d)
          
    Vdm = (MultivariateSeries.matrixof([ P[i]^d for i in 1:length(P)],L))'
    b   = (MultivariateSeries.matrixof([T],L))[1,:]

    Vdm\b
end

#------------------------------------------------------------------------
function normlz(M,i=1)
   M*diagm(0 => [1/M[i,j] for j in 1:size(M,2)])
end

"""

"""
function affine(w, Xi, d::Int, i0=1 )
    for i in 1:size(Xi,2)
        w[i] *= Xi[i0,i]^d
        Xi[:,i] /= Xi[1,i]
    end
    w, Xi
end
