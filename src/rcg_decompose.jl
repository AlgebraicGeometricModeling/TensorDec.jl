export rcg_decompose
import MultivariateSeries: decompose
import LinearAlgebra: diagm
using MultivariatePolynomials
using DynamicPolynomials
#------------------------------------------------------------------------
function power_vec(d, L, pt)
 [m(pt)*binomial(d,exponents(m)) for m in L]
end

#------------------------------------------------------------------------
"""
```
rcg_decompose(p :: Polynomial{true,T},  rkf :: Function)
```
Decompose the homogeneous polynomial ``p`` as ``∑ ω_i (ξ_{i1} x_1 + ... + ξ_{in} x_n)ᵈ `` where ``d`` is the degree of ``p``.

The optional argument `rkf` is the rank function used to determine the numerical rank from the vector S of singular values. Its default value `eps_rkf(1.e-6)` determines the rank as the first i s.t. S[i+1]/S[i]< 1.e-6 where S is the vector of singular values.

If the rank function `cst_rkf(r)` is used, the SVD is truncated at rank r.

A Riemannian conjugate gradient algorithm is used (RCG) in the algorithm decompose (rcg_decompose) to approximate the pencil of submatrices of the Hankel matrix by a pencil of real simultaneous diagonalizable matrices.
"""
function rcg_decompose(pol::Polynomial{true,C}, rkf::Function=eps_rkf(1.e-6), lbd = :Random  ) where C
    d  = deg(pol)
    X = variables(pol)
    n = length(X)
    sigma = dual(pol,d)

    d0 = div(d-1,2); d1 = d-1-d0
    B0 = monomials(X, d0)
    B1 = monomials(X, d1)

    H = Matrix{C}[]
    for x in X
        push!(H, hankel(sigma, B0, [b*x for b in B1]))
    end

    if lbd == :Random
        lambda = randn(n);
    else
        # Choose a specific lambda from the SVD of M = [H1; ... ;Hn]
        M = cat(H...; dims=2)
        U,S,V = svd(M)
        v = V[:,1]
        s = size(H[1])[2]
        Lambda = zeros(size(M,2),n);
        for i in 1:n
            Lambda[(i-1)*s+1:i*s] = fill(1.0,s)
        end
        lambda = Lambda\v
    end

    lambda /= norm(lambda)


    # Simulatneous Diagonalisation of the pencil
    Xi, Uxi, Vxi = rcg_simdiag(H, lambda, rkf)

    n, r = size(Xi)

    w0 = power_vec(d0,B0,lambda)'*Uxi
    w1 = Vxi*power_vec(d1,B1,lambda)

    w  = [w0[i]*w1[i] for i in 1:r]

    # normalize the vectors Xi
    for i in 1:r
        w[i] *= norm(Xi[:,i])^d;
        Xi[:,i] /= norm(Xi[:,i])
    end

    return w, Xi
end


function rcg_decompose(pol::Polynomial{true,C}, r::Int64) where {C}
    return rcg_decompose(pol, cst_rkf(r))
end
