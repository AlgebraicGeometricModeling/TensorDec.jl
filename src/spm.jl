using MultivariateSeries, DynamicPolynomials, LinearAlgebra

function normalize!(Xi::Matrix)
    for i in 1:size(Xi,2)
        Xi[:,i]/= norm(Xi[:,i])
    end
end

function trk_svd(H,r)
    U,S,V = svd(H)
    # println("Svd: ", S)
    return U[:,1:r], S[1:r], V[:,1:r]
end

function proj(Vr, T)
    return Vr*Vr'*T
end

function tensvec(T, L)
    d = maxdegree(T)
    reshape(hankel(dual(T,d),L,[1]), length(L))
end

function val_vec(u, L)
    return [m(u) for m in L]
end

function tensgrad(U, L, X, v)
    d = maxdegree(L)
    T = sum(U[i]*L[i]*binomial(d,exponents(L[i])) for i in 1:length(L))
    #T = dot(L,U)
    G = differentiate(T,X)
    return [G[i](v) for i in 1:length(X)]
end

function spm_iter(Vr, L, X, v)
    vd = val_vec(v,L)
    ud = proj(Vr, vd)

    n = length(X)
    if n>4
        gamma = 0.3*sqrt(n)
    else
        gamma = sqrt((n-1)/(2*n))
    end
    w = tensgrad(ud, L, X, conj(v)) + gamma*v
    w /= norm(w)
    return w
end

function spm_loop!(Ur, Sr, Vr, L,  r, v, X, Info)
    u0 = v/norm(v)

    N   = (haskey(Info,"maxIter") ? Info["maxIter"] : 400)
    eps = (haskey(Info,"epsIter") ? Info["epsIter"] : 1.e-10)

   nIt = Int64[]

    delta  = 1.
    i = 0
    while i<N && delta > eps
        u1 = spm_iter(Vr, L, X, u0)
        delta = norm(u1-u0)
        u0 = u1
        i+=1
    end
    # println("nit: ",i,"   eps: ", delta)
    push!(Info["nIter"], i)
    return conj(u0)
end

function spm_power_vec(d, L, pt)
    [m(pt)*binomial(d,exponents(m)) for m in L]
end


function spm_deflate(U, S, V, L0, L1, u)
    r = length(S)

    ud0 = U'*val_vec(u,L0)
    vd1 = V'*val_vec(conj(u),L1)

    w = 1.0/(transpose(ud0)*diagm([1/sigma for sigma in S])*conj(vd1))[1]

    D1 = diagm(S) - w*ud0*vd1'
    U1, S1, V1 = svd(D1)

    #println("deflate: ", S1)
    return w, U*U1[:,1:r-1], S1[1:r-1], V*V1[:,1:r-1]
end

"""
Decomposition of the tensor T in rank r with the Power Method.
"""
function spm_decompose(T,
                       r::Int64,
                       v0::Vector,
                       Info::Dict{String,Any} = Dict{String,Any}(
                           "maxIter" => 400,
                           "epsIter" => 1.e-10))

    d = maxdegree(T)
    X = variables(T)
    n = length(X)

    L0 = monomials(X, div(d,2))
    L1 = monomials(X, d-div(d,2))

    H0 = hankel(dual(T,d), L0, L1)

    U, S, V = trk_svd(H0,r)

    #initial point
    u0 = v0/norm(v0)

    w  = fill(zero(v0[1]),r)
    Xi = fill(zero(v0[1]),n,r)

    Info["nIter"] = Int64[]
    for k in 1:r
        u = spm_loop!(U, S, V, L1, r, u0, X, Info)
        Xi[:,k] = u
        wu, U, S, V = spm_deflate(U, S, V, L0, L1, u)
        #println("  -> ", " w", k ,": ", wu, "  Xi",k,": ", u)

        w[k] = wu
    end
    return w, Xi, Info
end
