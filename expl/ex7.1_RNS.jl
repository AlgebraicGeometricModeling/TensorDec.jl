using TensorDec
using DynamicPolynomials
using LinearAlgebra

X = @ring x0 x1 x2
r = 3
d = 5
d0=2
e = 10^-2
n=size(X,1)
W0=randn(Float64,r)
V0=randn(Float64,n,r)
T0=tensor(W0,V0,X,d)

t=terms(T0)
s=size(t,1)

t1 = Any[]
for i in 1:s
    c = coefficient(t[i])
    m = monomial(t[i])
    push!(t1,(c+e*randn(Float64))*m)
end

T1 = sum(t1[i] for i in 1:s)
println("d0e:",norm(T0-T1))
#W01, V01 = TR_RNS_R(T1,r,500)


function hankel_mat(pol::Polynomial{true,C}, d0 = div(maxdegree(pol)-1,2)) where C
    d     = maxdegree(pol)
    X     = variables(pol)
    sigma = dual(pol,d)

    B0 = monomials(X, d0)
    B1 = monomials(X, d-d0)

    H = hankel(sigma, B0, B1)
    return H, B1
end

function hankel_red(H, rkf::Function=MultivariateSeries.eps_rkf(1.e-6))

    U, S, V = svd(H)       # H0= U*diag(S)*V'
    r = rkf(S)
    L = V[:,1:r]'
    return L
    
end

function hankel_qrbasis(D, L)

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
    
    B = Any[]
    for i in 1:size(D0,1)
        m = copy(L0[F.p[i]])
        m.z[1]-=1
        push!(B, m)
    end

    B, F.Q'*D, Idx
end

function hankel_pencil(D, Idx, B, X)
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

function eigen_pencil(M)
    n = length(M)
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

    
W0, V0 = decompose(T0)
T00 = tensor(W0,V0,X,d)  
println("dec T0: ", norm(T0-T00))

W1, V1 = decompose(T1)
T10 = tensor(W1,V1,X,d)  
println("dec T1: ",norm(T1-T10))

w, Xi = decompose_qr(T1,cst_rkf(3))
println("dqr T1: ",norm(T1-tensor(w,Xi,X,d)))

TR_RNS_R(T1,r,w, Xi)
