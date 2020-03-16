using TensorDec
using DynamicPolynomials
using MultivariateSeries
using LinearAlgebra

X = @ring x0 x1 x2
r = 3
d = 4
e = 10^-5
n=size(X,1)
W0=rand(Float64,r)
V0=rand(Float64,n,r)
T0=tensor(W0,V0,X,d)

t=terms(T0)
s=size(t,1)

t1 = Any[]
for i in 1:s
    c = coefficient(t[i])
    m = monomial(t[i])
    push!(t1,(c+e*randn(Float64))*m)
end

T1=sum(t1[i] for i in 1:s)
println("d*:",norm(T0-T1))
#W1,V1 = TR_RNS_R(T1,r,500)


function hankel_pencil(pol::Polynomial{true,C}) where C

    d     = MultivariateSeries.deg(pol)
    X     = variables(pol)
    sigma = TensorDec.dual(pol,d)

    B0 = monomials(X, d-1-div(d-1,2))
    B1 = monomials(X, div(d-1,2))

    H = Matrix{C}[]
    for x in X
        push!(H, TensorDec.hankel(sigma, B0, [b*x for b in B1]))
    end
    H
end

function reduce_pencil(H, rkf::Function=TensorDec.eps_rkf(1.e-6))

    U, S, V = svd(H[1])       # H0= U*diag(S)*V'
    r = rkf(S)
    
    Sr = S[1:r]
    Sinv = diagm(0 => [one(H[1][1,1])/S[i] for i in 1:r])

    M = []
    for i in 1:length(H)
    	push!(M, Sinv*conj(U[:,1:r]')*H[i]*V[:,1:r] )
    end
    return M
    
end

function eigen_pencil(M)
    n = length(M)
    M0 = sum(M[i]*rand(Float64) for i in 2:n);
    E = eigvecs(M0)
    Xi = fill(zero(E[1,1]),n,r)
    for i in 1:r
    	for j in 1:n
	    Xi[j,i] = (E[:,i]\(M[j]*E[:,i]))[1]
	end
    end
    Xi
end

    
W0, V0 = TensorDec.decompose(T0)
T00 = tensor(W0,V0,X,d)  
println("dec T0: ",norm(T0-T00))

W1, V1 = TensorDec.decompose(T1)
T10 = tensor(W1,V1,X,d)  
println("dec T1: ",norm(T1-T10))

H = hankel_pencil(T1)
M = reduce_pencil(H)
eigen_pencil(M)
