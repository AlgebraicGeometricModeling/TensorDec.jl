using LinearAlgebra
using MultivariatePolynomials
using DynamicPolynomials

function rqn_tr_ver(P,W,V)
    X=variables(P)
    r=size(W,1)
    n=size(X,1)
    d=maxdegree(P)
    a=zeros((n+1)*r)
    #K'F
    for i in 1:r
        a[(i-1)*(n+1)+1]=sum(W[l]*(V[:,l]'*V[:,i])^d for l in 1:r)-P(V[:,i])
        u=gradeval(P,X,V[:,i])
        b=d*sum(W[l]*(dot(V[:,l],V[:,i]))^(d-1)*V[:,l] for l in 1:r)-u
        a[(i-1)*(n+1)+2:i*(n+1)]=b
    end
#K'K
J=zeros(Polynomial{true,Float64},1,(n+1)*r)
A=zeros((n+1)*r,(n+1)*r)

for i in 1:r
    J[(i-1)*(n+1)+1]=(V[:,i]'*X)^d
    J[(i-1)*(n+1)+2:i*(n+1)]=[d*X[j]*(V[:,i]'*X)^(d-1) for j in 1:n]
end

for i in 1:(n+1)*r
    A[i,i:end]=[apolarpro(J[i],J[j],X) for j in i:(n+1)*r]
end

C=Symmetric(A,:U)
C=convert(Array,C)

#Base
B=zeros((n+1)*r,n*r)

for i in 1:r
    q,s=qr(Matrix(1.0*I, n, n)-V[:,i]*V[:,i]',Val(true))
    T=zeros(n+1,n)
    T[1,1]=1.0
    T[2:end,2:end]=(1/sqrt(d))*q[1:n,1:n-1]
    B[(i-1)*(n+1)+1:i*(n+1),(i-1)*n+1:i*n]=T
end

#Solve GN equation
N=zeros(n*r)
H=B'*C*B
G=B'*a
N=-pinv(H)*G

#Retraction
W1=zeros(r)
V1=zeros(n-1,r)
W2=zeros(r)
V2=zeros(n,r)
for i in 1:r
    W1[i]=N[(i-1)*n+1]
    V1[:,i]=N[(i-1)*n+2:i*n]
end
for i in 1:r
    M=B[(i-1)*(n+1)+2:i*(n+1),(i-1)*n+2:i*n]
    L=M'*X
    tg=sum(V1[l,i]*L[l] for l in 1:n-1)
    Q=(W[i]+W1[i])*(V[:,i]'*X)^d+d*(V[:,i]'*X)^(d-1)*tg
    U=transpose(hankel(Q,1))
    u,s,v=svd(U)
    W2[i]=Q(u[:,1])
    V2[:,i]=u[:,1]
end

return W2, V2

end
#Test
function rand_sym_tens(X, d::Int64)
    L = monomials(X,d)
    c = randn(length(L))
    T = sum(c[i]*L[i]*binomial(d, exponents(L[i])) for i in 1:length(L))
    T/norm_apolar(T)
end
function test1(n,r,d)
X = (@polyvar x[1:n])[1]
V1=rand(n,r)
V2=zeros(n,r)
W1=zeros(r)
W2=zeros(r)
V2=V1+1.e-3*rand(n,r)
for i in 1:r
    W1[i]=norm(V1[:,i])^d
    V1[:,i]=V1[:,i]/norm(V1[:,i])
end

for i in 1:r
    W2[i]=norm(V2[:,i])^d
    V2[:,i]=V2[:,i]/norm(V2[:,i])
end
P1=hpol(W1,V1,X,d)
P2=hpol(W2,V2,X,d)
println("err0:",norm_apolar(P1-P2))

W3,V3=rqn_tr_ver(P2,W1,V1)
P3=hpol(W3,V3,X,d)
println("err1:",norm_apolar(P3-P2))
i=1
while i < 5
    W3,V3=rqn_tr_ver(P2,W3,V3)
    P3=hpol(W3,V3,X,d)
    println("err",i,":",norm_apolar(P3-P2))
i +=1
end
end

function test2(n,r,d)
X = (@polyvar x[1:n])[1]
Pper=rand_sym_tens(X, d)
V=rand(n,r)
W=zeros(r)

for i in 1:r
    W[i]=norm(V[:,i])^d
    V[:,i]=V[:,i]/norm(V[:,i])
end
P=hpol(W,V,X,d)
P1=P+1.e-2*Pper
println("err0:",norm_apolar(P1-P))

W3,V3=rqn_tr_ver(P1,W,V)
P3=hpol(W3,V3,X,d)
println("err1:",norm_apolar(P3-P1))
i=1
while i < 5
    W3,V3=rqn_tr_ver(P1,W3,V3)
    P3=hpol(W3,V3,X,d)
    println("err",i,":",norm_apolar(P3-P1))
i +=1
end
end
