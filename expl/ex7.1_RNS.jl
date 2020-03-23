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


W0, V0 = decompose(T0)
T00 = tensor(W0,V0,X,d)  
println("dec T0: ", norm(T0-T00))

W1, V1 = decompose(T1)
T10 = tensor(W1,V1,X,d)  
println("dec T1: ",norm(T1-T10))

w, Xi = decompose_qr(T1,cst_rkf(3))
println("dqr T1: ",norm(T1-tensor(w,Xi,X,d)))

TR_RNS_R(T1,r,w, Xi)
