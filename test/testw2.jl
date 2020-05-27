include("../src/weierstrass.jl")
include("../src/prelim.jl")
using LinearAlgebra
using MultivariatePolynomials
using DynamicPolynomials
W0=randn(ComplexF64,3)
V0=randn(ComplexF64,3,3)
V1=V0+1.e-1*randn(ComplexF64,3,3)
W1=W0+1.e-1*randn(ComplexF64,3)
X=@ring x0 x1 x2
P0=hpol(W0,V0,X,5)
P1=hpol(W1,V1,X,5)
println("d0:", norm(P1-P0))
for i in 1:3
    W0[i]=W0[i]*(V0[1,i])^5
    V0[:,i]=(1/V0[1,i])*V0[:,i]
end
B = DynamicPolynomials.Monomial{true}[1, x1^2, x1*x2]
w = [W0]
A = [V0]
for j in 1:500
    nw, nA =  Weierstrass1(P1, 3, B, w[j], A[j])
    push!(w,nw)
    push!(A,nA)
    R = hpol(nw,nA,X,5)
    println("d",j,": ", norm(P1-R))
end
