include("../src/TR_RNS.jl")
function test_pr_CHED(r,d,X,eps)
    n=size(X,1)
    W0=rand(Float64,r)
    V0=rand(ComplexF64,n,r)
    T0=hpol(W0,V0,X,d)
    t=terms(T0)
    s=size(t,1)
    t1=fill((0.0+0.0im)*x1,s)
    for i in 1:s
    c=coefficient(t[i])
    m=monomial(t[i])
    t1[i]=(c+(10.0)^(-eps)*randn(ComplexF64))*m
end
T1=sum(t1[i] for i in 1:s)
println("d*:",norm(T0-T1))
W1,V1=TR_RNS_SPED(T1,r,500)
end

function test_pr_C(r,d,X,eps)
n=size(X,1)
W0=rand(Float64,r)
V0=rand(ComplexF64,n,r)
T0=hpol(W0,V0,X,d)
t=terms(T0)
s=size(t,1)
t1=fill((0.0+0.0im)*x1,s)
for i in 1:s
    c=coefficient(t[i])
    m=monomial(t[i])
    t1[i]=(c+(10.0)^(-eps)*randn(ComplexF64))*m
end
T1=sum(t1[i] for i in 1:s)
println("d*:",norm(T0-T1))
W1,V1=TR_RNS_C(T1,r,500)
end
function test_pr_RSHD(r,d,X,eps)
n=size(X,1)
W0=rand(Float64,r)
V0=rand(Float64,n,r)
T0=hpol(W0,V0,X,d)
t=terms(T0)
s=size(t,1)
t1=fill((0.0+0.0im)*x1,s)
for i in 1:s
    c=coefficient(t[i])
    m=monomial(t[i])
    t1[i]=(c+(10.0)^(-eps)*randn(Float64))*m
end
T1=sum(t1[i] for i in 1:s)
println("d*:",norm(T0-T1))
W1,V1=TR_RNS_SPED(T1,r,500)
end
function test_pr_RR(r,d,X,eps)
n=size(X,1)
W0=rand(Float64,r)
V0=rand(Float64,n,r)
T0=hpol(W0,V0,X,d)
t=terms(T0)
s=size(t,1)
t1=fill((0.0+0.0im)*x1,s)
for i in 1:s
    c=coefficient(t[i])
    m=monomial(t[i])
    t1[i]=(c+(10.0)^(-eps)*randn(Float64))*m
end
T1=sum(t1[i] for i in 1:s)
println("d*:",norm(T0-T1))
W1,V1=TR_RNS_R(T1,r,500)
end

function test_pr_RC(r,d,X,eps)
    n=size(X,1)
W0=rand(Float64,r)
V0=rand(Float64,n,r)
T0=hpol(W0,V0,X,d)
t=terms(T0)
s=size(t,1)
t1=fill((0.0+0.0im)*x1,s)
for i in 1:s
    c=coefficient(t[i])
    m=monomial(t[i])
    t1[i]=(c+(10.0)^(-eps)*randn(Float64))*m
end
T1=sum(t1[i] for i in 1:s)
println("d*:",norm(T0-T1))
W1,V1=TR_RNS_C(T1,r,500)
end

X = @ring x1 x2 x3
r = 3
d = 5
test_pr_RSHD(r, d, X, 3)
