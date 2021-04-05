include("../src/TR_RNS.jl")
"""
test_C_SHED(r,d,X,eps)
Take a complex symmetric decomposition of rank r of order d and dimension n=size(X,1) where X=@ring x1...xn.
Make a noise of order eps 'P1'.
Apply sym_SHED_iter on P1 to find a rank r approximation.

"""
function test_C_SHED(r,d,X,eps)
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
    W1,V1=sym_SHED_iter(T1,r)
end
"""
test_C_C(r,d,X,eps)
Take a complex symmetric decomposition of rank r of order d and dimension n=size(X,1) where X=@ring x1...xn.
Make a noise of order eps 'P1'.
Apply sym_C_iter on P1 to find a rank r approximation.

"""
function test_C_C(r,d,X,eps)
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
    W1,V1=sym_C_iter(T1,r)
end
"""
test_R_SHED(r,d,X,eps)
Take a real symmetric decomposition of rank r of order d and dimension n=size(X,1) where X=@ring x1...xn.
Make a noise of order eps 'P1'.
Apply sym_SHED_iter on P1 to find a rank r approximation.

"""
function test_R_SHED(r,d,X,eps)
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
    W1,V1=sym_SHED_iter(T1,r)
end
"""
test_R_R(r,d,X,eps)
Take a real symmetric decomposition of rank r of order d and dimension n=size(X,1) where X=@ring x1...xn.
Make a noise of order eps 'P1'.
Apply sym_R_iter on P1 to find a rank r approximation.

"""
function test_R_R(r,d,X,eps)
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
    W1,V1=sym_R_iter(T1,r)
end
"""
test_R_C(r,d,X,eps)
Take a real symmetric decomposition of rank r of order d and dimension n=size(X,1) where X=@ring x1...xn.
Make a noise of order eps 'P1'.
Apply sym_C_iter on P1 to find a rank r approximation.

"""
function test_R_C(r,d,X,eps)
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
    W1,V1=sym_C_iter(T1,r)
end
