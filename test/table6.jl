using DynamicPolynomials, LinearAlgebra
include("sym_tens_fct.jl")
include("../src/RNE_N_TR.jl")
include("../src/RGN_V_TR.jl")
include("../src/apolar.jl")
include("../src/decompose.jl")
include("../src/multilinear.jl")
include("../src/prelim.jl")

function scale2(r,s)
    n=7
    d=3
    A=randn(n,r);
    for i in 1:r
        A[:,i]/=norm(A[:,i])
    end
    for i in 1:r
       A[:,i]=10^((i*s)/(3*r))*A[:,i]
    end
    X = (@polyvar x[1:n])[1]
    T=sum((A[:,i]'*X)^d for i in 1:r)
    T=T/norm_apolar(T)
    T1=rand_sym_tens(X,d)
    Tp=T+T1*1.e-5
    Tp+=0.0im*X[1]
    d1=Float64[]
    d2=Float64[]
    N1=Float64[]
    ts1=Float64[]
    N2=Float64[]
    ts2=Float64[]
    m1=0
    m2=0
    for i in 1:20
    V=randn(n,r);
    w=ones(r);
    t1 = @elapsed w1,V1, Info = rne_n_tr(Tp, w, V, Dict{String,Any}("maxIter" => 500,"epsIter" => 1.e-3));
    push!(N1,Info["nIter"])
    push!(ts1,t1)
    T2 = tensor(w1,V1,X,d);
    err1=norm_apolar(T-T2);
    if err1<=1.1*1.e-5
        m1+=1
    end
    push!(d1,err1)

    C=opt(w,V,Tp);
    for j in 1:r
        w[j]=w[j]*C[j];
    end
    w+=fill(0.0+0.0im,r);
    V+=fill(0.0+0.0im,n,r);
    for j in 1:r
    V[:,j]=(w[j])^(1/d)*V[:,j]
    end
    t2 = @elapsed V2, Info = rgn_v_tr(Tp, V, Dict{String,Any}("maxIter" => 500,"epsIter" => 1.e-3))
    push!(N2,Info["nIter"])
    push!(ts2,t2)
    T3 = sum(((transpose(V2[:,i]))*X)^d for i in 1:r);
    err2=norm_apolar(T-T3);
    if err2<=1.1*1.e-5
        m2+=1
    end
    push!(d2,err2)
end

return m1, sum(d1)/length(d1), sum(ts1)/length(ts1), sum(N1)/length(N1),m2, sum(d2)/length(d2), sum(ts2)/length(ts2), sum(N2)/length(N2)
end
