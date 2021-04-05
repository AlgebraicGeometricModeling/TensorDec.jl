#FYI: The stopping criterion for the trust rigion's radius is by default 1.e-3.
#Nevertheless, for some examples of table 4 (in particular when the order of perturbation is less than 1.e-3 i.e. 1.e-4, 1.e-6);
#we choose this criterion to be 1.e-7 to get relative errors factors less than 1.

using DynamicPolynomials, TensorDec, MultivariateSeries,JuliaDB

include("sym_tens_fct.jl")
include("../src/RNE_N_TR.jl")
include("../src/RGN_V_TR.jl")
include("../src/apolar.jl")
include("../src/decompose.jl")
include("../src/multilinear.jl")
include("../src/prelim.jl")

function table_4(r,eps)
    n=10
    d=4
    N=100
    #epsiter defines the minimal of the trust region's radius.
    d1=Float64[]
    d2=Float64[]
    t1s=Float64[]
    N1=Float64[]

    d3=Float64[]
    t2s=Float64[]
    N2=Float64[]

    for i in 1:N
        X = (@polyvar x[1:n])[1]
        w0 = fill(one(Complex{Float64}),r)
        V0 = randn(Complex{Float64},n,r)
        T0 = tensor(w0, V0, X, d)
        T   = T0 + rand_sym_tens_C(X,d)*eps
        t1 = @elapsed w1, V1, Info = decompose(T,r)
        T1 = tensor(w1, V1, X, d)
        err1=norm_apolar(T1-T0)
        push!(d1,err1/eps)
        t2 = @elapsed w2, V2, Info = rne_n_tr(T, w1, V1,Dict{String,Any}("maxIter" => 6,
                         "epsIter" => 1.e-3))
        push!(N1,Info["nIter"])
        push!(t1s,t1+t2)
        T2 = tensor(w2, V2, X, d)
        err2=norm_apolar(T2-T0)
        push!(d2,err2/eps)

        V11=zeros(ComplexF64,n,r)
        w11=fill(0.0+0.0im,r)
        C=op(w1,V1,T)
        V11=V1
        for i in 1:r
            w11[i]=w1[i]*C[i]
        end

        P1=tensor(w11,V11,X,d)
        l1=norm_apolar(P1-T)
        if err1<l1
            w11=w1
        end

        for i in 1:r
            V11[:,i]=(w11[i])^(1/d)*V11[:,i]
        end
        t3=@elapsed V3, Info = rgn_v_tr(T,V11,Dict{String,Any}("maxIter" => 6,
                         "epsIter" => 1.e-3))
        push!(N2,sum(Info["nIter"]))
        push!(t2s,t1+t3)
        T3 = sum((transpose(V3[:,i])*X)^d for i in 1:r)
        err3=norm_apolar(T3-T0)
        push!(d3,err3/eps)

    end

    d1_sort = sort(d1)
    min_dc=d1_sort[1]
    med_dc=d1_sort[div(N,2)]
    max_dc=d1_sort[end]

    d2_sort = sort(d2)
    min_rns = d2_sort[1]
    med_rns = d2_sort[div(N,2)]
    max_rns = d2_sort[end]

    d3_sort = sort(d3)
    min_rgn = d3_sort[1]
    med_rgn = d3_sort[div(N,2)]
    max_rgn = d3_sort[end]

    return (min_rns+med_rns+max_rns)/3, sum(t1s)/length(t1s), sum(N1)/length(N1), (min_rgn+med_rgn+max_rgn)/3, sum(t2s)/length(t2s), sum(N2)/length(N2)

end

function table_5(n,d,r,eps,N)
#epsiter defines the minimal of the trust region's radius.
    d1=Float64[]
    d2=Float64[]
    t1s=Float64[]
    N1=Float64[]

    d3=Float64[]
    t2s=Float64[]
    N2=Float64[]

    for i in 1:N
        X = (@polyvar x[1:n])[1]
        w0 = fill(one(Complex{Float64}),r)
        V0 = randn(Complex{Float64},n,r)
        T0 = tensor(w0, V0, X, d)
        T   = T0 + rand_sym_tens_C(X,d)*eps
        t1 = @elapsed w1, V1, Info = decompose(T,r)
        T1 = tensor(w1, V1, X, d)
        err1=norm_apolar(T1-T0)
        push!(d1,err1/eps)
        t2 = @elapsed w2, V2, Info = rne_n_tr(T, w1, V1,Dict{String,Any}("maxIter" => 6,
                         "epsIter" => 1.e-3))
        push!(N1,Info["nIter"])
        push!(t1s,t1+t2)
        T2 = tensor(w2, V2, X, d)
        err2=norm_apolar(T2-T0)
        push!(d2,err2/eps)

        V11=zeros(ComplexF64,n,r)
        w11=fill(0.0+0.0im,r)
        C=op(w1,V1,T)
        V11=V1
        for i in 1:r
            w11[i]=w1[i]*C[i]
        end

        P1=tensor(w11,V11,X,d)
        l1=norm_apolar(P1-T)
        if err1<l1
            w11=w1
        end

        for i in 1:r
            V11[:,i]=(w11[i])^(1/d)*V11[:,i]
        end
        t3=@elapsed V3, Info = rgn_v_tr(T,V11,Dict{String,Any}("maxIter" => 6,
                         "epsIter" => 1.e-3))
        push!(N2,sum(Info["nIter"]))
        push!(t2s,t1+t3)
        T3 = sum((transpose(V3[:,i])*X)^d for i in 1:r)
        err3=norm_apolar(T3-T0)
        push!(d3,err3/eps)

    end

    d1_sort = sort(d1)
    min_dc=d1_sort[1]
    med_dc=d1_sort[div(N,2)]
    max_dc=d1_sort[end]

    d2_sort = sort(d2)
    min_rns = d2_sort[1]
    med_rns = d2_sort[div(N,2)]
    max_rns = d2_sort[end]

    d3_sort = sort(d3)
    min_rgn = d3_sort[1]
    med_rgn = d3_sort[div(N,2)]
    max_rgn = d3_sort[end]

    return min_rns, max_rns, sum(t1s)/length(t1s), sum(N1)/length(N1), min_rgn, max_rgn), sum(t2s)/length(t2s), sum(N2)/length(N2)

end
