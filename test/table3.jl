using DynamicPolynomials, TensorDec, MultivariateSeries, JuliaDB

include("sym_tens_fct.jl")
include("spm.jl")
include("../src/RNE_N_TR.jl")
include("../src/RGN_V_TR.jl")
include("../src/apolar.jl")
include("../src/decompose.jl")
include("../src/multilinear.jl")
include("../src/prelim.jl")
function tst_table3(r, eps)
    n=10
    d=4
    N=100
    d1=Float64[]
    d2=Float64[]
    d3=Float64[]
    ts=Float64[]
    ts1=Float64[]
    N1=Float64[]
    N2=Float64[]

    for i in 1:N
        X = (@polyvar x[1:n])[1]
        w0 = fill(one(Float64),r)
        V0 = randn(Float64,n,r)
        T0 = tensor(w0, V0, X, d)
        T   = T0 + rand_sym_tens(X,d)*eps
        t1 = @elapsed w1, V1, Info = decompose(T,r)
        T1 = tensor(w1, V1, X, d)
        err1=norm_apolar(T1-T0)
        push!(d1,err1/eps)
        t2 = @elapsed w2, V2, Info = rne_n_tr(T, w1, V1)
        push!(N1,Info["nIter"])
        push!(ts,t1+t2)
        T2 = tensor(w2, V2, X, d)
        err2=norm_apolar(T2-T0)
        push!(d2,err2/eps)


        v0 = randn(n);
        t3=@elapsed w3, V3, Info = spm_decompose(T, r, v0,
                                     Dict{String,Any}("maxIter" => 400,
                                                      "epsIter" => 1.e-10))
    push!(N2,sum(Info["nIter"]))
    push!(ts1,t3)
    T3 =tensor(w3, V3, X, d)
    err3=norm_apolar(T3-T0)
    push!(d3,err3/eps)


        c=typeof(V1[1,1])
        V11=fill(zero(c), n, r)
        w1+=fill(0.0+0.0im,r)
        V1+=fill(0.0+0.0im,n,r)
        T+=0.0im*X[1]
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
        t4=@elapsed V3, Info = rgn_v_tr(T,V11)
        push!(N3,sum(Info["nIter"]))
        push!(ts2,t1+t4)
        T4 = sum((transpose(V3[:,i])*X)^d for i in 1:r)
        err4=norm_apolar(T4-T0)
        push!(d4,err4/eps)

end

    d1_rne = sort(d2)
    min_rne=d1_sort[1]
    med_rne=d1_sort[div(N,2)]
    max_rne=d1_sort[end]

    d2_sort = sort(d3)
    min_spm = d2_sort[1]
    med_spm = d2_sort[div(N,2)]
    max_spm = d2_sort[end]

    d3_sort = sort(d4)
    min_rgn = d3_sort[1]
    med_rgn = d3_sort[div(N,2)]
    max_rgn = d3_sort[end]

    return (min_rne+med_rne+max_rne)/3, sum(ts)/length(ts), sum(N1)/length(N1), (min_spm+med_spm+max_spm)/3, sum(ts1)/length(ts1), sum(N2)/length(N2), (min_rgn+med_rgn+max_rgn)/3, sum(ts2)/length(ts2), sum(N3)/length(N3)

end

t = table([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5], [1.e-0,1.e-1, 1.e-2, 1.e-4, 1.e-6,1.e-0,1.e-1, 1.e-2, 1.e-4, 1.e-6,1.e-0,1.e-1, 1.e-2, 1.e-4, 1.e-6,1.e-0,
1.e-1, 1.e-2, 1.e-4, 1.e-6,1.e-0,1.e-1, 1.e-2, 1.e-4, 1.e-6] ; names = [:r, :eps])

z=select(t, (:r, :eps) => row-> tst_table1_rns_spm(row[1],row[2]))

transform(t, :ref_t_N => z)
