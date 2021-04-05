using DynamicPolynomials
include("sym_tens_fct.jl")
include("../src/RNE_N_TR.jl")
include("../src/RGN_V_TR.jl")
include("../src/apolar.jl")
include("../src/decompose.jl")
include("../src/multilinear.jl")
include("../src/prelim.jl")
function tst_ex53(r)
N=50
n=10
d=3
X = (@polyvar x[1:n])[1]
T = sum([(i^2)*X[i]^d for i in 1:n]) +
   sum([1.0*X[i]^(d-1) for i in 1:n])*sum([1.0*X[i] for i in 1:n])
    d1=Float64[]
    d2=Float64[]
    d3=Float64[]
    t1s=Float64[]
    t2s=Float64[]
    N1=Float64[]
    N2=Float64[]

    for i in 1:N
         t1 = @elapsed w1, V1, Info = decompose(T,r)

         T1 = tensor(w1, V1, X, d)
         err1=norm_apolar(T1-T)
         push!(d1,err1)
         t2 = @elapsed w2, V2, Info = rne_n_tr(T, w1, V1,Dict{String,Any}("maxIter" => 10,"epsIter" => 1.e-1))
         push!(N1,Info["nIter"])
         push!(t1s,t1+t2)
         T2 = tensor(w2, V2, X, d)
         err2=norm_apolar(T2-T)
         push!(d2,err2)

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
         t3=@elapsed V3, Info = rgn_v_tr(T,V11,Dict{String,Any}("maxIter" => 10,"epsIter" => 1.e-1))
         push!(N2,sum(Info["nIter"]))
         push!(t2s,t1+t3)
         T3 = sum((transpose(V3[:,i])*X)^d for i in 1:r)
         err3=norm_apolar(T3-T)
         push!(d3,err3)

    end



    d2_sort = sort(d2)
    min_rns = d2_sort[1]
    med_rns = d2_sort[div(N,2)]
    max_rns = d2_sort[end]

    d3_sort = sort(d3)
    min_rqn=d3_sort[1]
    med_rqn=d3_sort[div(N,2)]
    max_rqn=d3_sort[end]

    return min_rns, med_rns, max_rns, sum(t1s)/length(t1s), sum(N1)/length(N1), min_rqn, med_rqn, max_rqn, sum(t2s)/length(t2s), sum(N2)/length(N2)

    end

    function tst_ex54(r)
    N=50
    n=10
    d=3
    X = (@polyvar x[1:n])[1]
    T = sum([exp(1.0*sqrt(i)+im*i^2)*(X[i])^d for i in 1:n]) +
        sum([(1.0*im*i/n)*X[i]^(d-1) for i in 1:n])*sum([X[i] for i in 1:n])
        d1=Float64[]
        d2=Float64[]
        d3=Float64[]
        t1s=Float64[]
        t2s=Float64[]
        N1=Float64[]
        N2=Float64[]

        for i in 1:N
             t1 = @elapsed w1, V1, Info = decompose(T,r)
             T1 = tensor(w1, V1, X, d)
             err1=norm_apolar(T1-T)
             push!(d1,err1)
             t2 = @elapsed w2, V2, Info = rne_n_tr(T, w1, V1,Dict{String,Any}("maxIter" => 10,"epsIter" => 1.e-1))
             push!(N1,Info["nIter"])
             push!(t1s,t1+t2)
             T2 = tensor(w2, V2, X, d)
             err2=norm_apolar(T2-T)
             push!(d2,err2)

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
             t3=@elapsed V3, Info = rgn_v_tr(T,V11,Dict{String,Any}("maxIter" => 10,"epsIter" => 1.e-1))
             push!(N2,sum(Info["nIter"]))
             push!(t2s,t1+t3)
             T3 = sum((transpose(V3[:,i])*X)^d for i in 1:r)
             err3=norm_apolar(T3-T)
             push!(d3,err3)

        end



        d2_sort = sort(d2)
        min_rns = d2_sort[1]
        med_rns = d2_sort[div(N,2)]
        max_rns = d2_sort[end]

        d3_sort = sort(d3)
        min_rqn=d3_sort[1]
        med_rqn=d3_sort[div(N,2)]
        max_rqn=d3_sort[end]

        return min_rns, med_rns, max_rns, sum(t1s)/length(t1s), sum(N1)/length(N1), min_rqn, med_rqn, max_rqn, sum(t2s)/length(t2s), sum(N2)/length(N2)



        end
