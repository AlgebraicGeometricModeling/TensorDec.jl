using DynamicPolynomials, MultivariateSeries
include("sym_tens_fct.jl")
include("../src/RNE_N_TR.jl")

function tst_ex51(n)
d=3
X = (@polyvar x[1:n])[1]
T=zeros(n,n,n)
for i in 1:n
    for j in 1:n
        for k in 1:n
            T[i,j,k]= (-1)^i/i+(-1)^j/j+(-1)^k/k
        end
    end
end

P=sum(T[i,j,k]*X[i]*X[j]*X[k] for i in 1:n for j in 1:n for k in 1:n)

t1 = @elapsed w1, V1, Info = decompose(P,1)
P1=hpol(w1,V1,X,d)
d0=norm_apolar(P-P1)
t2 = @elapsed w2, V2, Info = rne_n_tr(P, w1, V1)
P2=hpol(w2,V2,X,d)
d1=norm_apolar(P-P2)
N=Info["nIter"]
return w2[1,1], d0, d1, t1+t2, N
end

function tst_ex52(n)
d=5
X = (@polyvar x[1:n])[1]
T=zeros(n,n,n,n,n);
for i in 1:n
    for j in 1:n
        for k in 1:n
            for l in 1:n
                for m in 1:n
            T[i,j,k,l,m]= (-1)^i*log(i)+(-1)^j*log(j)+(-1)^k*log(k)+(-1)^l*log(l)+(-1)^m*log(m);
                end
            end
        end
    end
end

P=sum(T[i,j,k,l,m]*X[i]*X[j]*X[k]*X[l]*X[m] for i in 1:n for j in 1:n for k in 1:n for l in 1:n for m in 1:n);

t1 = @elapsed w1, V1, Info = decompose(P,1)
P1=hpol(w1,V1,X,d)
d0=norm_apolar(P-P1)
t2 = @elapsed w2, V2, Info = rne_n_tr(P, w1, V1)
P2=hpol(w2,V2,X,d)
d1=norm_apolar(P-P2)
N=Info["nIter"]
return w2[1,1], d0, d1, t1+t2, N
end
