using LinearAlgebra, DynamicPolynomials, MultivariateSeries
# To make the sum of the entries of the weight vector in a symmetric tensor decomposition equal to 1.
function mix_nrmlz(w, Xi, d)
    s   = sum(abs(x) for x in w)
    w1  = w/s
    Xi1 = Xi*s^(1/d)
    if !iseven(d)
        for i in 1:length(w1)
            if w1[i] <0
                w1[i] = - w1[i]
                Xi1[:,i] = -Xi1[:,i]
            end
        end
    end
    return w1, Xi1
end
"""
```
moment_var_diff(S) → gives the first, second and third order moments M1, M2 and M3
                     computed using the dataset S.
```
"""
function moment_var_diff(S) #method of moments
    n=size(S,1)
    N=size(S,2)
    d=3
    X = (@polyvar x[1:n])[1]
    E = sum(S[:,i] for i in 1:N)/N
    K=sum((S[:,i]-E)*(S[:,i]-E)' for i in 1:N)/N
    v=svd(K).U[:,end]
    s=svd(K).S[end]
    M1=sum(S[:,i]*(v'*(S[:,i]-E))^2 for i in 1:N)/N
    M2=sum(S[:,i]*S[:,i]' for i in 1:N)/N-s*diagm(ones(n))
    M=sum(dot(S[:,i],X)^d for i in 1:N)/N
    M3=M-d*(dot(M1,X))*sum(X[i]^(d-1) for i in 1:length(X))
    M1, M2, M3
end

"""
```
optdec(w, Xi, M2) → uses the second order moments M2 to identify the cluster proportions w and the matrix of the means in columns Xi.
```
"""
function optdec(w, Xi, M2)
    n=size(Xi,1)
    r=size(Xi,2)
    foo = [w[i]*vec(Xi[:,i]*Xi[:,i]') for i in 1:r]
    A=hcat(foo...)
    B=vec(M2)
    a=(A'*A)\(A'*B)
    w1=zeros(r)
    Xi1=zeros(n,r)
    for i in 1:r
        w1[i]=w[i]*a[i]^3
        Xi1[:,i]=Xi[:,i]/a[i]
    end
    return w1, Xi1
end
"""
```
var_diff(w, Xi, M1) → uses the first order moments M1 to identify the variance in each cluster.
```
"""
function var_diff(w, Xi, M1)
    n = size(Xi,1)
    r = length(w)
    A=zeros(n,r)
    for i in 1:r
        A[:,i]=w[i]*Xi[:,i]
    end
    a= (A'*A)\(A'*M1)
end

"""
```
moment_var_diff(S, r) → given the dataset S and the number of cluster r such that r ≤ number of features, this function estimates using the method of moments the cluster proportion, the means and the covariance for each Gaussian distribution within a spherical Gaussian mixture model that fits with the studied data S.
```
"""
function moment_var_diff(S,r) #method of moments
    d=3
    M1, M2, M3 = moment_var_diff(S)
    w1, Mu1 = decompose(M3,r)
    w2, Mu2 = rne_n_tr_r(M3, copy(w1), copy(Mu1), Dict{String,Any}("maxIter" => 500,"epsIter" => 1.e-3));
    w, Mu = optdec(w2, Mu2, M2)
    w, Mu = mix_nrmlz(w, Mu, d)
    Sigma = var_diff(w,Mu, M1)
    for i in 1:r
        if Sigma[i]<0
        Sigma[i]=-Sigma[i]
        end
    end
    return w, Mu, Sigma
end
