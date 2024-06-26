export opt, rne_n_tr_r_step, rne_n_tr_r

using LinearAlgebra
using MultivariatePolynomials
using DynamicPolynomials

#==
```
opt(W,V,P) ⤍ Vector
```
This function solves the linear least square problem: 1/2 min_{α1,...,αr} ||∑αiW[i](V[:,i]'x)^d-P||^2 over the real field.

==#
function opt(W::Vector, V::Matrix,P)
    d=maxdegree(P)
    r=size(W,1)
    A=fill(0.0,r,r)
    B=fill(0.0,r)
    for i in 1:r
        A[i,:]=[W[i]*W[j]*(dot(V[:,i],V[:,j]))^d for j in 1:r]
    end
    for i in 1:r
        B[i]=W[i]*P(V[:,i])
    end
    C=A\B
end


#==
```
rne_n_tr_r_step(delta, W::Vector, V::Matrix,P) ➡ gives symmetric decomposition W1, V1 of rank r=size(W,1).
```
Riemannian Newton method with trust region (one iteration) from real initial point W, V.

W is a real vector and V is a real matrix and its columns are normalized.

delta is the radius of the sphere i.e. the trust region.

==#
function rne_n_tr_r_step(delta, W::Vector, A::Matrix, P)
    r=size(W,1)
    n=size(A,1)
    d=maxdegree(P)
    X=variables(P)
    Q=fill(0.0, r+n*r, r+(n-1)*r)
    Q[1:r,1:r]=Matrix(1.0*I, r, r)

    for i in 1:r
        Pr = Matrix(1.0*I, n, n)-A[:,i]*A[:,i]'
        F = qr(Pr, Val(true))
        k = F.Q
        Q[r+(i-1)*n+1:r+i*n,r+(i-1)*(n-1)+1:r+i*(n-1)]=k[:,1:n-1]
    end

    M1=fill(0.0, r)
    for i in 1:r
        M1[i]=sum(W[l]*(dot(A[:,i],A[:,l]))^d for l in 1:r)-P(A[:,i])
    end

    M2=fill(0.0, n*r)
    for i in 1:r
        u=W[i]*gradeval(P,X,A[:,i])
        a=fill(0.0, n)
        for j in 1:n
            a[j]=sum(d*W[i]*W[l]*A[j,l]*(dot(A[:,i],A[:,l]))^(d-1) for l in 1:r)
        end
        p=a-u
        M2[(i-1)*n+1:i*n]=p
    end

    M3=fill(0.0, r+n*r)
    G =fill(0.0, r+(n-1)*r)
    M3[1:r]       = M1
    M3[r+1:r+n*r] = M2
    G = Q'*M3
    Mat = fill(0.0, r+n*r,r+n*r)
    Mat1= fill(0.0, r,r)
    for i in 1:r

        Mat1[:,i]=[dot(A[:,j],A[:,i])^d for j in 1:r]
    end
    Mat2=zeros(Float64,r,0)
    for i in 1:r
        a=zeros(r,n)
        for j in 1:r
            a[j,:]=(d*W[i]*(dot(A[:,j],A[:,i]))^(d-1))*A[:,j]'
        end
        Mat2=hcat(Mat2,a)
    end

    Mat3=fill(0.0, n*r,n*r)
    for i in 1:r
        a=d*W[i]^2*(Matrix(1.0*I, n, n)+(d-1)*A[:,i]*A[:,i]')
        Mat3[(i-1)*n+1:i*n,(i-1)*n+1:i*n]=a
    end

    for i in 1:r-1
        for j in i+1:r
            b=d*W[i]*W[j]*((dot(A[:,i],A[:,j]))^(d-1)*Matrix(1.0*I, n, n)+((d-1)*(dot(A[:,i],A[:,j]))^(d-2))*A[:,j]*A[:,i]')
            Mat3[(i-1)*n+1:i*n,(j-1)*n+1:j*n]=b
            Mat3[(j-1)*n+1:j*n,(i-1)*n+1:i*n]=b'
        end
    end
    Mat[1:r,1:r]=Mat1
    Mat[1:r,r+1:r+n*r]=Mat2
    Mat[r+1:r+n*r,1:r]=Mat2'
    Mat[r+1:r+n*r,r+1:r+n*r]=Mat3
    GL=zeros(n,r)

    for i in 1:r
        GL[1:n,i]=M3[r+(i-1)*n+1:r+(i-1)*n+n]
    end
    VS=zeros(r+n*r)
    for i in 1:r
        VS[r+(i-1)*n+1:r+(i-1)*n+n]=dot(A[:,i],GL[:,i])*ones(n)
    end
    DEL=-diagm(VS)
    Mat+=DEL
    H=Q'*Mat*Q
    N1=-H\G
    N1=Q*N1
    l1=G'*G
    l2=G'*H*G
    l=l1/l2
    N2=-l*G
    N2=Q*N2
    xi=norm(N1)
    pi=norm(N2)

    if xi <= delta
        Ns=N1

    elseif xi > delta && pi >= delta
        Ns=-Q*((delta/norm(G))*G)
    else
        a1=(norm(N1-N2))^2
        a2=2*(-(norm(N1))^2-2*(norm(N2))^2+3*dot(N1,N2))
        a3=-delta^2+4*(norm(N2))^2-4*dot(N1,N2)+(norm(N1))^2
        tau=solve_tr(a1,a2,a3)
        Ns=N2+(tau-1)*(N1-N2)
    end
    W1=zeros(r)
    W1=Ns[1:r]
    A1=fill(0.0,n,r)
    for i in 1:r
        A1[:,i]=Ns[r+(i-1)*n+1:r+i*n]
    end
    W2=zeros(r)
    A2=zeros(n, r)

    for i in 1:r
        #tg=(W1[i]+W[i])*(transpose(A[:,i])*X)^d+d*W[i]*(((transpose(A[:,i])*X)^(d-1))*(transpose(A1[:,i])*X))
        #println(tg)
        #V=transpose(hankel(tg,1))
        #u,s,v=svd(V)
        #A2[:,i]=u[:,1]
        #W2[i]=(W1[i]+W[i])*(dot(A2[:,i],A[:,i]))^d+d*W[i]*((dot(A2[:,i],A[:,i]))^(d-1))*(dot(A2[:,i],A1[:,i]))
        W2[i]=W1[i]+W[i]
        A2[:,i]=(A[:,i]+A1[:,i])/(norm(A[:,i]+A1[:,i]))
    end
    S=Q'*Ns
    w1=0.5*(norm_apolar(hpol(W,A,X,d)-P))^2
    w2=0.5*(norm_apolar(hpol(W2,A2,X,d)-P))^2
    w3=w1+G'*S+0.5*S'*H*S
    r1=w1-w2
    r2=w1-w3
    ki=r1/r2
    if ki>=0.2
        op1,op2=W2,A2
    else
        op1,op2=W,A
    end

    al=0.5*(norm_apolar(P))
    t=exp(-14*(ki-1/3))
    er=(1/3+(2/3)*(1/(1+t)))*delta
    if ki > 0.6
        delta=min(2*norm(Ns),al)
    else
        delta=min(er,al)
    end
    return delta,op1,op2
end



"""
```
rne_n_tr_r(P, A0, B0, Dict{String,Any}("maxIter" => N,"epsIter" => ϵ))⤍ A, B, Info
```
This function gives a low symmetric rank approximation of a real valued
symmetric tensor by applying an exact Riemannian Newton iteration with
dog-leg trust region steps to the associate non-linear-least-squares
problem. The optimization set is parameterized by weights and unit vectors.
Let r be the approximation rank. The approximation is of the form
of linear combination of r linear forms to the d-th power ∑w_i*(v_i^tx)^d, with i=1,...,r.
This approximation is represented by a vector of r real numbers W=(w_i) (weight vector), and a matrix
of normalized columns V=[v_1;...;v_r].

Input:
 - P: Homogeneous polynomial (associated to the symmetric tensor to approximate).
 - A0: Initial weight vector of size equal to the approximation rank.
 - B0: Initial matrix of row size equal to the dimension of P and column size equal to the
    approximation rank.

The options are
 - N: Maximal number of iterations (by default 500).
 - ϵ: The radius of the trust region (by default 1.e-3).

Output:
  - A: Weight vector of size equal to the approximation rank.
  - B: Matrix of row size equal to the dimension of P and column size equal to the
   approximation rank. The columns vectors of B are normalized.
  - Info: 'd0' (resp. 'd*') represents the initial (resp. the final) residual error,
      'nIter' is for the number of iterations needed to find the approximation.

"""
function rne_n_tr_r(P, A0::Vector, B0::Matrix,
                Info = Dict(
                    "maxIter" => 500,
                    "epsIter" => 1.e-3))
    d = maxdegree(P)
    X = variables(P)
    r=size(A0,1)
    n=size(X,1)
    N   = (haskey(Info,"maxIter") ? Info["maxIter"] : 500)
    eps = (haskey(Info,"epsIter") ? Info["epsIter"] : 1.e-3)
    De=fill(0.0,N)
    E=fill(0.0,N*r)
    F=fill(0.0,n,N*r)
    P0=hpol(A0,B0,X,d)
    d0=norm_apolar(P-P0)
    C=opt(A0,B0,P)
    A1=fill(0.0,r)
    B1=fill(0.0,n,r)
    B1=B0
    for i in 1:r
        A1[i]=A0[i]*C[i]
    end
    P1=hpol(A1,B1,X,d)
    d1=norm_apolar(P1-P)
    if d0<d1
        A1=A0
    end
    for i in 1:r
        A1[i]=A1[i]*norm(B1[:,i])^d
        B1[:,i]=B1[:,i]/norm(B1[:,i])
    end
    a0=Delta(P,A1)
    De[1], E[1:r], F[1:n,1:r] = rne_n_tr_r_step(a0,A1,B1,P)
    W=fill(0.0,r)
    V=fill(0.0,n,r)
    i = 2
    while  i < N && De[i-1] > 1.e-3
          De[i], E[(i-1)*r+1:i*r], F[1:n,(i-1)*r+1:i*r]=rne_n_tr_r_step(De[i-1],E[(i-2)*r+1:(i-1)*r],F[1:n,(i-2)*r+1:(i-1)*r],P)
          W,V=E[(i-1)*r+1:i*r], F[1:n,(i-1)*r+1:i*r]
          i += 1
     end
    P4=hpol(W,V,X,d)
    d2=norm_apolar(P4-P)
    A=fill(0.0,r)
    B=fill(0.0,n,r)
    if d2<d1
        A,B=W,V
    else
        A,B=A1,B1
    end
    P5=hpol(A,B,X,d)
    d3=norm_apolar(P-P5)
    #println("N:",i)
    #println("dist0: ",d0)
    #println("dist*: ",d3)
    Info["nIter"] = i
    Info["d0"] = d0
    Info["d*"] = d3

    return A,B,Info

end

#random initial point
function rne_n_tr_r(P, r::Int64, N::Int64=500)
    d = maxdegree(P)
    X = variables(P)
    n = size(X,1)
    A0 = rand(Float64,r)
    B0 = rand(Float64,n,r)

    #A0, B0 = decompose_qr(P,cst_rkf(r))
    return rne_n_tr_r(P, A0,B0, N)
end
