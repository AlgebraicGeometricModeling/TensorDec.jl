using LinearAlgebra
using MultivariatePolynomials
using DynamicPolynomials

function h(u,v,u1,v1,d)
    a=(v1'*v)^(d-2)*((u'*u1)*(v1'*v)+(d-1)*(u1'*v)*(u'*v1))
    if (u==v && u1!=v1) || (u!=v && u1==v1)
        a=(1/sqrt(d))*a
    end
    if u==v && u1==v1
        a=(1/d)*a
    end
    return a
end

function rqn_tr_ver_step(delta,P,W,V)
    X=variables(P)
    r=size(W,1)
    n=size(X,1)
    d=maxdegree(P)
    C=opt(W,V,P)
    for i in 1:r
        W[i]=W[i]*C[i]
    end


    #B=[v_1,u_1,1,...,u_1,n-1,...,v_r,u_r,1,...,u_r,n-1]
    B=zeros(n,n*r)
    for i in 1:r
        q,s=qr(Matrix(1.0*I, n, n)-V[:,i]*V[:,i]',Val(true))
        B[:,(i-1)*n+1]=V[:,i]
        B[:,(i-1)*n+2:(i-1)*n+n]=q[1:n,1:n-1]
    end


    #gradient
    K=zeros(n,r)
    gr=[DynamicPolynomials.differentiate(P, X[i]) for i in 1:n]
    for i in 1:r
        K[1,i]=sum(W[j]*(V[:,j]'*V[:,i])^d for j in 1:r)-P(V[:,i])
        gr1=[gr[l](V[:,i]) for l in 1:n]
        for j in 2:n
            K[j,i]=sqrt(d)*sum(W[l]*(B[:,(i-1)*n+j]'*V[:,l])*(V[:,l]'*V[:,i])^(d-1) for l in 1:r)-(1/sqrt(d))*(B[:,(i-1)*n+j]'*gr1)
        end
    end
    G=vec(K)
    #Approximation Gauss-Newton of the Hessian
    A=zeros(n*r,n*r)
    for i in 1:r
       for j in 1:n
       if i < r
       a=[h(B[:,(i-1)*n+j],V[:,i],B[:,(i-1)*n+l],V[:,i],d) for l in j:n]
       b=[h(B[:,(i-1)*n+j],V[:,i],B[:,(m-1)*n+l],V[:,m],d) for m in i+1:r for l in 1:n]
       A[(i-1)*n+j,(i-1)*n+j:end]=vcat(a,b)
       end
       if i==r
       A[(i-1)*n+j,(i-1)*n+j:end]=[h(B[:,(i-1)*n+j],V[:,i],B[:,(i-1)*n+l],V[:,i],d) for l in j:n]
       end
       end
       end
       H1=Symmetric(A,:U)
       H1=convert(Array,H1)
       # solve Gauss-Newton equation
       f=norm_apolar(hpol(W,V,X,d)-P)
       S=10^(-1)*(f/norm_apolar(P))^(3/4)*norm(H1)
       H=H1+S*Matrix(1.0*I, n*r, n*r)
       N=zeros(n*r)
       N=-H\G
       #trust region
       l1=G'*G
       l2=G'*H*G
       l=l1/l2
       N1=-l*G
       xi=norm(N)
       pi=norm(N1)

       if xi <= delta
           Ns=N

       elseif xi > delta && pi >= delta
           Ns=-((delta/norm(G))*G)
       else
           a1=(norm(N-N1))^2
           a2=2*(-(norm(N))^2-2*(norm(N1))^2+3*dot(N,N1))
           a3=-delta^2+4*(norm(N))^2-4*dot(N,N1)+(norm(N))^2
           tau=solve(a1,a2,a3)
           Ns=N1+(tau-1)*(N-N1)
       end
       #retraction
       W1=zeros(r)
       V1=zeros(n-1,r)
       W2=zeros(r)
       V2=zeros(n,r)
       for i in 1:r
           W1[i]=Ns[(i-1)*n+1]
           V1[:,i]=Ns[(i-1)*n+2:i*n]
       end
       for i in 1:r
           M=B[:,(i-1)*n+2:(i-1)*n+n]
           L=M'*X
           tg=sum(V1[l,i]*L[l] for l in 1:n-1)
           Q=(V[:,i]'*X)^(d-1)*((W[i]+W1[i])*(V[:,i]'*X)+sqrt(d)*tg)
           U=transpose(hankel(Q,1))
           u,s,v=svd(U)
           W2[i]=Q(u[:,1])
           V2[:,i]=u[:,1]
       end
       #Accept or reject solution W2,V2
       w1=0.5*(norm_apolar(hpol(W,V,X,d)-P))^2
       w2=0.5*(norm_apolar(hpol(W2,V2,X,d)-P))^2
       w3=w1+G'*Ns+0.5*Ns'*H*Ns
       r1=w1-w2
       r2=w1-w3
       ki=r1/r2
       if ki>=0.2
           op1,op2=W2,V2
       else
           op1,op2=W,V
       end
       #update raduis
       al=0.5*(norm_apolar(P))
       t=exp(-14*(ki-1/3))
       er=(1/3+(2/3)*(1/(1+t)))*delta
       if ki > 0.6
           delta=min(2*norm(Ns),al)
       else
           delta=min(er,al)
       end


       delta,op1,op2,H

end

#algorithm rqn_tr_ver


function rqn_tr_ver(P, A0::Vector, B0::Matrix,
                Info = Dict(
                    "maxIter" => 500,
                    "epsIter" => 1.e-3))
    r = length(A0)
    d = maxdegree(P)
    X = variables(P)
    n=size(X,1)

    N   = (haskey(Info,"maxIter") ? Info["maxIter"] : 500)
    eps = (haskey(Info,"epsIter") ? Info["epsIter"] : 1.e-3)

    De=0.0 #fill(0.0,N)
    E=fill(0.0,r)
    F=fill(0.0,n,r)

    P0=tensor(A0,B0,X,d)
    d0=norm_apolar(P-P0)
    for i in 1:r
        A0[i]=norm(B0[:,i])^d
        B0[:,i]=(B0[:,i]/norm(B0[:,i]))
    end


    a0=Delta(P,A0)
    De, E, F = rqn_tr_ver_step(a0,P,A0,B0)

    W = fill(0.0,r)
    V = fill(0.0,n,r)
    i = 2
    while  i < N && De > eps
        De, E, F = rqn_tr_ver_step(De,P,E,F)
        W, V = E, F
        i += 1
    end
    P4 = tensor(W,V,X,d)
    d2 = norm_apolar(P4-P)
    A=fill(0.0,r)
    B=fill(0.0,n,r)
    if d2<d0
        A,B=W,V
    else
        A,B=A0,B0
    end
    P5=tensor(A,B,X,d)
    d3=norm_apolar(P-P5)

    Info["nIter"] = i
    Info["d0"] = d0
    Info["d*"] = d3

    #println("N:",i)
    #println("dist0: ",d0)
    #println("dist*: ",d3)

    return A,B, Info
end
