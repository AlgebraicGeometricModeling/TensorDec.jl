using LinearAlgebra
using MultivariatePolynomials
using DynamicPolynomials



#Riemannian Gauss-Newton iteration on r-cartesian product of Veronese manifold.
function rgn_v_step(P,V)
    X=variables(P)
    r=size(V,2)
    n=size(V,1)
    d=maxdegree(P)
    Q=zeros(ComplexF64,n,2*n*r)
    for i in 1:r
        vR=zeros(2*n)
        vI=zeros(2*n)
        Q1=zeros(2*n,2*n)
        Q2=zeros(ComplexF64,n,2*n)
        vR[1:n]=real(V[:,i])
        vR[n+1:2*n]=imag(V[:,i])
        vI[1:n]=-imag(V[:,i])
        vI[n+1:2*n]=real(V[:,i])
        M=Matrix(1.0*I,2*n,2*n)+(d-1)*(norm(V[:,i]))^(-2)*((vR*vR')+(vI*vI'))
        u,s,v=svd(M)
        u1=vR/(sqrt(d)*norm(vR))
        u11=(diagm(s))^(1/2)*u'*u1
        q,l=qr(Matrix(1.0*I,2*n,2*n)-u11*u11',Val(true))
        Q1[:,1]=u1
        Q1[:,2:end]=u*(diagm(s))^(-1/2)*q[:,1:2*n-1]
        for j in 1:2*n
            Q2[:,j]=Q1[1:n,j]+Q1[n+1:2*n,j]*im
        end
        Q[1:n,2*n*(i-1)+1:2*i*n]=Q2
    end
    #gradient
    G=zeros(2*n*r)
    for i in 1:r
        a=gradeval(P,X,conj(V[:,i]))
        b=sqrt(1/d)*(norm(V[:,i]))^(-d+1)
        for j in 1:2*n
            c=Q[1:n,2*n*(i-1)+j]
            G[2*n*(i-1)+j]=b*(d*(sum(real(dot(c,V[:,k])*(dot(V[:,i],V[:,k]))^(d-1)) for k in 1:r))-real(dot(c,a)))
        end
    end
    #Approximation Gauss-Newton of Hessian
    H=zeros(2*n*r,2*n*r)
    for i in 1:r
        a=norm(V[:,i])^(-d+1)
        for j in 1:2*n
            b=Q[1:n,2*n*(i-1)+j]
            for k in 1:r
                c=norm(V[:,k])^(-d+1)
                for l in 1:2*n
                    f=Q[1:n,2*n*(k-1)+l]
                    H[2*n*(i-1)+j,2*n*(k-1)+l]=a*c*(real(dot(b,f)*(dot(V[:,i],V[:,k]))^(d-1))+(d-1)*(real(dot(b,V[:,k])*dot(V[:,i],f)*
                    (dot(V[:,i],V[:,k]))^(d-2))))
                end
            end
        end
    end
    #solve Gauss-Newton equation
    N=-pinv(H)*G
    V1=zeros(2*n,r)
    for i in 1:r
        V1[:,i]=N[(i-1)*2*n+1:2*i*n]
    end
    #retraction
    V2=zeros(ComplexF64,n,r)
    for i in 1:r
        a=(transpose(V[:,i])*X)^(d-1)
        b=sqrt(d)*(norm(V[:,i]))^(-d+1)
        tg=a*((transpose(V[:,i])*X)+b*sum(V1[k,i]*(transpose(Q[1:n,2*n*(i-1)+k])*X) for k in 1:2*n))
        U=transpose(hankel(tg,1))
        u,s,v=svd(U)
        w=tg(conj(u[:,1]))
        V2[:,i]=w^(1/d)*u[:,1]
    end
    return V2
end

#Riemannian Gauss-Newton with trust region iteration on r-cartesian product of Veronese manifold.
function rgn_v_tr_step(delta,V,P)
    X=variables(P)
    n=size(V,1)
    r=size(V,2)
    d=maxdegree(P)
    Q=zeros(ComplexF64,n,2*n*r)
    for i in 1:r
        vR=zeros(2*n)
        vI=zeros(2*n)
        Q1=zeros(2*n,2*n)
        Q2=zeros(ComplexF64,n,2*n)
        vR[1:n]=real(V[:,i])
        vR[n+1:2*n]=imag(V[:,i])
        vI[1:n]=-imag(V[:,i])
        vI[n+1:2*n]=real(V[:,i])
        M=Matrix(1.0*I,2*n,2*n)+(d-1)*(norm(V[:,i]))^(-2)*((vR*vR')+(vI*vI'))
        u,s,v=svd(M)
        u1=vR/(sqrt(d)*norm(vR))
        u11=(diagm(s))^(1/2)*u'*u1
        q,l=qr(Matrix(1.0*I,2*n,2*n)-u11*u11',Val(true))
        Q1[:,1]=u1
        Q1[:,2:end]=u*(diagm(s))^(-1/2)*q[:,1:2*n-1]
        for j in 1:2*n
            Q2[:,j]=Q1[1:n,j]+Q1[n+1:2*n,j]*im
        end
        Q[1:n,2*n*(i-1)+1:2*i*n]=Q2
    end
    #gradient
    G=zeros(2*n*r)
    for i in 1:r
        a=gradeval(P,X,conj(V[:,i]))
        b=sqrt(1/d)*(norm(V[:,i]))^(-d+1)
        for j in 1:2*n
            c=Q[1:n,2*n*(i-1)+j]
            G[2*n*(i-1)+j]=b*(d*(sum(real(dot(c,V[:,k])*(dot(V[:,i],V[:,k]))^(d-1)) for k in 1:r))-real(dot(c,a)))
        end
    end
    #Approximation Gauss-Newton of Hessian
    H=zeros(2*n*r,2*n*r)
    for i in 1:r
        a=norm(V[:,i])^(-d+1)
        for j in 1:2*n
            b=Q[1:n,2*n*(i-1)+j]
            for k in 1:r
                c=norm(V[:,k])^(-d+1)
                for l in 1:2*n
                    f=Q[1:n,2*n*(k-1)+l]
                    H[2*n*(i-1)+j,2*n*(k-1)+l]=a*c*(real(dot(b,f)*(dot(V[:,i],V[:,k]))^(d-1))+(d-1)*(real(dot(b,V[:,k])*dot(V[:,i],f)*
                    (dot(V[:,i],V[:,k]))^(d-2))))
                end
            end
        end
    end
    #solve Gauss-Newton equation

    N1=-pinv(H)*G

    l1=G'*G
    l2=G'*H*G
    l=l1/l2
    N2=-l*G
    xi=norm(N1)
    pi=norm(N2)

    if xi <= delta
        Ns=N1

    elseif xi > delta && pi >= delta
        Ns=-((delta/norm(G))*G)
    else
        a1=(norm(N1-N2))^2
        a2=2*(-(norm(N1))^2-2*(norm(N2))^2+3*dot(N1,N2))
        a3=-delta^2+4*(norm(N2))^2-4*dot(N1,N2)+(norm(N1))^2
        tau=solve(a1,a2,a3)
        Ns=N2+(tau-1)*(N1-N2)
    end
    V1=zeros(2*n,r)
    for i in 1:r
        V1[:,i]=Ns[(i-1)*2*n+1:2*i*n]
    end
    #retraction
    V2=zeros(ComplexF64,n,r)
    for i in 1:r
        a=(transpose(V[:,i])*X)^(d-1)
        b=sqrt(d)*(norm(V[:,i]))^(-d+1)
        tg=a*((transpose(V[:,i])*X)+b*sum(V1[k,i]*(transpose(Q[1:n,2*n*(i-1)+k])*X) for k in 1:2*n))
        U=transpose(hankel(tg,1))
        u,s,v=svd(U)
        w=tg(conj(u[:,1]))
        V2[:,i]=w^(1/d)*u[:,1]
    end
    #Accept or reject solution
    P1=sum((transpose(V[:,i])*X)^d for i in 1:r)
    P2=sum((transpose(V2[:,i])*X)^d for i in 1:r)
    w1=0.5*(norm_apolar(P1-P))^2
    w2=0.5*(norm_apolar(P2-P))^2
    w3=w1+G'*Ns+0.5*Ns'*H*Ns
    r1=w1-w2
    r2=w1-w3
    ki=r1/r2
    if ki>=0.2
        sol=V2
    else
        sol=V
    end
#update trut region radius
    al=0.5*(norm_apolar(P))
    t=exp(-14*(ki-1/3))
    er=(1/3+(2/3)*(1/(1+t)))*delta
    if ki > 0.6
        delta=min(2*norm(Ns),al)
    else
        delta=min(er,al)
    end


    delta,sol

    end

    #loop RGN with trust region
    function rgn_v_tr(P, B0,
                    Info = Dict(
                        "maxIter" => 500,
                        "epsIter" => 1.e-3))
        r = size(B0,2)
        d = maxdegree(P)
        X = variables(P)
        n=size(X,1)

        N   = (haskey(Info,"maxIter") ? Info["maxIter"] : 500)
        eps = (haskey(Info,"epsIter") ? Info["epsIter"] : 1.e-3)

        De=0.0
        F=zeros(ComplexF64,n,r)

        P0=sum((transpose(B0[:,i])*X)^d for i in 1:r)
        d0=norm_apolar(P-P0)
        a0=Delta1(P,B0)
        De, F = rgn_v_tr_step(a0,B0,P)


        V = zeros(ComplexF64,n,r)
        i = 2
        while  i < N && De > eps
            De, F = rgn_v_tr_step(De,F,P)
            V = F
            i += 1
        end
        P4 = sum((transpose(V[:,i])*X)^d for i in 1:r)
        d2 = norm_apolar(P4-P)
        B=zeros(ComplexF64,n,r)
        if d2<d0
            B=V
        else
            B=B0
        end
        P5=sum((transpose(B[:,i])*X)^d for i in 1:r)
        d3=norm_apolar(P-P5)

        Info["nIter"] = i
        Info["d0"] = d0
        Info["d*"] = d3

        return B, Info
    end
