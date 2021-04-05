using LinearAlgebra
using MultivariatePolynomials
using DynamicPolynomials


function rn_tr_mat_step(delta, W::Vector, V::Matrix,P)
    X=variables(P)
    r=size(W,1)
    n=size(X,1)
    d=maxdegree(P)
    c= typeof(V[1,1])
    G1=fill(0.0, r)
    G2=fill(zero(c), n*r)
    A=fill(0.0, r, r)
    B=fill(zero(c), n*r, r)
    C=fill(zero(c), n*r, n*r)
    D=fill(zero(c), n*r, n*r)
    for j in 1:r
        G1[j]=sum(W[i]*real((dot(V[:,j],V[:,i]))^d) for i in 1:r)-real(P(conj(V[:,j])))
    end
    for j in 1:r
        u=W[j]*conj(gradeval(P,X,conj(V[:,j])))
        a=0.5*(d*sum(W[j]*W[l]*(dot(V[:,l],V[:,j]))^(d-1)*conj(V[:,l]) for l in 1:r)-u)
        G2[(j-1)*n+1:j*n]=a
    end
    for i in 1:r
        A[i,:]=[real((dot(V[:,i],V[:,j]))^d) for j in 1:r]
    end
    for i in 1:r
        for j in 1:r
            if i != j
                B[(i-1)*n+1:i*n, j]=0.5*d*W[i]*(dot(V[:,j],V[:,i]))^(d-1)*conj(V[:,j])
            else
                u=sum(W[l]*(dot(V[:,l],V[:,j]))^(d-1)*conj(V[:,l]) for l in 1:r)
                v=conj(gradeval(P,X,conj(V[:,j])))
                B[(i-1)*n+1:i*n, j]=0.5*(d*(W[j]*(dot(V[:,j],V[:,j]))^(d-1)*conj(V[:,j])+u)-v)
            end
        end
    end

    for j in 1:r
        b=fill(zero(c), n, n)
        for i in 1:n
            b[i,:]=[d*(d-1)*sum(W[l]*W[j]*conj(V[i,l])*conj(V[k,l])*(dot(V[:,l],V[:,j]))^(d-2) for l in 1:r) for k in 1:n]
        end
        C[(j-1)*n+1:j*n,(j-1)*n+1:j*n]=0.5*(b-W[j]*conj(hessianeval(P,X,conj(V[:,j]))))
    end
    for i in 1:r
        for j in 1:r
            D[(i-1)*n+1:i*n,(j-1)*n+1:j*n]=0.5*(d*W[i]*W[j]*(dot(V[:,i],V[:,j]))^(d-2)*(dot(V[:,i],V[:,j])*Matrix(one(c)*I, n, n)+(d-1)*V[:,j]*V[:,i]'))
        end
    end
    G=zeros(r+2*n*r)
    H=zeros(r+2*n*r,r+2*n*r)
    G[1:r]=G1
    G[r+1:r+r*n]=2*real(G2)
    G[r+r*n+1:r+2*r*n]=-2*imag(G2)
    H[1:r,1:r]=A
    H[1:r,r+1:r+n*r]=2*transpose(real(B))
    H[1:r,r+n*r+1:r+2*n*r]=-2*transpose(imag(B))
    H[r+1:r+n*r,1:r]=2*real(B)
    H[r+1:r+n*r,r+1:r+n*r]=2*(real(C)+real(D))
    H[r+1:r+n*r,r+n*r+1:r+2*n*r]=-2*(imag(C)+imag(D))
    H[r+n*r+1:r+2*n*r,1:r]=-2*imag(B)
    H[r+n*r+1:r+2*n*r,r+1:r+n*r]=2*(imag(D)-imag(C))
    H[r+n*r+1:r+2*n*r,r+n*r+1:r+2*n*r]=2*(real(D)-real(C))
    M=zeros(r+2*n*r,r+r*(2*n-1))
    M1=zeros(n*r,r*(2*n-1))
    M2=zeros(n*r,r*(2*n-1))
    for i in 1:r
        v=zeros(2*n)
        v[1:n]=real(V[:,i])
        v[n+1:2*n]=imag(V[:,i])
        q,s=qr(Matrix(1.0*I, 2*n, 2*n)-v*v',Val(true))
        M1[(i-1)*n+1:i*n,(i-1)*(2*n-1)+1:i*(2*n-1)]=q[1:n,1:2*n-1]
        M2[(i-1)*n+1:i*n,(i-1)*(2*n-1)+1:i*(2*n-1)]=q[n+1:2*n,1:2*n-1]
    end
    M[1:r,1:r]=Matrix(1.0*I, r, r)
    M[r+1:r+n*r,r+1:r+r*(2*n-1)]=M1
    M[r+n*r+1:r+2*n*r,r+1:r+r*(2*n-1)]=M2

    #Weingarten term computation
    #-----------------------------------------------------------------
    GL=zeros(2*n,r)

    for i in 1:r
        GL[1:n,i]=G[r+(i-1)*n+1:r+(i-1)*n+n]
        GL[n+1:2*n,i]=G[r+r*n+(i-1)*n+1:r+r*n+(i-1)*n+n]
    end

    VR=zeros(2*n,r)

    for i in 1:r
        VR[1:n,i]=real(V[:,i])
        VR[n+1:2*n,i]=imag(V[:,i])
    end

    VS=zeros(r+2*n*r)
    for i in 1:r
        VS[r+(i-1)*n+1:r+(i-1)*n+n]=dot(VR[:,i],GL[:,i])*ones(n)
        VS[r+r*n+(i-1)*n+1:r+r*n+(i-1)*n+n]=dot(VR[:,i],GL[:,i])*ones(n)
    end
    DEL=-diagm(VS)
    H1=H+DEL
    #----------------------------------------------------------
    Ge=M'*G
    He=M'*H1*M
    N=zero(Ge)
    #Fe=pinv(He)
    #N=-Fe*Ge
    N=-pinv(He)*Ge
    #N1=M*N
    l1=Ge'*Ge
    l2=Ge'*He*Ge
    l=l1/l2
    N2=-l*Ge
    #N2=M*N2
    xi=norm(N)
    pi=norm(N2)

    if xi <= delta
        Ns=N

    elseif xi > delta && pi >= delta
        Ns=-((delta/norm(Ge))*Ge)
    else
        a1=(norm(N-N2))^2
        a2=2*(-(norm(N))^2-2*(norm(N2))^2+3*dot(N,N2))
        a3=-delta^2+4*(norm(N2))^2-4*dot(N,N2)+(norm(N))^2
        tau=solve(a1,a2,a3)
        Ns=N2+(tau-1)*(N-N2)
    end
    Ns=M*Ns
    W1=zeros(r)
    W1=Ns[1:r]
    A=fill(0.0+0.0*im,n*r)
    B=fill(0.0+0.0*im, n, r)
    A=Ns[r+1:r+n*r]+Ns[r+n*r+1:r+2*n*r]*im
    for i in 1:r
        B[:,i]=A[(i-1)*n+1:i*n]
    end
    W2=zeros(r)
    V1=fill(0.0+0.0*im, n, r)
    for i in 1:r
        W2[i]=W[i]+W1[i]
        V1[:,i]=(V[:,i]+B[:,i])/(norm(V[:,i]+B[:,i]))
        if W2[i]<0
            W2[i]=-W2[i]
            V1[:,i]=exp((pi/d)*im)*V1[:,i]
        end
    end

    S=M'*Ns
    w1=0.5*(norm_apolar(hpol(W,V,X,d)-P))^2
    w2=0.5*(norm_apolar(hpol(W2,V1,X,d)-P))^2
    w3=w1+Ge'*S+0.5*S'*He*S
    r1=w1-w2
    r2=w1-w3
    ki=r1/r2
    if ki>=0.2
        op1,op2=W2,V1
    else
        op1,op2=W,V
    end

    al=0.5*(norm_apolar(P))
    t=exp(-14*(ki-1/3))
    er=(1/3+(2/3)*(1/(1+t)))*delta
    if ki > 0.6
        delta=min(2*norm(Ns),al)
    else
        delta=min(er,al)
    end


    delta,op1,op2
end

function rn_tr_mat(P, A0::Vector, B0::Matrix,
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
    E=fill(0.0+0.0im,r)
    F=fill(0.0+0.0im,n,r)
    A0+=fill(0.0im,r)
    B0+=fill(0.0im,n,r)
    P0=tensor(A0,B0,X,d)
    d0=norm_apolar(P-P0)
    C=op(A0,B0,P)
    A1=fill(0.0+0.0im,r)
    B1=fill(0.0+0.0im,n,r)
    B1=B0
    for i in 1:r
        A1[i]=A0[i]*C[i]
    end

    P1=tensor(A1,B1,X,d)
    d1=norm_apolar(P1-P)
    if d0<d1
        A1=A0
    end
    for i in 1:r
        y=abs(A1[i])
        z=angle(A1[i])
        A1[i]=y*norm(B1[:,i])^d
        B1[:,i]=exp((z/d)*im)*(B1[:,i]/norm(B1[:,i]))
    end
    a0=Delta(P,A1)
    De, E, F = rn_tr_mat_step(a0,A1,B1,P)

    W = fill(0.0+0.0im,r)
    V = fill(0.0+0.0im,n,r)
    i = 2
    while  i < N && De > eps
        De, E, F = rn_tr_mat_step(De,E,F,P)
        W, V = E, F
        i += 1
    end
    P4 = tensor(W,V,X,d)
    d2 = norm_apolar(P4-P)
    A=fill(0.0+0.0im,r)
    B=fill(0.0+0.0im,n,r)
    if d2<d1
        A,B=W,V
    else
        A,B=A1,B1
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
