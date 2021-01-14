export sym_step_mat, rns_tr_mat
using LinearAlgebra
using MultivariatePolynomials
using DynamicPolynomials

"""
sym_step_mat(delta, W, V, P) ➡ gives symmetric decomposition W1, V1 of rank r=size(W,1).

Riemannian Newton method with trust region (one iteration) from initial point W, V.

W is a real positif vector and V is a complex matrix and its columns are normalized.

r must be strictly lower than the subgeneric rank

delta is the radius of the trust region

"""

function sym_step_mat(delta, W::Vector, V::Matrix,P)
    X=variables(P)
    r=size(W,1)
    n=size(X,1)
    d=maxdegree(P)
    c= typeof(V[1,1])
    G1=fill(0.0, r)
    G2=fill(zero(c), n,r)
    A=fill(0.0, r, r)
    B=fill(zero(c), n*r, r)
    B1=fill(0.0, 2*n*r,r)
    C=fill(zero(c), n*r, n*r)
    D=fill(zero(c), n*r, n*r)
    for j in 1:r
        G1[j]=sum(W[i]*real((dot(V[:,j],V[:,i]))^d) for i in 1:r)-real(P(conj(V[:,j])))
    end
    for j in 1:r
        u=W[j]*conj(gradeval(P,X,conj(V[:,j])))
        a=0.5*(d*sum(W[j]*W[l]*(dot(V[:,l],V[:,j]))^(d-1)*conj(V[:,l]) for l in 1:r)-u)
        G2[:,j]=a
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
    for i in 1:r
        for j in 1:r
            B1[2*(i-1)*n+1:2*(i-1)*n+n,j]=2*real(B[(i-1)*n+1:i*n, j])
            B1[2*(i-1)*n+n+1:2*(i-1)*n+2n,j]=-2*imag(B[(i-1)*n+1:i*n, j])
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
    CD=fill(0.0,2*n*r,2*n*r)
    for i in 1:r
        for j in 1:r
            CD[2*(i-1)*n+1:2*(i-1)*n+n,2*(i-1)*n+1:2*(i-1)*n+n]=
        2*real(C[(i-1)*n+1:i*n,(j-1)*n+1:j*n]+D[(i-1)*n+1:i*n,(j-1)*n+1:j*n])
            CD[2*(i-1)*n+1:2*(i-1)*n+n,2*(i-1)*n+n+1:2*(i-1)*n+2*n]=
        -2*imag(C[(i-1)*n+1:i*n,(j-1)*n+1:j*n]+D[(i-1)*n+1:i*n,(j-1)*n+1:j*n])
            CD[2*(i-1)*n+n+1:2*(i-1)*n+2*n,2*(i-1)*n+1:2*(i-1)*n+n]=
        2*imag(-C[(i-1)*n+1:i*n,(j-1)*n+1:j*n]+D[(i-1)*n+1:i*n,(j-1)*n+1:j*n])
            CD[2*(i-1)*n+n+1:2*(i-1)*n+2*n,2*(i-1)*n+n+1:2*(i-1)*n+2*n]=
        2*real(-C[(i-1)*n+1:i*n,(j-1)*n+1:j*n]+D[(i-1)*n+1:i*n,(j-1)*n+1:j*n])
        end
    end
    G=zeros(r+2*n*r)
    H=zeros(r+2*n*r,r+2*n*r)
    DEL=zeros(r+2*n*r,r+2*n*r)
    H1=zeros(r+2*n*r,r+2*n*r)
    G[1:r]=G1
    for i in 1:r
    G[r+2*(i-1)*n+1:r+2*(i-1)*n+n]=2*real(G2[:,i])
    G[r+2*(i-1)*n+n+1:r+2*(i-1)*n+2*n]=-2*imag(G2[:,i])
    end
    H[1:r,1:r]=A
    H[r+1:r+2*n*r,1:r]=B1
    H[1:r,r+1:r+2*n*r]=B1'
    H[r+1:r+2*n*r,r+1:r+2*n*r]=CD
    for i in 1:r
        v=zeros(2*n)
        v[1:n]=real(V[:,i])
        v[n+1:2*n]=imag(V[:,i])
        je=G[r+2*(i-1)*n+1:r+2*(i-1)*n+2*n]
        de=diagm(-dot(v,je)*ones(2*n))
        DEL[r+2*(i-1)*n+1:r+2*(i-1)*n+2*n,r+2*(i-1)*n+1:r+2*(i-1)*n+2*n]=de
    end
    H1=H+DEL
    M=zeros(r+2*n*r,r+r*(2*n-1))
    M[1:r,1:r]=Matrix(1.0*I, r, r)
    for i in 1:r
        v=zeros(2*n)
        v[1:n]=real(V[:,i])
        v[n+1:2*n]=imag(V[:,i])
        q,s=qr(Matrix(1.0*I, 2*n, 2*n)-v*v',Val(true))
        M[r+2*(i-1)*n+1:r+2*(i-1)*n+2*n,r+(2*n-1)*(i-1)+1:r+(2*n-1)*(i-1)+(2*n-1)]=q[1:2*n,1:2*n-1]

    end


    Ge=M'*G
    He=M'*H1*M
    N=zero(Ge)
    #Fe=pinv(He)
    #N=-Fe*Ge
    N=-He\Ge
    N1=M*N
    l1=Ge'*Ge
    l2=Ge'*He*Ge
    l=l1/l2
    N2=-l*Ge
    N2=M*N2
    xi=norm(N1)
    pi=norm(N2)

    if xi <= delta
        Ns=N1

    elseif xi > delta && pi >= delta
        Ns=-M*((delta/norm(Ge))*Ge)
    else
        a1=(norm(N1-N2))^2
        a2=2*(-(norm(N1))^2-2*(norm(N2))^2+3*dot(N1,N2))
        a3=-delta^2+4*(norm(N2))^2-4*dot(N1,N2)+(norm(N1))^2
        tau=solve(a1,a2,a3)
        Ns=N2+(tau-1)*(N1-N2)
    end
    W1=zeros(r)
    W1=Ns[1:r]
    B=fill(0.0+0.0*im, n, r)
    for i in 1:r
        B[:,i]=Ns[r+2*(i-1)*n+1:r+2*(i-1)*n+n]+Ns[r+2*(i-1)*n+n+1:r+2*(i-1)*n+2*n]*im
    end
    W2=zeros(r)
    V1=fill(0.0+0.0*im, n, r)
    W2=[abs(W[i]+W1[i]) for i in 1:r]
    for i in 1:r
        V1[:,i]=(V[:,i]+B[:,i])/(norm(V[:,i]+B[:,i]))
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


"""
    rns_tr_mat(P, W0, V0, Info = Dict( "maxIter" => 500, "epsIter" => 1.e-3))

    ➡ gives symmetric decomposition W1, V1 of rank r = length(W0).

    Riemannian Newton loop with trust region starting from initial point W0, V0.

    The default maximal number of iteration is N = 500.

    r must be strictly lower than the generic rank and the interpolation degree must be lower than (d-1)/2 where d is the degree of P.
"""
function rns_tr_mat(P, A0::Vector, B0::Matrix,
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
    De, E, F = sym_step_mat(a0,A1,B1,P)

    W = fill(0.0+0.0im,r)
    V = fill(0.0+0.0im,n,r)
    i = 2
    while  i < N && De > eps
        De, E, F = sym_step_mat(De,E,F,P)
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
