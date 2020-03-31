export Matrixindices, Vandermonde, Weierstrass1
using LinearAlgebra
using MultivariatePolynomials
using DynamicPolynomials

function Matrixindices(B,X)
    n=size(X,1)
    r=size(B,1)
    M=fill(0,n,r)
    for i in 1:r
        for j in 1:n
            M[j,i]=degree(B[i],X[j])
        end
    end
    return M
end
function Vandermonde(A,B)
    r=size(A,2)
    n=size(A,1)
    V=fill(0.0+0.0im,r,r)
    for i in 1:r
        for j in 1:r
            V[i,j]=prod(A[l,i]^(B[l,j]) for l in 1:n)
        end
    end
    return V
end
function Weierstrass1(P,r,B,W,U)
    Y=variables(P)
    X=Y[2:end]
    n=size(X,1)
    d=maxdegree(P)
    F=hpol(W,U,Y,d)-P
    M=Matrixindices(B,X)
    #M=[0 2 1;0 0 1]
    V=fill(0.0+0.0im,n,r)
    V[1:end,:]=U[2:end,:]
    M1=Vandermonde(V,M)
    S=conj(M1)
    E=fill(0.0+0.0im,r,r)
    u,s,v=svd(S[2:r,:])
    E[:,1]=v[:,end]
    a=1\((S[1,:])'*E[:,1])
    E[:,1]=a*E[:,1]
    for i in 2:r-1
        Z=fill(0.0+0.0im,r-1,r)
        Z[1:i-1,:]=S[1:i-1,:]
        Z[i:r-1,:]=S[i+1:r,:]
        u,s,v=svd(Z)
        E[:,i]=v[:,end]
        b=1/((S[i,:])'*E[:,i])
        E[:,i]=b*E[:,i]
    end
    u,s,v=svd(S[1:r-1,:])
    E[:,r]=v[:,end]
    c=1\((S[r,:])'*E[:,r])
    E[:,r]=c*E[:,r]
    N=fill(0.0+0.0im,r*(n+1))
    D=fill(0.0+0.0im,r*(n+1),r*(n+1))
    for i in 1:r
        C=fill(0.0+0.0im,n+1,n+1)
        G=fill(0.0+0.0im,n+1)
        Q=sum(E[l,i]*B[l] for l in 1:r)
        A=gradeval(Q,X,conj(V[:,i]))
        L=0.5*W[i]*Matrix((1.0+0.0im)*I,n,n)+V[:,i]*adjoint(A)
        C[2:n+1,2:n+1]=L
        C[:,1]=0.5*U[:,i]
        C[1,2:n+1]=W[i]*conj(A)
        p,m,f=svd(C)
        C=C+r^2*d*m[1]*Matrix((1.0+0.0im)*I,n+1,n+1)
        H=adjoint(C)*C
        #H=H+d^10*Matrix((1.0+0.0im)*I,n+1,n+1)
        G[1]=F(conj(U[:,i]))
        G[2:n+1]=conj(W[i])*gradeval(F,X,conj(U[:,i]))
        N[(i-1)*(n+1)+1:i*(n+1)]=-H\G
        D[(i-1)*(n+1)+1:i*(n+1),(i-1)*(n+1)+1:i*(n+1)]=H
    end
    W1=fill(0.0+0.0im,r)
    V1=fill(0.0+0.0im,n,r)
    for i in 1:r
        W1[i]=N[(i-1)*(n+1)+1]
        V1[:,i]=N[(i-1)*(n+1)+2:i*(n+1)]
    end
    W2=W+W1
    V2=V+V1
    U2=fill(0.0+0.0im,n+1,r)
    for i in 1:r
        U2[1,i]=1
        U2[2:end,i]=V2[:,i]
    end
    return W2,U2
end
