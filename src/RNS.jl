export gradeval, hessianeval, hpol, cnewton11, op, RNS_SPED, RNS_R, RNS_C
using LinearAlgebra
using MultivariatePolynomials
using TensorDec
using DynamicPolynomials

function gradeval(F,X,a)
        [DynamicPolynomials.differentiate(F, X[i])(a) for i in 1:length(X)]
end

function hessianeval(F,X,a)
    n=length(a)
    v=fill(0.0+0.0im, (n,n))
    for i in 1:n
        p=DynamicPolynomials.differentiate(F, X[i])
        v[i,:]=transpose([DynamicPolynomials.differentiate(p,X[j])(a) for j in 1:n])
    end
    v
end
function hpol(W,A,X,d)
    P = sum(W[i]*(transpose(A[:,i])*X)^d for i in 1:length(W))
end
function cnewton11(W::Vector, V::Matrix,P)
    r=size(W,1)
    n=size(V,1)
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
Ge=M'*G
He=M'*H*M
Fe=pinv(He)
N=-Fe*Ge
N1=M*N
W1=zeros(r)
W1=N1[1:r]
A=fill(0.0+0.0*im,n*r)
B=fill(0.0+0.0*im, n, r)
A=N1[r+1:r+n*r]+N1[r+n*r+1:r+2*n*r]*im
for i in 1:r
    B[:,i]=A[(i-1)*n+1:i*n]
end
W2=zeros(r)
V1=fill(0.0+0.0*im, n, r)
for i in 1:r
    tg=(W1[i]+W[i])*(transpose(V[:,i])*X)^d+d*W[i]*(((transpose(V[:,i])*X)^(d-1))*(transpose(B[:,i])*X))
    #println(tg)
    U=transpose(hankel(tg,1))
    u,s,v=svd(U)
    a=(W1[i]+W[i])*((dot(u[:,1],V[:,i]))^d)+d*W[i]*((dot(u[:,1],V[:,i]))^(d-1))*(dot(u[:,1],B[:,i]))
    V1[:,i]=u[:,1]*exp((angle(a)/d)*im)
    W2[i]=abs(a)
end

return W2,V1,Ge
end
function op(W::Vector, V::Matrix,P)
    r=size(W,1)
    d=maxdegree(P)
    A=fill(0.0+0.0im,r,r)
    B=fill(0.0+0.0im,r)
    for i in 1:r
        A[i,:]=[conj(W[i])*W[j]*(dot(V[:,i],V[:,j]))^d for j in 1:r]
    end
    for i in 1:r
        B[i]=conj(W[i])*P(conj(V[:,i]))
    end
    C=A\B
end

function RNS_SPED(P,r,N)
    d = maxdegree(P)
    X = variables(P)
    n=size(X,1)
    A0, B0 = decompose(P,cst_rkf(r))
    E=fill(0.0+0.0im,N*r)
    F=fill(0.0+0.0im,n,N*r)
    Ge=fill(0.0,N*(r+(2*n-1)*r))
    A0+=fill(0.0im,r)
    B0+=fill(0.0im,n,r)
    P0=hpol(A0,B0,X,d)
    d0=norm(P-P0)
    C=op(A0,B0,P)
    A1=fill(0.0+0.0im,r)
    B1=fill(0.0+0.0im,n,r)
    B1=B0
      for i in 1:r
          A1[i]=A0[i]*C[i]
      end
        P1=hpol(A1,B1,X,d)
        d1=norm(P1-P)
        for i in 1:r
          y=abs(A1[i])
          z=angle(A1[i])
           A1[i]=y*norm(B1[:,i])^d
           B1[:,i]=exp((z/d)*im)*(B1[:,i]/norm(B1[:,i]))
          end
        E[1:r], F[1:n,1:r], Ge[1:r+(2n-1)*r] = cnewton11(A1,B1,P)
        W=fill(0.0+0.0im,r)
        V=fill(0.0+0.0im,n,r)
        i = 2
        @time(while norm(Ge[(i-2)*(r+(2*n-1)*r)+1:(i-1)*(r+(2*n-1)*r)])>1.e-1 && i<N
        E[(i-1)*r+1:i*r], F[1:n,(i-1)*r+1:i*r], Ge[(i-1)*(r+(2*n-1)*r)+1:i*(r+(2*n-1)*r)]=cnewton11(E[(i-2)*r+1:(i-1)*r],F[1:n,(i-2)*r+1:(i-1)*r],P)
        W,V=E[(i-1)*r+1:i*r], F[1:n,(i-1)*r+1:i*r]
        i += 1
    end)
    P4=hpol(W,V,X,d)
    d2=norm(P4-P)
    A=fill(0.0+0.0im,r)
    B=fill(0.0+0.0im,n,r)
    if d2<d1
        A,B=W,V
    else A,B=A1,B1
    end
    P5=hpol(A,B,X,d)
    d3=norm(P-P5)
    println("N:",i)
    println("d0:",d0)
    println("d1:",d1)
    println("d3:",d3)

    return A,B
    end
    function RNS_R(P,r,N)
        d = maxdegree(P)
        X = variables(P)
        n=size(X,1)
        A0=rand(Float64,r)
        B0 = rand(Float64,n,r)
        E=fill(0.0+0.0im,N*r)
        F=fill(0.0+0.0im,n,N*r)
        Ge=fill(0.0,N*(r+(2*n-1)*r))
        A0+=fill(0.0im,r)
        B0+=fill(0.0im,n,r)
        P0=hpol(A0,B0,X,d)
        d0=norm(P-P0)
        C=op(A0,B0,P)
        A1=fill(0.0+0.0im,r)
        B1=fill(0.0+0.0im,n,r)
        B1=B0
          for i in 1:r
              A1[i]=A0[i]*C[i]
          end
            P1=hpol(A1,B1,X,d)
            d1=norm(P1-P)
            for i in 1:r
              y=abs(A1[i])
              z=angle(A1[i])
               A1[i]=y*norm(B1[:,i])^d
               B1[:,i]=exp((z/d)*im)*(B1[:,i]/norm(B1[:,i]))
              end
            E[1:r], F[1:n,1:r], Ge[1:r+(2n-1)*r] = cnewton11(A1,B1,P)
            W=fill(0.0+0.0im,r)
            V=fill(0.0+0.0im,n,r)
            i = 2
            @time(while norm(Ge[(i-2)*(r+(2*n-1)*r)+1:(i-1)*(r+(2*n-1)*r)])>1.e-1 && i<N
            E[(i-1)*r+1:i*r], F[1:n,(i-1)*r+1:i*r], Ge[(i-1)*(r+(2*n-1)*r)+1:i*(r+(2*n-1)*r)]=cnewton11(E[(i-2)*r+1:(i-1)*r],F[1:n,(i-2)*r+1:(i-1)*r],P)
            W,V=E[(i-1)*r+1:i*r], F[1:n,(i-1)*r+1:i*r]
            i += 1
        end)
        P4=hpol(W,V,X,d)
        d2=norm(P4-P)
        A=fill(0.0+0.0im,r)
        B=fill(0.0+0.0im,n,r)
        if d2<d1
            A,B=W,V
        else A,B=A1,B1
        end
        P5=hpol(A,B,X,d)
        d3=norm(P-P5)
        println("N:",i)
        println("d0:",d0)
        println("d1:",d1)
        println("d3:",d3)

        return A,B
        end
        function RNS_C(P,r,N)
            d = maxdegree(P)
            X = variables(P)
            n=size(X,1)
            A0=rand(ComplexF64,r)
            B0 =rand(ComplexF64,n,r)
            E=fill(0.0+0.0im,N*r)
            F=fill(0.0+0.0im,n,N*r)
            Ge=fill(0.0,N*(r+(2*n-1)*r))
            P0=hpol(A0,B0,X,d)
            d0=norm(P-P0)
            C=op(A0,B0,P)
            A1=fill(0.0+0.0im,r)
            B1=fill(0.0+0.0im,n,r)
            B1=B0
              for i in 1:r
                  A1[i]=A0[i]*C[i]
              end
                P1=hpol(A1,B1,X,d)
                d1=norm(P1-P)
                for i in 1:r
                  y=abs(A1[i])
                  z=angle(A1[i])
                   A1[i]=y*norm(B1[:,i])^d
                   B1[:,i]=exp((z/d)*im)*(B1[:,i]/norm(B1[:,i]))
                  end
                E[1:r], F[1:n,1:r], Ge[1:r+(2n-1)*r] = cnewton11(A1,B1,P)
                W=fill(0.0+0.0im,r)
                V=fill(0.0+0.0im,n,r)
                i = 2
                @time(while norm(Ge[(i-2)*(r+(2*n-1)*r)+1:(i-1)*(r+(2*n-1)*r)])>1.e-1 && i<N
                E[(i-1)*r+1:i*r], F[1:n,(i-1)*r+1:i*r], Ge[(i-1)*(r+(2*n-1)*r)+1:i*(r+(2*n-1)*r)]=cnewton11(E[(i-2)*r+1:(i-1)*r],F[1:n,(i-2)*r+1:(i-1)*r],P)
                W,V=E[(i-1)*r+1:i*r], F[1:n,(i-1)*r+1:i*r]
                i += 1
            end)
            P4=hpol(W,V,X,d)
            d2=norm(P4-P)
            A=fill(0.0+0.0im,r)
            B=fill(0.0+0.0im,n,r)
            if d2<d1
                A,B=W,V
            else A,B=A1,B1
            end
            P5=hpol(A,B,X,d)
            d3=norm(P-P5)
            println("N:",i)
            println("d0:",d0)
            println("d1:",d1)
            println("d3:",d3)

            return A,B
            end
