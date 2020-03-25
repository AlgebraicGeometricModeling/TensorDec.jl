export opt, symr, Sym_R
using LinearAlgebra
using MultivariatePolynomials
using DynamicPolynomials

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

function symr(delta, W::Vector, A::Matrix,P)
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
    #Z=zeros(r+n*r,r+n*r)
    #L=zeros(n*r,r)
    #for i in 1:r
        #L[(i-1)*n+1:i*n,i]=d*sum(W[l]*(dot(A[:,i],A[:,l]))^(d-1)*A[:,l] for l in 1:r)-gradeval(P,X,A[:,i])
    #end
    #Z[r+1:end,1:r]=L
    #Z[1:r,r+1:end]=L'
    #for i in 1:r
        #h=hessianeval(P,X,A[:,i])
        #Z[r+(i-1)*n+1:r+i*n,r+(i-1)*n+1:r+i*n]=W[i]*(d*(d-1)sum(W[l]*(dot(A[:,i],A[:,l]))^(d-2)*A[:,l]*A[:,l]' for l in 1:r)-h)
    #end
    #Mat=Mat+Z
    H=Q'*Mat*Q
    N1=-pinv(H)*G
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
            tau=solve(a1,a2,a3)
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
        tg=(W1[i]+W[i])*(transpose(A[:,i])*X)^d+d*W[i]*(((transpose(A[:,i])*X)^(d-1))*(transpose(A1[:,i])*X))
        #println(tg)
        V=transpose(hankel(tg,1))
        u,s,v=svd(V)
        A2[:,i]=u[:,1]
        W2[i]=(W1[i]+W[i])*(dot(A2[:,i],A[:,i]))^d+d*W[i]*((dot(A2[:,i],A[:,i]))^(d-1))*(dot(A2[:,i],A1[:,i]))
    end
    S=Q'*Ns
            w1=0.5*(norm(hpol(W,A,X,d)-P))^2
            w2=0.5*(norm(hpol(W2,A2,X,d)-P))^2
            w3=w1+G'*S+0.5*S'*H*S
            r1=w1-w2
            r2=w1-w3
            ki=r1/r2
            if ki>=0.2
                op1,op2=W2,A2
            else
               op1,op2=W,A
            end

            al=0.5*(norm(P))
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
 Newton-Riemman loop starting from a random decomposition
"""
function symr_iter(P, r, N::Int64=500)
    d = maxdegree(P)
    X = variables(P)
    n=size(X,1)
    A0 = rand(Float64,r)
    B0 = rand(Float64,n,r)
    #A0, B0 = decompose_qr(P,cst_rkf(r))
    De=fill(0.0,N)
    E=fill(0.0,N*r)
    F=fill(0.0,n,N*r)
    P0=hpol(A0,B0,X,d)
    d0=norm(P-P0)
    C=opt(A0,B0,P)
    A1=fill(0.0,r)
    B1=fill(0.0,n,r)
    B1=B0
      for i in 1:r
          A1[i]=A0[i]*C[i]
      end
        P1=hpol(A1,B1,X,d)
        d1=norm(P1-P)
        for i in 1:r
           A1[i]=A1[i]*norm(B1[:,i])^d
           B1[:,i]=B1[:,i]/norm(B1[:,i])
          end
        a0=Delta(P,A1)
        De[1], E[1:r], F[1:n,1:r] = symr(a0,A1,B1,P)
        W=fill(0.0,r)
        V=fill(0.0,n,r)
        i = 2
         @time(while  i < N && De[i-1] > 1.e-3
        De[i], E[(i-1)*r+1:i*r], F[1:n,(i-1)*r+1:i*r]=symr(De[i-1],E[(i-2)*r+1:(i-1)*r],F[1:n,(i-2)*r+1:(i-1)*r],P)
        W,V=E[(i-1)*r+1:i*r], F[1:n,(i-1)*r+1:i*r]
        i += 1
    end)
    P4=hpol(W,V,X,d)
    d2=norm(P4-P)
    A=fill(0.0,r)
    B=fill(0.0,n,r)
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
function symr_iter(P, A0, B0, N::Int64=500)
    d = maxdegree(P)
    X = variables(P)
    r=size(A0,1)
    n=size(X,1)
    De=fill(0.0,N)
    E=fill(0.0,N*r)
    F=fill(0.0,n,N*r)
    P0=hpol(A0,B0,X,d)
    d0=norm(P-P0)
    C=opt(A0,B0,P)
    A1=fill(0.0,r)
    B1=fill(0.0,n,r)
    B1=B0
      for i in 1:r
          A1[i]=A0[i]*C[i]
      end
        P1=hpol(A1,B1,X,d)
        d1=norm(P1-P)
        for i in 1:r
           A1[i]=A1[i]*norm(B1[:,i])^d
           B1[:,i]=B1[:,i]/norm(B1[:,i])
          end
        a0=Delta(P,A1)
        De[1], E[1:r], F[1:n,1:r] = symr(a0,A1,B1,P)
        W=fill(0.0,r)
        V=fill(0.0,n,r)
        i = 2
         @time(while  i < N && De[i-1] > 1.e-3
        De[i], E[(i-1)*r+1:i*r], F[1:n,(i-1)*r+1:i*r]=symr(De[i-1],E[(i-2)*r+1:(i-1)*r],F[1:n,(i-2)*r+1:(i-1)*r],P)
        W,V=E[(i-1)*r+1:i*r], F[1:n,(i-1)*r+1:i*r]
        i += 1
    end)
    P4=hpol(W,V,X,d)
    d2=norm(P4-P)
    A=fill(0.0,r)
    B=fill(0.0,n,r)
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
