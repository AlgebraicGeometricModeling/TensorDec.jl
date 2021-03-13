using LinearAlgebra
using MultivariatePolynomials
using DynamicPolynomials
include("sym_tens_fct.jl")
include("decompose.jl")
include("ahp.jl")
include("apolar.jl")
function rqn_ver_c_step(P,V)
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

#test1
function test1(n,r,d,N)
X = (@polyvar x[1:n])[1]
V1=rand(ComplexF64,n,r)
V2=zeros(ComplexF64,n,r)
V2=V1+1.e-1*rand(ComplexF64,n,r)
P1=sum((transpose(V1[:,i])*X)^d for i in 1:r)
P2=sum((transpose(V2[:,i])*X)^d for i in 1:r)
println("err0:",norm_apolar(P1-P2))

V3=rqn_ver_c_step(P2,V1)
P3=sum((transpose(V3[:,i])*X)^d for i in 1:r)
println("err1:",norm_apolar(P3-P2))
i=1
while i < N
    V3=rqn_ver_c_step(P2,V3)
    P3=sum((transpose(V3[:,i])*X)^d for i in 1:r)
    println("err",i,":",norm_apolar(P3-P2))
i +=1
end
end

#test2
function test2(n,r,d,N)
X = (@polyvar x[1:n])[1]
Pper=randc_sym_tens(X, d)
V=rand(ComplexF64,n,r)
P=sum((transpose(V[:,i])*X)^d for i in 1:r)
P1=P+1.e-3*Pper
println("err0:",norm_apolar(P1-P))

V3=rqn_ver_c_step(P1,V)
P3=sum((transpose(V3[:,i])*X)^d for i in 1:r)
println("err1:",norm_apolar(P3-P1))
i=1
while i < N
    V3=rqn_ver_c_step(P1,V3)
    P3=sum((transpose(V3[:,i])*X)^d for i in 1:r)
    println("err",i,":",norm_apolar(P3-P1))
i +=1
end
end
