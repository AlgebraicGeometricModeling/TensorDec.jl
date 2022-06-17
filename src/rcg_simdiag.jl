export rcg_simdiag
using LinearAlgebra

# norm of off diagonal terms of a square matrix
function norm_off(M)
    if size(M[1],1)>1
        return sqrt(sum(abs2(M[i,j]) + abs2(M[j,i]) for i in 1:size(M,1) for j in i+1:size(M,1)))
    else
        return 0.0
    end
end


# add code rcg_simdiag

function diag_multip(A,B)
    n=size(A,1)
    C=zeros(n,n)
    for i in 1:n
        C[i,i]=dot(A[i,:],B[:,i])
    end
    return C
end
##############################################
# retraction on GL_n at B using the exponential map according to the right metric
function expo(B,Bi,Xi) #Bi is the inverse of B
    l=Xi*Bi
    n=size(B,1)
    l1=zeros(n,n)
    for i in 1:n
        for j in 1:n
            l1[i,j]=l[j,i]
        end
    end
    l2=l-l1
    ex1=exp(l2)
    ex2=exp(l1)
    Ex=ex1*ex2*B
    EO=((diag_multip(Ex,Ex'))^(-0.5))*Ex
    return EO
end
################################################
function lie_bracket(A,B)
    return A*B-B*A
end
# The objective function to minimize
function obj_E(M,E,Ei) #Ei and Fi are respectivelly the inverse of E and F
    n=size(M[1],1)
    r=length(M)
    O=0.0
    for i in 1:r
        K=E*M[i]*Ei
        L=M[i]-Ei*diagm(diag(K))*E
        O=O+0.5*(norm(L))^2
    end
    return O
end
####################################################
# Conjugate gradient algorithm step 0
function conj_grad_step_zero(M,E) # M is a pencil of matrices
    r=length(M)
    n=size(M[1],1)
    Ei=inv(E)
    a=E*E'
    c=inv(a)

    #compute the gradient in the oblique manifold equiped with right-invariant metric
    J1=zeros(n,n)

    for i in 1:r
        K=E*M[i]*Ei
        Q=c*(K-diagm(diag(K)))*a
        J1=J1+2*(lie_bracket(Q,diagm(diag(K)))+lie_bracket(K',diagm(diag(Q))))*Ei'
    end


D1=inv(diag_multip(a,a))
JO1=J1-diag_multip(J1,E')*D1*a*E

# Define the direction
X1=-JO1


# Armijo backtracking and define the new iterate
e1=inv(E)
O1=obj_E(M,E,e1)
E1=expo(E,e1,X1)
while isnan(E1[1,1]) == true
    X1=0.5*X1
    E1=expo(E,e1,X1)
end
C=sort([abs(eigvals(E1)[i]) for i in 1:n])[1]
while C < 1.e-15
    X1=0.5*X1
    E1=expo(E,e1,X1)
    C=sort([abs(eigvals(E1)[i]) for i in 1:n])[1]
end

sigma3=tr((X1*e1)*(JO1*e1)')
E1i=inv(E1)
O2=obj_E(M,E1,E1i)
S=O1-O2
gama=-0.0001*sigma3
m=0
while S < (0.5^m)*gama && m<51
    m=m+1
    X1=(0.5^m)*X1
    E1=expo(E,e1,X1)
    E1i=inv(E1)
    O2=obj_E(M,E1,E1i)
    S=O1-O2
end
return E1,X1, JO1

end

# Conjugate gradient algo step i
function grad_conj_step(M,E,Ep,Jp1,Xp1)
    #Ep, Fp are the points, Jp1, Jp2 are the gradients, Xp1, Xp2 are the directions of the previous step i-1.
    r=length(M)
    n=size(M[1],1)
    a=E*E'
    Ei=inv(E)
    c=inv(a)
    #compute the gradient
    J1=zeros(n,n)
    for i in 1:r
        K=E*M[i]*Ei
        Q=c*(K-diagm(diag(K)))*a
        J1=J1+2*(lie_bracket(Q,diagm(diag(K)))+lie_bracket(K',diagm(diag(Q))))
    end
J1=J1*Ei'
D1=inv(diag_multip(a,a))
JO1=J1-diag_multip(J1,E')*D1*a*E

# Compute a conjugate direction

e=inv(Ep)
e1=inv(E)
a1=Jp1*e
b1=Xp1*e
TJp1=(a1-diag_multip(a1,a)*D1*a)*E
TXp1=(b1-diag_multip(b1,a)*D1*a)*E
delta1=JO1-TJp1
sigma1=tr((delta1*e1)*(JO1*e1)')
sigma2=tr((Jp1*e)*(Jp1*e)')
beta=max(0,sigma1/sigma2)
X1=-JO1+beta*TXp1



# Apply Armijo backtracking+obtain + use retraction to define the new points
O1=obj_E(M,E,e1)
E1=expo(E,e1,X1)

while isnan(E1[1,1])==true
    X1=0.5*X1
    E1=expo(E,e1,X1)
end

C=sort([abs(eigvals(E1)[i]) for i in 1:n])[1]
while C < 1.e-6
    X1=0.5*X1
    E1=expo(E,e1,X1)
    C=sort([abs(eigvals(E1)[i]) for i in 1:n])[1]
end

sigma6=tr((X1*e1)*(JO1*e1)')
gama=-0.0001*sigma6
E1i=inv(E1)
O2=obj_E(M,E1,E1i)
S=O1-O2
m=0
while S < 0.5^m*gama && m<51
    m=m+1
    E1=expo(E,e1,0.5^m*X1)
    E1i=inv(E1)
    O2=obj_E(M,E1,E1i)
    S=O1-O2
end
return E1,E,JO1,X1
end
##############################################################
function rcg_simdiag_alg(M,threshold=1.1*1.e-5,N_max=2000) # n size of matrices, s number of matrices and N maximal iterations
    n=size(M[1],1)
	E=rand(n,n)
    E=((diag_multip(E,E'))^(-0.5))*E
    err0=obj_E(M,E,inv(E))
    println("err0:",err0);
    E1, X1, JO1=conj_grad_step_zero(M,E)
    err1=obj_E(M,E1,inv(E1))
    grad1=norm(JO1)
    println("err1:",err1);
    println("grad1:",grad1)
    err2=Float64[]
    err2= push!(err2,err0)
    err2= push!(err2,err1)

    i=2
    while  grad1 > threshold && i<N_max
        E1,E,JO1,X1=grad_conj_step(M,E1,E,JO1,X1)
        err2=push!(err2,obj_E(M,E1,inv(E1)))
        grad1=norm(JO1)
        println("err",i,":",err2[i]);
        println("grad",i,":",grad1);
        i+=1
    end

println(grad1)
l=length(M)
Xi=zeros(l,n)
for i in 1:length(M)
	Xi[i,:]=reshape(diag(E1*M[i]*inv(E1)),1,n)
end
return inv(E1), Xi

end


#------------------------------------------------------------------------
# Simultaneous Diagonalisation of the pencil of matrices
function rcg_simdiag(H::Vector{Matrix{C}}, lambda::Vector, rkf::Function) where C
    n = length(H)

    H0 = sum(H[i]*lambda[i] for i in 1:length(lambda))

    U, S, V = svd(H0)       # H0= U*diag(S)*V'
    r = rkf(S)

    Sr  = S[1:r]
    Sri = diagm([one(C)/S[i] for i in 1:r])

    M = Matrix{C}[]
    for i in 1:length(H)
    	push!(M, Sri*(U[:,1:r]')*H[i]*(V[:,1:r]))
    end

    if r > 1
        E, Xi = rcg_simdiag_alg(M)
    else
        Xi = fill(zero(C),n,r)
        for i in 1:n
            Xi[i,1] = M[i][1,1]
        end
        E  = fill(1.0,1,1)
        #DiagInfo = Dict{String,Any}( "case" => "1x1" )
    end

    Uxi = (U[:,1:r].*Sr')*E
    Vxi = (E\ V[:,1:r]')

    #Info = Dict{String,Any}( "diagonalization" => DiagInfo)
    return Xi, Uxi, Vxi
end
