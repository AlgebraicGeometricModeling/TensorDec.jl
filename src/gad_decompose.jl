using LinearAlgebra, DynamicPolynomials, AlgebraicSolvers



function AlgebraicSolvers.mult_matrices(s::AlgebraicSolvers.Series{C,D}, rkf::Function=eps_rkf(1.e-4)) where {C,D}
    d  = maxdegree(s)
    X = variables(s)
    d0 = div(d-1,2)
    d1 = d-d0-1
    B0 = reverse(DynamicPolynomials.monomials(X, 0:d0))
    B1 = reverse(DynamicPolynomials.monomials(X, 0:d1))
  
    H=Matrix[]
    for x in X
        push!(H, hankel(s, B0, [b*x for b in B1]))
    end

    h = length(H)
    Hrnd = sum(H[i]*rand(Float64) for i in 1:h)
    U,S,V = svd(sum(H[i]*randn(h)[i] for i in 1:h))
    r = rkf(S)
    Sr  = S[1:r]
    Sri = LinearAlgebra.diagm([one(Sr[1])/S[i] for i in 1:r])

    M = Matrix{}[]
    for i in 1:h
    	push!(M, Sri*(U[:,1:r]')*H[i]*(V[:,1:r]))
    end
    return M
end

function LinearAlgebra.norm(P::DynamicPolynomials.Polynomial)
    LinearAlgebra.norm(DynamicPolynomials.coefficients(P))
end

export reconstruct
function reconstruct(W,L,d)
    sum(W[i]*L[i]^(d-maxdegree(W[i])) for i in 1:length(L))
end

function cluster(v)
    ms=Int64[]
    r=size(v)[1]
    m=1
    for i in 1:r-1 
        if norm(v[i]-v[i+1])>1.0e-2
            push!(ms,m)
            m=1
        else
            m+=1
        end
    end
    push!(ms,m)
    return ms 
end


function _multiplicities(M)
    while true
        Mrnd = sum(M[i] * randn(Float64) for i in 1:length(M))
        mrnd = sum(M[i] * randn(Float64) for i in 1:length(M))
	
        T, Z, E = schur(Mrnd)
        t, z, e = schur(mrnd)

        MS = cluster(E)
        ms = cluster(e)

        if MS == ms
            return ms, z, size(Mrnd,1)
        end
    end
end



function _local_mult_matrices(Tr,ms)
    t=length(Tr)
    s=length(ms)
    Subb=[]
    Subm=[]
    l=0
    for q in 1:s
        for i in 1:t
            push!(Subm, Tr[i][l+1:l+ms[q],l+1:l+ms[q]])
        end
        l+=ms[q]
    end
    o=0
    for i in 1:s 
        push!(Subb, Subm[o+1:o+t])
        o+=t     
    end 
    return Subb
end

function local_mult_matrices(Tr, ms)
    [[m[I,I] for I in ms] for m in Tr]
end

function nilindex(A::Matrix, v; tol=1.e-7,max_iter=1000)
    iter=0
    while norm(v) > tol && iter < max_iter
        v = A*v
        iter+=1
    end
    return iter,v
end



function nil_index(Subb::Vector, Pt::Vector)

    for i in 1:length(Subb)
        for j in 1:length(Subb[i])
            Subb[i][j] = Subb[i][j]-Pt[i][j]*I(size(Subb[i][j])[1])
        end
    end

    for j in 1:length(Subb)
        Subb[j]=sum(Subb[j][i]*randn(Float64) for i in 1:length(Subb[j]))
    end
    
    nilx=[]
    
    for k in 1:length(Subb)
        v=rand(size(Subb[k])[1])
        ns,v=nilindex(Subb[k],v)
        nilx = vcat(ns,nilx)
    end
    return reverse(nilx)
end

function nil_index(Subb::Vector, Xi::Matrix)

    for i in 1:length(Subb)
        for j in 1:length(Subb[i])
            Subb[i][j] = Subb[i][j]-Xi[:,i][j]*I(size(Subb[i][j])[1])
        end
    end

    for j in 1:length(Subb)
        Subb[j]=sum(Subb[j][i]*randn(Float64) for i in 1:length(Subb[j]))
    end
    
    nilx=[]
    
    for k in 1:length(Subb)
        v=rand(size(Subb[k])[1])
        ns,v=nilindex(Subb[k],v)
        nilx = vcat(ns,nilx)
    end
    return reverse(nilx)
end


export gad_decompose
"""
this function computes the generalized additive decomposition of symmetric forms
"""
function gad_decompose(F; verbose = false)
    
    X = variables(F)
    d = maxdegree(F)
    n = length(variables(F))-1
    verbose && println("--- d=", d, " n=", n+1)
    
    sigma = apolar_dual(F) 
    M = AlgebraicSolvers.mult_matrices(sigma)


    ms, Z, r = _multiplicities(M)
    Zt = transpose(Z)
    Tr = [Zt*M[i]*Z for i in 1:length(M)]

    Xi1, ms1, Z1, Tr1 = AlgebraicSolvers.schur_dcp(M, 1.e-3)

    verbose && println("--- multiplicities: ", ms, " ", ms1)


    Subb = _local_mult_matrices(Tr, ms)
    Subb1 = local_mult_matrices(Tr1, ms1)

   
    nPt = size(Subb)[1]
    c   = size(Subb[1])[1]

    Pt = []
    Xi = []
    for i in 1:nPt
        for j in 1:size(Subb[i])[1]
            push!(Xi,tr(Subb[i][j])/size(Subb[i][j],1))
        end
    end
    
    for i in 1:c:length(Xi)
        push!(Pt, Xi[i:i+c-1])
    end

    println(Pt, " ", Xi1 )

    nilx = _nil_index(Subb, Pt)
    nilx = nil_index(Subb1, Xi1)

    verbose && println("--- nil indices: ", nilx)
    dg = vcat([nilx[i]-1 for i in 1:length(nilx)]...)

    L=[]

    for i in 1:length(Pt)
        push!(L,dot(X,Pt[i]))
    end
   
    D = DynamicPolynomials.monomials(X,d)

    mlt = [DynamicPolynomials.monomials(X,dg[i]) for i in 1:length(dg)]

    P = vcat([L[i]^(d-dg[i])*mlt[i] for i in 1:length(Pt)]...)

    Vdm = matrixof(P,D)'
    b = matrixof([F],D)'

    
    ws = Vdm\b
   
   if dg == zeros(length(dg))
       return [one(F)*w for w in ws], L,ms
    else
   
       W = []; 

       s = 0
       for i in 1:length(Pt)
           l = length(mlt[i])
           w = dot(mlt[i], ws[s+1:s+l])
           push!(W, dot(mlt[i], ws[s+1:s+l]))
           s+=l
       end
       return W, L, ms
   end
end

