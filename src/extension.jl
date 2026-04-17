function cst(p)
    i = findfirst(t-> maxdegree(t)==0, p.x)
    return ( i != nothing ? p.a[i] : zero(coefficient_type(p)))
end

export degext_series
"""
   s, Y = degext_series(F,X0,d)

Compute the apolar_series extending F to the degree  `d`, shifting F by `X0^(d-d0)` where `d0=maxdegree(F)` and `X0` is a variable of `F`. 
The series `s` is a series in the dual monomials of `variables(F)`, and with coefficients which are polynomials in the pseudo-moment variables `Y`. 
 
"""
function degext_series(F, X0, d::Int64)

    d0 = maxdegree(F)
    X = variables(F)
    n = length(X)

    C = promote_type(coefficient_type(F),Rational{Int})
    
    @assert(d > d0)

    L0 = [m for m in DynamicPolynomials.monomials(F)]
    C0 = DynamicPolynomials.coefficients(F)
    Lm = union(X0^(d-d0)*L0, DynamicPolynomials.monomials(X,d))

    l = length(Lm)-length(L0)

    # the unknown moments
    Y = (@polyvar y[1:l] monomial_order = Graded{LexOrder})[1]

    Zero = zero(Y[1]+zero(C))
    Lc = fill(Zero, length(Lm))

    l0 = length(C0)
    for i in 1:l0
        Lc[i] = C(C0[i])/binomial(d0,exponents(L0[i])) 
    end
    for i in 1:l
        Lc[l0+i] = Y[i]
    end
    AlgebraicSolvers.series(Lc,Lm), Y
end

export dimext_series
"""

   s, Xe, Y = dimext_series(F,r)

Compute the apolar_series extending  the variables of `F`  to the size `r`.
The series `s` is a series in the dual monomials of the new variables `Xe`, and with coefficients which are polynomials in the pseudo-moment variables `Y`. 

"""
function dimext_series(F, r::Int64)
    d = maxdegree(F)
    X = variables(F)
    n = length(X)

    @assert(n < r)

    Xn = (@polyvar xe[1:r-n] monomial_order = ordering(F))[1]

    Xe = vcat(X, Xn)
    
    s = apolar_dual(F)
    Ln = union([xn*DynamicPolynomials.monomials(Xe,d-1) for xn in Xn]...)
    Lm = vcat(DynamicPolynomials.monomials(X,d), Ln)

    # the unknown moments
    Y = (@polyvar y[1:length(Ln)] monomial_order = Graded{LexOrder})[1]

    Zero = Y[1]+0.0
    Lc = fill(zero(Y[1]+0.0), length(Lm))

    cnt = 1
    for i in 1:length(Lm)
        Lc[i] = get(s.terms, Lm[i], Y[cnt])
        if maxdegree(Lc[i])>0
            cnt+=1
        end
    end
    AlgebraicSolvers.series(Lc,Lm), Xe, Y
end


export eval_series
"""
   Substitute in the parametric coefficients of the series, the parameters y by the substitution `sbs`
"""
function eval_series(s::Series, sbs)
    Lm = collect(monomials(s))
    Lc = collect(coefficients(s))
    
    Ls = fill(zero(sbs.second[1]),length(Lc))
    for i in 1:length(Lc)
        Ls[i] = coefficient(subs(Lc[i], sbs), one(variables(sbs.first)[1]))
    end
    return AlgebraicSolvers.series(Ls,Lm)
end
