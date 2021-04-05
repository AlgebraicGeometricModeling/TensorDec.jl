using LinearAlgebra, DynamicPolynomials
include("../src/apolar.jl")
function rand_sym_tens(X, d::Int64)
    L = monomials(X,d)
    c = randn(length(L))
    T = sum(c[i]*L[i]*binomial(d, exponents(L[i])) for i in 1:length(L))
    T/norm_apolar(T)
end
function randc_sym_tens(X, d::Int64)
    L = monomials(X,d)
    c = rand(ComplexF64,length(L))
    T = sum(c[i]*L[i]*binomial(d, exponents(L[i])) for i in 1:length(L))
    T/norm_apolar(T)
end


function rand_sym_tens_C(X, d::Int64)
    L = monomials(X,d)
    c = randn(Complex{Float64},length(L))
    T = sum(c[i]*L[i]*binomial(d, exponents(L[i])) for i in 1:length(L))
    T/norm_apolar(T)
end

function idx_sum(v::Vector)
    sum([v[i]*i for i in 1:length(v)])
end

function sym_tensor_id(X, d::Int64=3)
    L = monomials(X,d)
    T = sum(m*idx_sum(exponents(m)) for m in L)
    return T
end

function sym_tensor_sin(X, d::Int64=3)
    L = monomials(X,d)
    T = sum(m*sin(idx_sum(exponents(m))) for m in L)
    return T
end

function sym_tensor_sqrt(X, d::Int64=3)
    L = monomials(X,d)
    T = sum(m*sqrt(idx_sum(exponents(m))) for m in L)
    return T, X
end

function arcsin_sum(v::Vector)
    n = length(v)
    sum([asin((-1)^i*i/n)*v[i] for i in 1:n])
end

function sym_tensor_arcsin(X, d::Int64=3)
    L = monomials(X,d)
    T = sum(m*arcsin_sum(exponents(m))*binomial(d,exponents(m)) for m in L)
    return T
end
