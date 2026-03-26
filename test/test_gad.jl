using LinearAlgebra, DynamicPolynomials, AlgebraicSolvers, TensorDec

X = @polyvar x y z
F = (x+z)^4 * (x+y) - 2 * x^3 * (x*y+y^2)
W, L, mu = gad_decompose(F)
T = reconstruct(W, L, maxdegree(F))
e = norm(F-T)
