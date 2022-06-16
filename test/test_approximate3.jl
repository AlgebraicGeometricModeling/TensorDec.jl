using LinearAlgebra, DynamicPolynomials, TensorDec



n = 3
X = (@polyvar x[1:n])[1]

d = 3

r0 = 1
w  = [1, 0.5]
Xi = [1.0 0.1;
      2.0 1;
      3.0 -30]

s = series(w, Xi, X, d)

@polyvar x0

X0 = cat([x0],X; dims=1)

T = tensor(s,X0)


r = 2

w0, Xi0 = approximate(s, r; iter = :RGN)

