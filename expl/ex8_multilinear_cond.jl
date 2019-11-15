using TensorDec

A0 = randn(25,25)
B0 = randn(25,25)
C0 = randn(25,25)

T0 = tensor(A0, B0, C0)

w, A, B, C = decompose(T0, eps_rkf(1.e-10), mode=2)

T = tensor(w, A, B, C)

er = norm(T-T0)/norm(T)

c = er/2.e-16
