include("prelim.jl")
include("RNE_N_TR_R.jl")
include("RNE_N_TR.jl")
include("RGN_V_TR.jl")
include("spm.jl")

export approximate
"""
```
approximate(P::Polynomial, r:: Int64; mthd = :RNE, sdm = :Random)
```
This function approximates a symmetric tensor (real or complex valued) into a low rank symmetric tensor.\\
    Input:\\
    P: The homogeneous polynomial associated to the symmetric tensor to approximate.\\
    r: Approximation rank.\\
    mthd: The method to apply in order to find the approximation, there are 4 options (this function apply by default rne_n_tr):\\
    *RNE: To apply the function 'rne_n_tr';\\
    *RNER: To apply the function 'rne_n_tr_r'(when the symmetric tensor is real and the symmetric tensor approximation is required to be real);\\
    *RGN: To apply the function 'rgn_v_tr';\\
    *SPM: To apply the function 'spm_decompose'.\\
    sdm: The initial point for the first three options in mthd is chosen by the function decompose, where there are two options:
    *Random: To choose a random combination (default option);\\
    *NRnd: To choose non-random combination.

```
"""

function approximate(P::Polynomial, r:: Int64; mthd = :RNE, sdm = :Random)

    if mthd == :SPM
        V0 = randn(length(variables(P)))
        return spm_decompose(P,r,V0)
    end

    w0, V0, Info = decompose(P,cst_rkf(r), sdm)
    d = maxdegree(P)
    if mthd == :RNER
        return rne_n_tr_r(P, w0, V0)
    end

    if mthd == :RNE
        return rne_n_tr(P, w0, V0)
    elseif mthd == :RGN
        C0 = fill(zero(Complex{Float64}), size(V0,1), size(V0,2))
        for i in 1:r
            C0[:,i]=complex(w0[i])^(1/d)*V0[:,i]
        end
        w1 = fill(one(Complex{Float64}),r)
        return w1, rgn_v_tr(P,C0)...
    end
end


function approximate1(s::MultivariateSeries.Series, r:: Int64; args...)

    d = maxdegree(s)
    @polyvar x0
    X = cat(variables(s),[x0];dims=1)
    P = tensor(s,X)

    w, Xi, Info = approximate(P, r; args...)

    n = size(Xi,1)
    r = length(w)

    w = [w[i]*Xi[n,i]^d for i in 1:r]
    for i in 1:r
        Xi[:,i] /= Xi[n,i]
    end
    return w, Xi[1:n-1,:], Info
end
