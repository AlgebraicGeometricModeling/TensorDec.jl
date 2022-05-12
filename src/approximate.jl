include("prelim.jl")
include("RNE_N_TR_R.jl")
include("RNE_N_TR.jl")
include("RGN_V_TR.jl")
include("spm.jl")

export approximate

"""
```
approximate(P::Polynomial, r:: Int64; iter = :RNE, init = :Random)
```
This function approximates a symmetric tensor (real or complex valued) into a low rank symmetric tensor.

Input:
   - P: The homogeneous polynomial associated to the symmetric tensor to approximate.
   - r: Approximation rank.

The option `iter` specifies the method to apply in order to find the approximation, there are 4 options
(the default is rne_n_tr):

     * RNE: To apply the function 'rne_n_tr';
     * RNER: To apply the function 'rne_n_tr_r'(when the symmetric tensor is real and the symmetric tensor approximation is required to be real);
     * RGN: To apply the function 'rgn_v_tr';
     * SPM: To apply the function 'spm_decompose'.

The option `init` specifies the way the initial point for the first three methods is chosen by the function decompose:

     * Random: To choose a random combination (default option);
     * Rnd: To choose non-random combination.

"""
function approximate(P::Polynomial, r:: Int64; iter = :RNE, init = :Random)

    if iter == :SPM
        V0 = randn(length(variables(P)))
        return spm_decompose(P,r,V0)
    end

    w0, V0, Info = decompose(P,cst_rkf(r), init)
    d = maxdegree(P)

    if iter == :RNER

        c = 0
        while c<5 && !isreal(V0)
            w0, V0, Info = decompose(P,cst_rkf(r))
            c += 1
        end
        if c >0
           println("\n[ initial decomposition repeated (", string(c),") ]")
        end
        return rne_n_tr_r(P, w0, V0)

    elseif iter == :RNE

        return rne_n_tr(P, w0, V0)

    elseif iter == :RGN

        C0 = fill(zero(Complex{Float64}), size(V0,1), size(V0,2))
        for i in 1:r
            C0[:,i]=complex(w0[i])^(1/d)*V0[:,i]
        end
        w1 = fill(one(Complex{Float64}),r)
        return w1, rgn_v_tr(P,C0)...

    end
end

"""
```
approximate(P::Polynomial, w0, V0; iter = :RNE, init = :Random)
```
This function approximates a symmetric tensor (real or complex valued) into a low rank symmetric
tensor starting from an initial decomposition (w0, V0)

Input:
   - P: The homogeneous polynomial associated to the symmetric tensor to approximate.
   - w0: Initial weights of the decomposition
   - V0: Initial vectors of the decomposition

The option `iter` specifies the method to apply in order to find the approximation, there are 4 options
(the default is :RNE):

     * RNE: To apply the function 'rne_n_tr';
     * RNER: To apply the function 'rne_n_tr_r'(when the symmetric tensor is real and the symmetric tensor approximation is required to be real);
     * RGN: To apply the function 'rgn_v_tr';
     * SPM: To apply the function 'spm_decompose'.

"""
function approximate(P::Polynomial, w0, V0; iter = :RNE)

    r = length(w0)
    d = maxdegree(P)
    
    if iter == :SPM
        C0 = fill(zero(Complex{Float64}), size(V0,1), size(V0,2))
        for i in 1:r
            C0[:,i]=complex(w0[i])^(1/d)*V0[:,i]
        end
        return spm_decompose(P,r,C0[:,1])
    end

    if iter == :RNE
        if isreal(V0) && isreal(w0)

            return rne_n_tr_r(P, w0, V0)

        else

            return rne_n_tr(P, w0, V0)
        end
    elseif iter == :RGN

        C0 = fill(zero(Complex{Float64}), size(V0,1), size(V0,2))
        for i in 1:r
            C0[:,i]=complex(w0[i])^(1/d)*V0[:,i]
        end
        w1 = fill(one(Complex{Float64}),r)
        return w1, rgn_v_tr(P,C0)...

    end
end



function approximate(s::MultivariateSeries.Series, r:: Int64; args...)

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
