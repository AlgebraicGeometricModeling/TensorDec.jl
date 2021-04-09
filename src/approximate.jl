#include("ahp.jl")
include("prelim.jl")
include("symr.jl")
include("RNE_N_TR.jl")
include("RGN_V_TR.jl")
include("spm.jl")

export approximate

function approximate(P, r:: Int64; mthd = :RNE)

    w0, V0, Info = decompose(P,r)
    d = maxdegree(P)

    if mthd == :RNE
        return rne_n_tr(P, w0, V0)
    elseif mthd == :RGN
        C0 = fill(complex(0.0), size(V0,1), size(V0,2))
        for i in 1:r
            C0[:,i]=complex(w0[i])^(1/d)*V0[:,i]
        end
        w1 = fill(1.,r)
        return w1, rgn_v_tr(P,C0)...
    elseif mthd == :SPM
        V0 = randn(length(variables(P)))
        return spm_decompose(P,r,V0)
    end
end
