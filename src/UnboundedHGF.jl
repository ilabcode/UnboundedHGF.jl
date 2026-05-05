module UnboundedHGF

include("types.jl")
include("updates.jl")
include("hgf.jl")

export ClassicUpdate, UHGFUpdate, HGFParams, HGFTrajectory
export run_hgf, first_nan_row, n_levels

end # module UnboundedHGF
