using Documenter
using TensorDec

dir = "mrkd"
Expl = map(file -> joinpath("expl", file), filter(x ->endswith(x, "md"), readdir(dir*"/expl")))
Code = map(file -> joinpath("code", file), filter(x ->endswith(x, "md"), readdir(dir*"/code")))

makedocs(
         sitename = "TensorDec",
         authors = "B. Mourrain",
         modules = [TensorDec],
         build = "TensorDec.jl/docs",
         source = dir,
         pages = Any[
                     "Home" => "index.md",
#                     "Examples" => Expl,
#                     "Functions & types" => Code
                     ],
         repo = Remotes.GitHub("AlgebraicGeometricModeling", "TensorDec.jl"),
         doctest = false
         )
#=
deploydocs(
           repo = "github.com/AlgebraicGeometricModeling/TensorDec.jl.git",
           target = "site"
           )
=#
