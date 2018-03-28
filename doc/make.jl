using Documenter
using TensorDec

dir = "mrkd"
Expl = map(file -> joinpath("expl", file), filter(x ->endswith(x, "md"), readdir(dir*"/expl")))
Code = map(file -> joinpath("code", file), filter(x ->endswith(x, "md"), readdir(dir*"/code")))

makedocs(
         format = :html,
         sitename = "TensorDec",
         authors = "B. Mourrain",
         modules = [TensorDec],
         build = "html",
         source = dir,
         pages = Any[
                     "Home" => "index.md",
                     "Example" => Expl,
                     "Functions & types" => Code
                     ],
         repo = "https://gitlab.inria.fr/AlgebraicGeometricModeling/TensorDec.jl/tree/master",
         doctest = false
         )

deploydocs(
           deps = Deps.pip("mkdocs", "python-markdown-math"),
           repo = "gitlab.inria.fr/AlgebraicGeometricModeling/TensorDec.jl.git",
           target = "site",
           julia  = "0.6",
           osname = "osx",
           deps = nothing,
           make = nothing
           )
