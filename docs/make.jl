using Documenter, TensorDec

dir = "src"
Expl = map(file -> joinpath("expl", file), filter(x ->endswith(x, "md"), readdir(dir*"/expl")))
Code = map(file -> joinpath("code", file), filter(x ->endswith(x, "md"), readdir(dir*"/code")))

makedocs(
    sitename = "TensorDec",
    authors = "B. Mourrain, R. Khouja",
    modules = [TensorDec],
    build = "TensorDec.jl/docs",
    source = dir,
    pages = Any[
        "Home" => "index.md",
        "Tutorials" => Expl,
        "Manual" => Code
    ],
    repo = Remotes.GitHub("AlgebraicGeometricModeling", "TensorDec.jl"),
    doctest = false,
)

