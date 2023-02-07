all: TensorDec.jl
	julia --color=yes make.jl

TensorDec.jl:
	git clone -b gh-pages https://github.com/AlgebraicGeometricModeling/TensorDec.jl.git

publish:
	cd TensorDec.jl; git add docs; git commit -am "update gh-pages"; git push
