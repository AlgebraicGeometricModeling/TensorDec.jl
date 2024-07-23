.SUFFIXES: .ipynb .md

MD := $(patsubst %.ipynb,%.md,$(wildcard src/expl/*.ipynb))

.ipynb.md:
	jupyter nbconvert --to markdown $*.ipynb
	perl -pi -e "s/\\$$\\$$/\\$$/g" $*.md



all:  $(MD) TensorDec.jl
	julia --color=yes make.jl

TensorDec.jl:
	git clone -b gh-pages https://github.com/AlgebraicGeometricModeling/TensorDec.jl.git

publish:
	cd TensorDec.jl; git add docs; git commit -am "update gh-pages"; git push

clean:
	rm ./mrkd/expl/*.md
