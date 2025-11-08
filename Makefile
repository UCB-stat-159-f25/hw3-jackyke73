env:
	conda env update -f environment.yml --prune || conda env create -f environment.yml

html:
	myst build --html

clean:
	rm -rf figures audio _build
