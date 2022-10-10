.PHONY : clean-pyc clean-so clean-build clean in inplace test-code test-doc test-coverage test-manifest test trailing-spaces doc-plot doc

all: clean test doc-noplot

clean-pyc:
	find . -name "*.pyc" | xargs rm -f
	find . -name "__pycache__" | xargs rm -rf

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf build

clean: clean-build clean-pyc clean-so clean-ctags

in: inplace # just a shortcut
inplace:
	python setup.py build_ext -i

test-code:
	pytest ./autoreject

test-doc:
	pytest --doctest-glob='.rst'

test-coverage:
	rm -rf .coverage
	pytest --cov=autoreject/tests

test-manifest:
	check-manifest --ignore doc;

test: test-code test-doc test-manifest

trailing-spaces:
	find . -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

doc-plot:
	make -C doc html

doc:
	make -C doc html-noplot
