# Makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    = -vT
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = autoreject
PAPER         =
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help Makefile view install

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# View the built documentation
view:
	@python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/_build/html/index.html')"

install:
	rm -rf _build/doctrees _build/autoreject.github.io
	# first clone the autoreject/autoreject.github.io repo because it may ask
	# for password and we don't want to delay this long build in
	# the middle of it
	# --no-checkout just fetches the root folder without content
	# --depth 1 is a speed optimization since we don't need the
	# history prior to the last commit
	git clone --no-checkout --depth 1 git@github.com:autoreject/autoreject.github.io.git _build/autoreject.github.io
	touch _build/autoreject.github.io/.nojekyll
	make html
	cd _build/ && \
	cp -r html/* autoreject.github.io && \
	cd autoreject.github.io && \
	git add * && \
	git add .nojekyll && \
	git commit -a -m 'Make install' && \
	git push

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

html_dev-pattern:
	BUILD_DEV_HTML=1 $(SPHINXBUILD) -D sphinx_gallery_conf.filename_pattern=$(PATTERN) -D sphinx_gallery_conf.run_stale_examples=True -b html $(ALLSPHINXOPTS) $(SOURCEDIR) _build/html
	@echo
	@echo "Build finished. The HTML pages are in _build/html"
