include common.mk

PYDIR=finquant
DATADIR=data
EXAMPLEDIR=example
EXAMPLEFILES=$(wildcard example/Example*.py)
TESTDIR=tests
DOCDIR=docs
DISTDIR=dist
BUILDDIR=build
AUTODOCEXAMPLES=autodoc-examples.sh
CLEANDIRS = $(PYDIR:%=clean-%) \
$(EXAMPLEDIR:%=clean-%) \
$(TESTDIR:%=clean-%) \
$(DOCDIR:%=clean-%)

SEARCH=

.PHONY: test
.PHONY: doc
.PHONY: EXAMPLEFILES $(EXAMPLEFILES)
.PHONY: cleandirs $(CLEANDIRS)
.PHONY: clean
.PHONY: dirclean

all: clean

test:copyexamples
	@echo "Running tests"
	@$(MAKE) -C $(TESTDIR)

copyexamples: $(EXAMPLEFILES)
$(EXAMPLEFILES):
	@cp $(@) $(subst example/,tests/test_,$(@))

pypi:
	@$(PYTHON) setup.py sdist bdist_wheel
	@$(PYTHON) -m twine upload dist/FinQuant-*

doc:
	@$(MAKE) -C $(DOCDIR) clean
	@$(MAKE) -C $(DOCDIR) html

clean: dirclean
clean:
	-@rm -rf .pytest_cache FinQuant.egg-info $(BUILDDIR)/ $(DISTDIR)/

dirclean: $(CLEANDIRS)
$(CLEANDIRS):
	@echo "cleaning directory $(@:clean-%=%):"
	-@$(MAKE) -C $(@:clean-%=%) clean

search:
	@echo "searching all python files for $(SEARCH):"
	@find . \( -name "*.py" -o -name "README.tex.md" \) -not -path "./*/bkup-files/*" -not -path "./docs/build/*" -not -path "./build/*" | xargs grep -i --color=auto $(SEARCH)

