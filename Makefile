include common.mk

PYDIR=finquant
DATADIR=data
EXAMPLEDIR=example
EXAMPLEFILES=$(wildcard example/Example*.py)
TESTDIR=tests
CLEANDIRS = $(PYDIR:%=clean-%) \
$(EXAMPLEDIR:%=clean-%) \
$(TESTDIR:%=clean-%)

SEARCH=

.PHONY: test
.PHONY: EXAMPLEFILES $(EXAMPLEFILES)
.PHONY: cleandirs $(CLEANDIRS)
.PHONY: clean

all: clean

test:
	@echo "Running tests"
	@$(MAKE) -C tests

copyexamples: $(EXAMPLEFILES)
$(EXAMPLEFILES):
	@cp $(@) $(subst example/,tests/test_,$(@))

clean: $(CLEANDIRS)
$(CLEANDIRS):
	@echo "cleaning directory $(@:clean-%=%):"
	@$(MAKE) -C $(@:clean-%=%) clean

search:
	@echo "searching all python files for $(SEARCH):"
	@find . \( -name "*.py" -o -name "README.tex.md" \) -not -path "./*/bkup-files/*" | xargs grep -i --color=auto $(SEARCH)

