include common.mk

PYDIR=quantpy
DATADIR=data
EXAMPLEDIR=example
CLEANDIRS = $(PYDIR:%=clean-%) $(EXAMPLEDIR:%=clean-%)

SEARCH=

.PHONY: cleandirs $(CLEANDIRS)
.PHONY: clean

all: clean

clean: $(CLEANDIRS)
$(CLEANDIRS):
	@echo "cleaning directory $(@:clean-%=%):"
	@$(MAKE) -C $(@:clean-%=%) clean

search:
	@echo "searching all python files for $(SEARCH):"
	@find . -name "*.py" | xargs grep -i --color=auto $(SEARCH)

