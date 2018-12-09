include common.mk

PYDIR=qpy
DATADIR=data
CLEANDIRS = $(PYDIR:%=clean-%)

SEARCH=

.PHONY: pydir $(PYDIR)
.PHONY: datadir $(DATADIR)
.PHONY: cleandirs $(CLEANDIRS)
.PHONY: data
.PHONY: clean

all: data

data:
	@$(MAKE) -C data convert

clean: $(CLEANDIRS)
$(CLEANDIRS):
	@echo "cleaning directory $(@:clean-%=%):"
	@$(MAKE) -C $(@:clean-%=%) clean
