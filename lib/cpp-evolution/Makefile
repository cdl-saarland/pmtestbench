BUILDDIR ?= build
CFG      ?= debug
NAME     ?= pm-evo
SRCDIR   ?= src

all:

-include build_configs/$(CFG).cfg

# Be verbose about the build.
Q ?= @

BINDIR := $(BUILDDIR)/$(CFG)
BIN    := $(BINDIR)/$(NAME)
SRC    := $(sort $(wildcard $(SRCDIR)/*.cpp) $(wildcard $(SRCDIR)/*/*.cpp))
OBJ    := $(SRC:$(SRCDIR)/%.cpp=$(BINDIR)/%.o)
DEP    := $(OBJ:%.o=%.d)

DEFAULT_CFG_PATH_DIR := ${CURDIR}/run_configs/default.cfg

CFLAGS   := -Wall -W $(CFLAGS) -DDEFAULT_CFG_PATH="\"$(DEFAULT_CFG_PATH_DIR)\""
CXXFLAGS += $(CFLAGS) -std=c++17

# CFLAGS += -g -fno-omit-frame-pointer
# LDFLAGS += -g -fno-omit-frame-pointer

# CXX := clang++

DUMMY := $(shell mkdir -p $(sort $(dir $(OBJ))))

.PHONY: all clean link

all: $(BIN) link

-include $(DEP)

clean:
	@echo "[$(CFG)] ===> CLEAN"
	$(Q)rm -fr $(BINDIR)

link:
	@echo "[$(CFG)] ===> SYMLINK"
	$(Q)ln -sf $(CFG)/$(NAME) $(BUILDDIR)/last_build

$(BIN): $(OBJ)
	@echo "[$(CFG)] ===> LD $@"
	$(Q)$(CXX) -o $(BIN) $(OBJ) $(LDFLAGS)

$(BINDIR)/%.o: $(SRCDIR)/%.cpp
	@echo "[$(CFG)] ===> CXX $<"
	$(Q)$(CXX) $(CXXFLAGS) -MMD -c -o $@ $<
