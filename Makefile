#
#flags
#
CC = g++

CFLAGS = -Wall -Wextra -fopenmp
DBGCFLAGS = -DDEBUG -g
RELCFLAGS = -o3

#
#project files
#
SRCS = main.cpp sample.cpp neural.cpp nnLayer.cpp plot.cpp stringCheck.cpp nnio.cpp algorithm.cpp nnfun.cpp nnGetSample.cpp nnLoad.cpp nnInfo.cpp nnTest.cpp
OBJS = $(SRCS:.cpp=.o)
DBGBIN = neuraldbg
RELBIN = neural
#
#include files

LIB = -larmadillo -lmgl

#
#build settings
#
SRCDIR = src

DBGOBJDIR = ./obj/debug
RELOBJDIR = ./obj/release
DBGBINDIR = ./#bin/debug
RELBINDIR = ./#bin/release

DBGOBJS = $(addprefix $(DBGOBJDIR)/, $(OBJS))
RELOBJS = $(addprefix $(RELOBJDIR)/, $(OBJS))

.PHONY: all clean debug prep release remake

all:prep release

#
#
#
debug: prep $(DBGBIN)
	@echo "debug build"

$(DBGBINDIR)/$(DBGBIN) : $(DBGOBJS)
	$(CC) $(CFLAGS) $(DBGCFLAGS) $^ -o $(DBGBIN) $(LIB)
	@echo "Linked !"

$(DBGOBJS) : $(DBGOBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CC) -c $(CFLAGS) $(DBGCFLAGS) -c $< -o $@
	@echo "Compiled "$<" !"

#
#release
#

release: prep $(RELBIN)
	@echo "release build"

$(RELBINDIR)/$(RELBIN) : $(RELOBJS)
	$(CC) $(CFLAGS) $(RELCFLAGS) $^ -o $(RELBIN) $(LIB)
	@echo "Linked !"

$(RELOBJS) : $(RELOBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CC) $(CFLAGS) $(RELCFLAGS) -c $< -o $@
	@echo "Compiled "$<" !"

#
#other
#
prep:
	@mkdir -p $(DBGOBJDIR) $(DBGBINDIR) $(RELOBJDIR) $(RELBINDIR)

remake: clean all

clean:
	rm -f $(DBGOBJS)
	rm -f $(RELOBJS)
	rm -f $(DBGBIN)
	rm -f $(RELBIN)
