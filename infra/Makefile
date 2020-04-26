NVCC := nvcc
MAKE ?= make

LIBS := 

NVCCFLAGS := -I. \
			 -I.. \
			 -dlink

LDFLAGS += $(shell pkg-config opencv --cflags --libs)

OBJS := *.o

all:
	$(NVCC) $(NVCCFLAGS) -dc *.cu $(LDFLAGS)

clean:
	rm -f *.o
	rm -f *.a
