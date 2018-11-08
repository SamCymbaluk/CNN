CFLAGS = "-std=c99"

all: test boolean_demo

boolean_demo: boolean_demo.o tensor.o neuralnet.o functions.o
	gcc -o boolean_demo boolean_demo.o tensor.o neuralnet.o functions.o -lm

boolean_demo.o: boolean_demo.c
	gcc $(CFLAGS) -c boolean_demo.c

test: test.o tensor.o neuralnet.o functions.o
	gcc -o test test.o tensor.o neuralnet.o functions.o -lm

test.o: test.c
	gcc $(CFLAGS) -c test.c

neuralnet.o: neuralnet.c neuralnet.h tensor.o
	gcc $(CFLAGS) -c neuralnet.c

tensor.o: tensor.c tensor.h functions.o
	gcc $(CFLAGS) -c tensor.c

functions.o: functions.c functions.h
	gcc $(CFLAGS) -c functions.c
