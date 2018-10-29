all: test boolean_demo

boolean_demo: boolean_demo.o tensor.o neuralnet.o functions.o
	gcc -o boolean_demo boolean_demo.o tensor.o neuralnet.o functions.o -lm

boolean_demo.o: boolean_demo.c
	gcc -c boolean_demo.c

test: test.o tensor.o neuralnet.o functions.o
	gcc -o test test.o tensor.o neuralnet.o functions.o -lm

test.o: test.c
	gcc -c test.c

neuralnet.o: neuralnet.c neuralnet.h tensor.o
	gcc -c neuralnet.c

tensor.o: tensor.c tensor.h functions.o
	gcc -c tensor.c

functions.o: functions.c functions.h
	gcc -c functions.c

