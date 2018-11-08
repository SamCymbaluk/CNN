CFLAGS = "-std=c99"

all: boolean_demo

boolean_demo: boolean_demo.o tensor.o neuralnet.o functions.o loss_functions.o
	gcc -o boolean_demo boolean_demo.o tensor.o neuralnet.o functions.o loss_functions.o -lm

boolean_demo.o: boolean_demo.c
	gcc $(CFLAGS) -c boolean_demo.c

neuralnet.o: neuralnet.c neuralnet.h loss_functions.o tensor.o
	gcc $(CFLAGS) -c neuralnet.c

loss_functions.o: loss_functions.c loss_functions.h tensor.o
	gcc $(CFLAGS) -c loss_functions.c

tensor.o: tensor.c tensor.h functions.o
	gcc $(CFLAGS) -c tensor.c

functions.o: functions.c functions.h
	gcc $(CFLAGS) -c functions.c

clean:
	rm -rf *.o ||:
	rm -rf boolean_demo test ||:
