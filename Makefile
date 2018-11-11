CFLAGS = "-std=c99"

all: boolean_demo mnist_demo

mnist_demo: mnist_demo.o tensor.o neuralnet.o functions.o loss_functions.o mnist_dataset.o
	gcc -o mnist_demo mnist_demo.o tensor.o neuralnet.o functions.o loss_functions.o mnist_dataset.o -lm

mnist_demo.o: mnist_demo.c
	gcc $(CFLAGS) -c mnist_demo.c

boolean_demo: boolean_demo.o tensor.o neuralnet.o functions.o loss_functions.o mnist_dataset.o
	gcc -o boolean_demo boolean_demo.o tensor.o neuralnet.o functions.o loss_functions.o mnist_dataset.o -lm

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

mnist_dataset.o: mnist_dataset.c mnist_dataset.h dataset.o
	gcc $(CFLAGS) -c mnist_dataset.c

dataset.o: dataset.c dataset.h
	gcc $(CFLAGS) -c dataset.c

clean:
	rm -rf *.o ||:
	rm -rf boolean_demo test ||:
