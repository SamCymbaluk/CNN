CFLAGS = -std=c99 -fPIC -Wall

build: cnn.so

install: cnn.so
	cp cnn.so /usr/lib/cnn.so
	ldconfig /usr/lib

demos: xor_demo mnist_demo

cnn.so: cnn.h tensor.o neuralnet.o functions.o loss_functions.o dataset.o mnist_dataset.o trainer.o optimizer.o
	gcc -o cnn.so $? -shared

mnist_demo: mnist_demo.o cnn.so
	gcc -o mnist_demo mnist_demo.o -L. -l:cnn.so -lm

mnist_demo.o: mnist_demo.c cnn.so
	gcc $(CFLAGS) -c mnist_demo.c

xor_demo: xor_demo.o
	gcc -o xor_demo xor_demo.o -L. -l:cnn.so -lm

xor_demo.o: xor_demo.c
	gcc $(CFLAGS) -c xor_demo.c

trainer.o: trainer.c trainer.h optimizer.o neuralnet.o dataset.o
	gcc $(CFLAGS) -c trainer.c

optimizer.o: optimizer.c optimizer.h neuralnet.o tensor.o
	gcc $(CFLAGS) -c optimizer.c

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
	rm -rf *.o *.so ||:
	rm -rf xor_demo mnist_demo ||:
