# CNN
C NeuralNets: A simple neural network library in C

Developed as part of my final project for PHYSICS 2G03 at McMaster.
Currently being prepared for full release.

### Release roadmap
- [ ] Convert to C++
- [ ] Remove small memory leaks
- [ ] Create user documentation

### Installation
```bash
git clone https://github.com/SamCymbaluk/CNN.git
cd CNN
make build
sudo make install
```
This will create a `cnn.so` shared object file in the CNN directory and copy it to /usr/lib to allow use by other programs.

You can use this in your own C program like so:
```C
#include "path/to/cnn.h"

int main() {
  ...
};
```
And compile with
```bash
gcc -c myprog.c
gcc -o myprog myprog.o -Lpath/to/cnn.so -l:cnn.so
```

### Demos
This repo includes a number of demos to play with.
Compile the demos by running `make demos`.
You can then try them out by executing one of the following commands in the CNN directory:
```bash
./xor_demo
./mnist_demo
```
