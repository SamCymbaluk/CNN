#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "cpgplot.h"


void plotMetric(char* title, float min, float max, float* losses, size_t n) {
    if (n == 0) return;
    if (!cpgopen("/XWINDOW")) return;

    float x[n];

    for (size_t i = 0; i < n; i++) {
        x[i] = i + 1;
    }
    
    printf("Length: %d\n", n);
     
    printf("x[0] = %f, losses[0] = %f\n", x[0], losses[0]);
    
    cpgenv(1, (int) n + 1, min, max, 0, 1);

    cpgline(n, x, losses);
}
