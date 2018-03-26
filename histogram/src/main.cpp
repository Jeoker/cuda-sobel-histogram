#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;

extern void GetHist(unsigned char *in_mat, 
                   unsigned int height, 
                   unsigned int width, 
                   unsigned int *hist);

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,  &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

int main( int argc, const char** argv ) { 
        // arg 1: Input image
        
        double start_gpu, finish_gpu;
        
        // Read input image from argument in black and white
        Mat input_image = imread(argv[1], IMREAD_GRAYSCALE);

        if (input_image.empty()) {
            cout << "Image cannot be loaded..!!" << endl;
            return -1;
        }
        
        unsigned int height = input_image.rows;
        unsigned int  width = input_image.cols;
        unsigned int  hist[256] = {0};
        unsigned int minval, maxval;
        
        cout << "please specify the range of the listogram: \n";
        cin >> minval >> maxval;
        
        // START GPU Processing
        start_gpu = CLOCK();

        GetHist((unsigned char *)input_image.data, 
                 height, width, 
                 (unsigned int *)hist);
        finish_gpu = CLOCK();
        
        cout << "GPU execution time: " << finish_gpu - start_gpu << " ms" << endl;
        printf("value\tfreq\n");
        for (int i = minval; i < maxval; i++)
            printf(" %d\t%d\n", i, hist[i]);

        return 0;
}
