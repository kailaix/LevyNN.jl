#define PI 3.14159265359
#include <algorithm>    // std::min

void forward(double *y, const double *x, const double *u, double c, int n, int m){
  double h = 2*PI/n;
  for(int i=0;i<m;i++) y[i] = 0.0;
  for(int i = 0; i<m;i++){
        for(int j=0;j<n;j++){
          // double d = std::min(fabs(x[i]-j*h), 2*PI-fabs(x[i]-j*h));
          double d = x[i]-j*h;
          y[i] += u[j]/sqrt(d*d+c*c);
        }
  }
}

void backward(double *grad_u, const double *grad_y, const double *y, const double *x, const double *u, double c, int n, int m){
  double h = 2*PI/n;
  for(int i=0;i<n;i++) grad_u[i] = 0.0;
  for(int i = 0; i<m;i++){
        for(int j=0;j<n;j++){
          double d = x[i]-j*h;
          grad_u[j] += 1/sqrt(d*d+c*c)*grad_y[i];
        }
  }
}