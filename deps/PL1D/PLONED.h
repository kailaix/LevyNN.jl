#define PI 3.14159265359
void forward(double *y, const double *x, const double *u, int n, int m){
  double h = 2*PI/n;
  for(int k = 0; k<m;k++){
    int i = int(x[k]/h);
    if (i==n) i = 0;
    if (i+1==n)
      y[k] = (((i+1)*h-x[k])*u[i] + (x[k]-i*h)*u[0])/h;
    else
      y[k] = (((i+1)*h-x[k])*u[i] + (x[k]-i*h)*u[i+1])/h;
  }
}

void backward(double *grad_u, const double *grad_y, const double *y, const double *x, const double *u, int n, int m){
  double h = 2*PI/n;
  for(int i=0;i<n;i++) grad_u[i] = 0.0;
  for(int k = 0; k<m;k++){
    int i = int(x[k]/h);
    if (i==n) i = 0;
    grad_u[i] += ((i+1)*h-x[k])/h*grad_y[k];
    if(i+1==n)
      grad_u[0] += (x[k]-i*h)/h*grad_y[k];
    else
      grad_u[i+1] += (x[k]-i*h)/h*grad_y[k];
  }
}