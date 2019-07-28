#include<cmath>

double calc_y(double a, double c, double h1, double h2, double x1, double x2, const double *theta, int n){
  int i = int(floor((x1-a)/h1)), j = int(floor((x2-c)/h2));
  if(i>=n-1 || j>=n-1){
    printf("points out of domain (%f %f)\n", x1, x2);
    exit(1);
  }
  double alpha = (x1-a)/h1 - i;
  double beta = (x2-c)/h2 - j;
  double w1,w2,w3,theta1, theta2, theta3;
  if(alpha>beta){
     w1 = 1-alpha;
     w2 = alpha-beta;
     w3 = beta;
     theta1 = theta[i+j*n];
     theta2 = theta[i+1+j*n];
     theta3 = theta[i+1+(j+1)*n];
    // printf("^^ %f %f\n", w1, theta1);
  }else{
    w1 = 1-beta;
    w2 = beta-alpha;
    w3 = alpha;
    theta1 = theta[i+j*n];
    theta2 = theta[i+(j+1)*n];
    theta3 = theta[i+1+(j+1)*n];
  }
  // printf("(%f, %f), [%d] %d %d ==> %f, %f, %f\n", x1, x2, i+j*n, i, j, theta1, theta2, theta3);
  return theta1*w1 + theta2*w2 + theta3*w3;
}

void calc_dtheta(double *dtheta, double dy, double a, double c, double h1, double h2, double x1, double x2, int n){
  int i = int(floor((x1-a)/h1)), j = int(floor((x2-c)/h2));
  if(i>=n-1 || j>=n-1){
    printf("points out of domain (%f %f)\n", x1, x2);
    exit(1);
  }
  // printf("(%d %d) a = %f c = %f, h1 = %f, h2 = %f\n", i, j, a, c, h1, h2);
  double alpha = (x1-a)/h1 - i;
  double beta = (x2-c)/h2 - j;
  double w1,w2,w3;
  if(alpha>beta){
     w1 = 1-alpha;
     w2 = alpha-beta;
     w3 = beta;
     dtheta[i+j*n] += dy*w1;
     dtheta[i+1+j*n] += dy*w2;
     dtheta[i+1+(j+1)*n] += dy*w3;
    //  printf("%d %f %f^^\n", i+1+(j+1)*n, w1, dy);
  }else{
    w1 = 1-beta;
    w2 = beta-alpha;
    w3 = alpha;
    dtheta[i+j*n] += dy*w1;
    dtheta[i+(j+1)*n] += dy*w2;
    dtheta[i+1+(j+1)*n] += dy*w3;
    // printf("%d __\n", i+1+(j+1)*n);
  }
}


// x: Nx x 2 
// theta: n x n
void forward(double *y, const double *ac, double h, const double*x, const double*theta, int Nx, int n){
  for(int i=0;i<Nx;i++){
    // printf("Calling...%f %f\n", ac[0], ac[1]);
    y[i] = calc_y(ac[0], ac[1], h, h, x[2*i], x[2*i+1], theta,  n);
  }
}

void backward(double *dtheta, const double*dy, const double *ac, double h, const double *x, 
          const double *theta, int Nx, int n){
  for(int i=0;i<n*n;i++){
    dtheta[i] = 0.0;
  }
  for(int i=0;i<Nx;i++){
    // printf("%d %d\n", i, Nx);
    calc_dtheta(dtheta, dy[i], ac[0], ac[1], h, h, x[2*i], x[2*i+1], n);
  }
}
