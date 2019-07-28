#include <cmath>
#include <iostream>
using namespace std;

void forward(double *out, const double *x, const double *theta, double c,const double*p, const double*ac, double h, int n, int N){
    auto xij = new double[n];
    auto yij = new double[n];
    for(int i=0;i<n;i++){
        xij[i] = ac[0] + h*i;
        yij[i] = ac[1] + h*i;
    }
    for(int i=0;i<N;i++){
        out[i] = 0.0;
    }
    for(int k=0;k<N;k++){
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                int idx = i+j*n;
                double c0 = sqrt(c*c + pow(x[2*k]-xij[i],2)+pow(x[2*k+1]-yij[j],2));
                out[k] += theta[idx]/c0;
            }
        }
        out[k] += p[0] + p[1]*x[2*k]+p[2]*x[2*k+1]+p[3]*x[2*k]*x[2*k]+p[4]*x[2*k]*x[2*k+1]+p[5]*x[2*k+1]*x[2*k+1];
    }    
    delete [] xij;
    delete [] yij;
}

void backward(double *grad_theta, double *grad_c, double *grad_p,
        const double *grad_out, const double *out, const double *x, const double *theta, double c,const double*p, const double*ac, double h, int n, int N){
    auto xij = new double[n];
    auto yij = new double[n];
    for(int i=0;i<n;i++){
        xij[i] = ac[0] + h*i;
        yij[i] = ac[1] + h*i;
    }
    *grad_c = 0.0;
    for(int i=0;i<n*n;i++){
        grad_theta[i] = 0.0;
    }
    for(int i=0;i<6;i++){
        grad_p[i] = 0.0;
    }

    for(int k=0;k<N;k++){
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                int idx = i+j*n;
                double c0 = sqrt(c*c + pow(x[2*k]-xij[i],2)+pow(x[2*k+1]-yij[j],2));
                grad_theta[idx] += grad_out[k]/c0;
                *grad_c -= grad_out[k]*c*theta[idx]/c0/c0/c0;
            }
        }
        grad_p[0] += grad_out[k];
        grad_p[1] += grad_out[k]*x[2*k];
        grad_p[2] += grad_out[k]*x[2*k+1];
        grad_p[3] += grad_out[k]*x[2*k]*x[2*k];
        grad_p[4] += grad_out[k]*x[2*k]*x[2*k+1];
        grad_p[5] += grad_out[k]*x[2*k+1]*x[2*k+1];
    }

    delete [] xij;
    delete [] yij;
}
