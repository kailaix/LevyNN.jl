#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
using std::string;
using namespace tensorflow;
// #include "ADEL.h"
#include "PL.h"

REGISTER_OP("PL")
  
  .Input("x : double")
  .Input("theta : double")
  .Input("ac : double")
  .Input("h : double")
  .Output("y : double")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle x_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x_shape));
        shape_inference::ShapeHandle theta_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &theta_shape));
        shape_inference::ShapeHandle ac_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &ac_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &h_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });
class PLOp : public OpKernel {
private:
  
public:
  explicit PLOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& x = context->input(0);
    const Tensor& theta = context->input(1);
    const Tensor& ac = context->input(2);
    const Tensor& h = context->input(3);
    
    
    const TensorShape& x_shape = x.shape();
    const TensorShape& theta_shape = theta.shape();
    const TensorShape& ac_shape = ac.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(x_shape.dims(), 2);
    DCHECK_EQ(theta_shape.dims(), 2);
    DCHECK_EQ(ac_shape.dims(), 1);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    int Nx = x_shape.dim_size(0), n = theta_shape.dim_size(0);
    TensorShape y_shape({Nx});
            
    // create output tensor
    
    Tensor* y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, y_shape, &y));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto theta_tensor = theta.flat<double>().data();
    auto ac_tensor = ac.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto y_tensor = y->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(y_tensor, ac_tensor, *h_tensor, x_tensor, theta_tensor, Nx, n);
  }
};
REGISTER_KERNEL_BUILDER(Name("PL").Device(DEVICE_CPU), PLOp);


REGISTER_OP("PLGrad")
  
  .Input("grad_y : double")
  .Input("y : double")
  .Input("x : double")
  .Input("theta : double")
  .Input("ac : double")
  .Input("h : double")
  .Output("grad_x : double")
  .Output("grad_theta : double")
  .Output("grad_ac : double")
  .Output("grad_h : double");
class PLGradOp : public OpKernel {
private:
  
public:
  explicit PLGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_y = context->input(0);
    const Tensor& y = context->input(1);
    const Tensor& x = context->input(2);
    const Tensor& theta = context->input(3);
    const Tensor& ac = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& grad_y_shape = grad_y.shape();
    const TensorShape& y_shape = y.shape();
    const TensorShape& x_shape = x.shape();
    const TensorShape& theta_shape = theta.shape();
    const TensorShape& ac_shape = ac.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_y_shape.dims(), 1);
    DCHECK_EQ(y_shape.dims(), 1);
    DCHECK_EQ(x_shape.dims(), 2);
    DCHECK_EQ(theta_shape.dims(), 2);
    DCHECK_EQ(ac_shape.dims(), 1);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    int Nx = x_shape.dim_size(0), n = theta_shape.dim_size(0);
    TensorShape grad_x_shape(x_shape);
    TensorShape grad_theta_shape(theta_shape);
    TensorShape grad_ac_shape(ac_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_x = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_x_shape, &grad_x));
    Tensor* grad_theta = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_theta_shape, &grad_theta));
    Tensor* grad_ac = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_ac_shape, &grad_ac));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto theta_tensor = theta.flat<double>().data();
    auto ac_tensor = ac.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_y_tensor = grad_y.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto grad_x_tensor = grad_x->flat<double>().data();
    auto grad_theta_tensor = grad_theta->flat<double>().data();
    auto grad_ac_tensor = grad_ac->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    backward(grad_theta_tensor, grad_y_tensor, ac_tensor, *h_tensor, x_tensor, theta_tensor, Nx, n);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("PLGrad").Device(DEVICE_CPU), PLGradOp);

