#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
using std::string;
using namespace tensorflow;
// If you want to use the PyTorch feature, uncomment the following line
// #include "la.h" 
#include "RBFONED.h"

REGISTER_OP("RBFONED")
  
  .Input("x : double")
  .Input("u : double")
  .Input("c : double")
  .Output("y : double")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle x_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &x_shape));
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &u_shape));
        shape_inference::ShapeHandle c_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &c_shape));

        c->set_output(0, c->input(0));
    return Status::OK();
  });
class RBFONEDOp : public OpKernel {
private:
  
public:
  explicit RBFONEDOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& x = context->input(0);
    const Tensor& u = context->input(1);
    const Tensor& c = context->input(2);
    
    
    const TensorShape& x_shape = x.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& c_shape = c.shape();
    
    
    DCHECK_EQ(x_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 0);

    // extra check
        
    // create output shape
    int m = x_shape.dim_size(0), n = u_shape.dim_size(0);
    
    TensorShape y_shape({m});
            
    // create output tensor
    
    Tensor* y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, y_shape, &y));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto u_tensor = u.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto y_tensor = y->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(y_tensor, x_tensor, u_tensor, *c_tensor, n, m);

  }
};
REGISTER_KERNEL_BUILDER(Name("RBFONED").Device(DEVICE_CPU), RBFONEDOp);


REGISTER_OP("RBFONEDGrad")
  
  .Input("grad_y : double")
  .Input("y : double")
  .Input("x : double")
  .Input("u : double")
  .Input("c : double")
  .Output("grad_x : double")
  .Output("grad_u : double")
  .Output("grad_c : double");
class RBFONEDGradOp : public OpKernel {
private:
  
public:
  explicit RBFONEDGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_y = context->input(0);
    const Tensor& y = context->input(1);
    const Tensor& x = context->input(2);
    const Tensor& u = context->input(3);
    const Tensor& c = context->input(4);
    
    
    const TensorShape& grad_y_shape = grad_y.shape();
    const TensorShape& y_shape = y.shape();
    const TensorShape& x_shape = x.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& c_shape = c.shape();
    
    
    DCHECK_EQ(grad_y_shape.dims(), 1);
    DCHECK_EQ(y_shape.dims(), 1);
    DCHECK_EQ(x_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
    int m = x_shape.dim_size(0), n = u_shape.dim_size(0);
        
    // create output shape
    
    TensorShape grad_x_shape(x_shape);
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_c_shape(c_shape);
            
    // create output tensor
    
    Tensor* grad_x = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_x_shape, &grad_x));
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_u_shape, &grad_u));
    Tensor* grad_c = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_c_shape, &grad_c));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto u_tensor = u.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto grad_y_tensor = grad_y.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto grad_x_tensor = grad_x->flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_c_tensor = grad_c->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    backward(grad_u_tensor, grad_y_tensor, y_tensor, x_tensor, u_tensor, *c_tensor, n, m);

    
  }
};
REGISTER_KERNEL_BUILDER(Name("RBFONEDGrad").Device(DEVICE_CPU), RBFONEDGradOp);

