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
#include "RBFPoly.h"

REGISTER_OP("RBFPoly")
  
  .Input("x : double")
  .Input("theta : double")
  .Input("c : double")
  .Input("p : double")
  .Input("ac : double")
  .Input("h : double")
  .Output("y : double")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle x_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x_shape));
        shape_inference::ShapeHandle theta_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &theta_shape));
        shape_inference::ShapeHandle c_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &c_shape));
        shape_inference::ShapeHandle p_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &p_shape));
        shape_inference::ShapeHandle ac_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &ac_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &h_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });
class RBFPolyOp : public OpKernel {
private:
  
public:
  explicit RBFPolyOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(6, context->num_inputs());
    
    
    const Tensor& x = context->input(0);
    const Tensor& theta = context->input(1);
    const Tensor& c = context->input(2);
    const Tensor& p = context->input(3);
    const Tensor& ac = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& x_shape = x.shape();
    const TensorShape& theta_shape = theta.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& p_shape = p.shape();
    const TensorShape& ac_shape = ac.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(x_shape.dims(), 2);
    DCHECK_EQ(theta_shape.dims(), 2);
    DCHECK_EQ(c_shape.dims(), 0);
    DCHECK_EQ(p_shape.dims(), 1);
    DCHECK_EQ(ac_shape.dims(), 1);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    int N = x_shape.dim_size(0);
    int n = theta_shape.dim_size(0);
    
    TensorShape y_shape({N});
            
    // create output tensor
    
    Tensor* y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, y_shape, &y));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto theta_tensor = theta.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto p_tensor = p.flat<double>().data();
    auto ac_tensor = ac.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto y_tensor = y->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(y_tensor, x_tensor, theta_tensor, *c_tensor,p_tensor, ac_tensor, *h_tensor, n, N);

  }
};
REGISTER_KERNEL_BUILDER(Name("RBFPoly").Device(DEVICE_CPU), RBFPolyOp);


REGISTER_OP("RBFPolyGrad")
  
  .Input("grad_y : double")
  .Input("y : double")
  .Input("x : double")
  .Input("theta : double")
  .Input("c : double")
  .Input("p : double")
  .Input("ac : double")
  .Input("h : double")
  .Output("grad_x : double")
  .Output("grad_theta : double")
  .Output("grad_c : double")
  .Output("grad_p : double")
  .Output("grad_ac : double")
  .Output("grad_h : double");
class RBFPolyGradOp : public OpKernel {
private:
  
public:
  explicit RBFPolyGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_y = context->input(0);
    const Tensor& y = context->input(1);
    const Tensor& x = context->input(2);
    const Tensor& theta = context->input(3);
    const Tensor& c = context->input(4);
    const Tensor& p = context->input(5);
    const Tensor& ac = context->input(6);
    const Tensor& h = context->input(7);
    
    
    const TensorShape& grad_y_shape = grad_y.shape();
    const TensorShape& y_shape = y.shape();
    const TensorShape& x_shape = x.shape();
    const TensorShape& theta_shape = theta.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& p_shape = p.shape();
    const TensorShape& ac_shape = ac.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_y_shape.dims(), 1);
    DCHECK_EQ(y_shape.dims(), 1);
    DCHECK_EQ(x_shape.dims(), 2);
    DCHECK_EQ(theta_shape.dims(), 2);
    DCHECK_EQ(c_shape.dims(), 0);
    DCHECK_EQ(p_shape.dims(), 1);
    DCHECK_EQ(ac_shape.dims(), 1);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
    int N = x_shape.dim_size(0);
    int n = theta_shape.dim_size(0);

    // create output shape
    
    TensorShape grad_x_shape(x_shape);
    TensorShape grad_theta_shape(theta_shape);
    TensorShape grad_c_shape(c_shape);
    TensorShape grad_p_shape(p_shape);
    TensorShape grad_ac_shape(ac_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_x = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_x_shape, &grad_x));
    Tensor* grad_theta = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_theta_shape, &grad_theta));
    Tensor* grad_c = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_c_shape, &grad_c));
    Tensor* grad_p = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_p_shape, &grad_p));
    Tensor* grad_ac = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_ac_shape, &grad_ac));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto theta_tensor = theta.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto p_tensor = p.flat<double>().data();
    auto ac_tensor = ac.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_y_tensor = grad_y.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto grad_x_tensor = grad_x->flat<double>().data();
    auto grad_theta_tensor = grad_theta->flat<double>().data();
    auto grad_c_tensor = grad_c->flat<double>().data();
    auto grad_p_tensor = grad_p->flat<double>().data();
    auto grad_ac_tensor = grad_ac->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    backward(grad_theta_tensor, grad_c_tensor, grad_p_tensor, grad_y_tensor, y_tensor, x_tensor, theta_tensor, *c_tensor,p_tensor, ac_tensor,
      *h_tensor, n, N);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("RBFPolyGrad").Device(DEVICE_CPU), RBFPolyGradOp);

