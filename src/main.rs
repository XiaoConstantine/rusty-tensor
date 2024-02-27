use std::cell::RefCell;
use std::rc::Rc;

use ndarray::{ArrayD, Ix2, IxDyn};
// Context to hold intermediate values for backward pass
struct Context {
    arg: Rc<RefCell<dyn Function>>,
    parents: Vec<Rc<Tensor>>,
    saved_tensors: Vec<ArrayD<f32>>,
}

impl Context {
    fn new(arg: Rc<RefCell<dyn Function>>, parents: Vec<Rc<Tensor>>) -> Self {
        Context {
            arg,
            parents,
            saved_tensors: vec![],
        }
    }

    fn save_for_backward(&mut self, tensors: ArrayD<f32>) {
        self.saved_tensors.push(tensors);
    }
}

struct Tensor {
    data: ArrayD<f32>,
    grad: RefCell<ArrayD<f32>>,
    ctx: Option<Rc<RefCell<Context>>>,
}

impl Tensor {
    fn new(data: ArrayD<f32>) -> Tensor {
        let grad = ArrayD::<f32>::zeros(data.raw_dim());
        Tensor {
            data,
            grad: RefCell::new(grad),
            ctx: None,
        }
    }

    fn backward(&self, allow_fill: bool) {
        if let Some(ctx) = &self.ctx {
            let ctx_ref = ctx.borrow();
            if self.grad.borrow().sum().abs() < f32::EPSILON && !allow_fill {
                assert!(
                    self.data.len() == 1,
                    "data size must be 1 for auto-fill gradient"
                );
                *self.grad.borrow_mut() = ArrayD::ones(self.data.raw_dim());
            }
            assert!(
                self.grad.borrow().sum().abs() >= f32::EPSILON,
                "grad must not be None"
            );
            let grads = ctx_ref.arg.backward(&ctx_ref, &self.grad.borrow());
        }
    }
}

pub trait Function {
    fn apply(&self, ctx: &mut Context, inputs: &[ArrayD<f32>]) -> ArrayD<f32>;
    fn backward(&self, ctx: &mut Context, grad_output: ArrayD<f32>) -> Vec<ArrayD<f32>>;
}

struct Dot;

impl Function for Dot {
    fn apply(&self, ctx: &mut Context, inputs: &[ArrayD<f32>]) -> ArrayD<f32> {
        // Ensure there are exactly two inputs: input and weight
        assert_eq!(inputs.len(), 2, "Dot function expects exactly 2 inputs.");

        let input = &inputs[0];
        let weight = &inputs[1];
        // Convert ArrayD<f32> to Array2<f32> for dot operation if necessary
        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Input must be 2D for dot product.");
        let weight_2d = weight
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Weight must be 2D for dot product.");
        let result = input_2d.dot(&weight_2d);

        let reconstructed = ArrayD::from_shape_vec(
            result.shape().to_owned().into(),
            result.iter().cloned().collect(),
        )
        .expect("Conversion to ArrayD failed");

        reconstructed
    }

    fn backward(&self, ctx: &mut Context, grad_output: ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let a = &ctx.saved_tensors[0];
        let b = &ctx.saved_tensors[1];
        let grad_a = grad_output.dot(b);
        let grad_b = a.dot(&grad_output);
        vec![grad_a, grad_b]
    }
}

fn main() {}
