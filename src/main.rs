use std::cell::RefCell;
use std::rc::Rc;

use ndarray::{ArrayD, Ix2};
// Context to hold intermediate values for backward pass
struct Context {
    arg: Rc<RefCell<dyn Function>>,
    parents: Vec<Rc<Tensor>>,
    saved_tensors: Vec<ArrayD<f32>>,
}

impl Context {
    fn new(arg: Rc<RefCell<dyn Function>>, parents: Vec<Rc<Tensor>>) -> Self {
        Context { arg, parents, saved_tensors: vec![] }
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
        Tensor { data, grad: RefCell::new(grad), ctx: None }
    }

    fn backward(&self, allow_fill: bool) {
        if let Some(ctx) = &self.ctx {
            let ctx_ref = ctx.borrow();
            if self.grad.borrow().sum().abs() < f32::EPSILON && !allow_fill {
                assert!(self.data.len() == 1, "data size must be 1 for auto-fill gradient");
                *self.grad.borrow_mut() = ArrayD::ones(self.data.raw_dim());
            }
            assert!(self.grad.borrow().sum().abs() >= f32::EPSILON, "grad must not be None");
            let grads = ctx_ref
                .arg
                .borrow()
                .backward(ctx_ref.saved_tensors.clone(), self.grad.borrow().clone());

            for (parent, grad) in ctx_ref.parents.iter().zip(grads) {
                *parent.grad.borrow_mut() += &grad;
                parent.backward(false);
            }
        }
    }
}

pub trait Function {
    fn apply(&self, inputs: &[ArrayD<f32>]) -> ArrayD<f32>;
    fn backward(
        &self,
        saved_tensors: Vec<ArrayD<f32>>,
        grad_output: ArrayD<f32>,
    ) -> Vec<ArrayD<f32>>;
}

struct Dot;

impl Function for Dot {
    fn apply(&self, inputs: &[ArrayD<f32>]) -> ArrayD<f32> {
        // Ensure there are exactly two inputs: input and weight
        assert_eq!(inputs.len(), 2, "Dot function expects exactly 2 inputs.");

        let input = &inputs[0];
        let weight = &inputs[1];
        // Convert ArrayD<f32> to Array2<f32> for dot operation if necessary
        let input_2d =
            input.view().into_dimensionality::<Ix2>().expect("Input must be 2D for dot product.");
        let weight_2d =
            weight.view().into_dimensionality::<Ix2>().expect("Weight must be 2D for dot product.");
        let result = input_2d.dot(&weight_2d);
        result.into_dyn()
    }

    fn backward(
        &self,
        saved_tensors: Vec<ArrayD<f32>>,
        grad_output: ArrayD<f32>,
    ) -> Vec<ArrayD<f32>> {
        let grad_output_2d = grad_output
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Grad output must be 2D for dot product.");
        let a_2d = saved_tensors[0].view().into_dimensionality::<Ix2>().unwrap();
        let b_2d = saved_tensors[1].view().into_dimensionality::<Ix2>().unwrap();

        let grad_a = grad_output_2d.dot(&b_2d.t());
        let grad_b = a_2d.t().dot(&grad_output_2d);
        vec![grad_a.into_dyn(), grad_b.into_dyn()]
    }
}

pub fn are_tensors_close(a: &ArrayD<f32>, b: &ArrayD<f32>, tolerance: f32) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(&a, &b)| (a - b).abs() < tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array};

    #[test]
    fn test_dot_operation_and_backward() {
        // Define input and weight tensors
        let input_data = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let weight_data =
            Array::from_shape_vec((3, 2), vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5]).unwrap();

        let input_tensor = Rc::new(Tensor::new(input_data.into_dyn()));
        let weight_tensor = Rc::new(Tensor::new(weight_data.into_dyn()));

        let dot = Dot;
        let mut context = Context::new(Rc::new(RefCell::new(dot)), vec![]);
        context.save_for_backward(input_tensor.data.clone());
        context.save_for_backward(weight_tensor.data.clone());

        let inputs = vec![input_tensor.data.clone(), weight_tensor.data.clone()];
        let result = context.arg.borrow().apply(&inputs);

        // Expected result of the dot operation
        let expected_result = arr2(&[[19.0, 25.0], [41.5, 56.5]]).into_dyn();
        assert!(
            are_tensors_close(&result, &expected_result, 1e-6),
            "Dot operation result does not match expected."
        );

        // Simulate backward propagation with a gradient output
        let grad_output =
            Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap().into_dyn();
        let grads = context.arg.borrow().backward(context.saved_tensors, grad_output);
        // Calculate expected gradients manually or through another method
        let expected_grad_input = arr2(&[[3.5, 9.5, 15.5], [7.5, 21.5, 35.5]]).into_dyn();
        let expected_grad_weight = arr2(&[[13.0, 18.0], [17.0, 24.0], [21.0, 30.0]]).into_dyn();

        // Extract the computed gradients for input and weight
        let grad_input_computed = &grads[0];
        let grad_weight_computed = &grads[1];
        assert!(
            are_tensors_close(&expected_grad_input, &grad_input_computed, 1e-6),
            "Dot operation result does not match expected."
        );
        assert!(
            are_tensors_close(&expected_grad_weight, &grad_weight_computed, 1e-6),
            "Dot operation result does not match expected."
        );
    }
}

fn main() {}
