use std::cell::RefCell;
use std::rc::Rc;

use ndarray::{ArrayD, Ix2};

struct Tensor {
    data: ArrayD<f32>,
    grad: RefCell<ArrayD<f32>>,
    ctx: Option<Rc<RefCell<Function>>>,
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
            let grads =
                ctx.borrow().backward(ctx_ref.saved_tensors.clone(), self.grad.borrow().clone());

            for (parent, grad) in ctx_ref.parents.iter().zip(grads) {
                *parent.grad.borrow_mut() += &grad;
                parent.backward(false);
            }
        }
    }
}

#[derive(Clone)]
struct Function {
    parents: Vec<Rc<Tensor>>,
    saved_tensors: Vec<ArrayD<f32>>,
}

trait FunctionTrait {
    fn apply(&self, inputs: &[ArrayD<f32>]) -> ArrayD<f32>;
    fn backward(
        &self,
        saved_tensors: Vec<ArrayD<f32>>,
        grad_output: ArrayD<f32>,
    ) -> Vec<ArrayD<f32>>;
}

impl Function {
    fn new(parents: Vec<Rc<Tensor>>) -> Self {
        Function { parents, saved_tensors: vec![] }
    }

    fn save_for_backward(&mut self, tensors: ArrayD<f32>) {
        self.saved_tensors.push(tensors);
    }
}

impl FunctionTrait for Function {
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
    a.iter().zip(b.iter()).all(|(&a, &b)| (a - b).abs() <= tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_backward() {
        // Create a tensor with some data
        let data = ArrayD::<f32>::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let tensor = Tensor::new(data);

        // Call the backward method
        tensor.backward(false);

        // Assert that the gradients have been updated correctly
        let grad = tensor.grad.borrow();
        assert_eq!(grad.shape(), &[2, 2]);
        assert_eq!(grad[[0, 0]], 0.0);
        assert_eq!(grad[[0, 1]], 0.0);
        assert_eq!(grad[[1, 0]], 0.0);
        assert_eq!(grad[[1, 1]], 0.0);
    }

    #[test]
    fn test_function_apply() {
        // Create a function
        let function = Function::new(vec![]);

        // Create input arrays
        let input1 = ArrayD::<f32>::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let input2 = ArrayD::<f32>::from_shape_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let inputs = vec![input1, input2];

        // Call the apply method
        let result = function.apply(&inputs);

        // Assert that the result is correct
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 19.0);
        assert_eq!(result[[0, 1]], 22.0);
        assert_eq!(result[[1, 0]], 43.0);
        assert_eq!(result[[1, 1]], 50.0);
    }

    #[test]
    fn test_function_backward() {
        // Create a function
        let function = Function::new(vec![]);

        // Create saved tensors and gradient output
        let saved_tensors = vec![
            ArrayD::<f32>::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            ArrayD::<f32>::from_shape_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap(),
        ];
        let grad_output =
            ArrayD::<f32>::from_shape_vec(vec![2, 2], vec![9.0, 10.0, 11.0, 12.0]).unwrap();

        // Call the backward method
        let grads = function.backward(saved_tensors, grad_output);

        // Assert that the gradients are correct
        assert_eq!(grads.len(), 2);

        let grad_a = &grads[0];
        assert_eq!(grad_a.shape(), &[2, 2]);
        assert_eq!(grad_a[[0, 0]], 105.0);
        assert_eq!(grad_a[[0, 1]], 143.0);
        assert_eq!(grad_a[[1, 0]], 127.0);
        assert_eq!(grad_a[[1, 1]], 173.0);

        let grad_b = &grads[1];
        assert_eq!(grad_b.shape(), &[2, 2]);
        assert_eq!(grad_b[[0, 0]], 42.0);
        assert_eq!(grad_b[[0, 1]], 46.0);
        assert_eq!(grad_b[[1, 0]], 62.0);
        assert_eq!(grad_b[[1, 1]], 68.0);
    }

    #[test]
    fn test_are_tensors_close() {
        let a = ArrayD::<f32>::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = ArrayD::<f32>::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(are_tensors_close(&a, &b, 0.0));

        let c = ArrayD::<f32>::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let d = ArrayD::<f32>::from_shape_vec(vec![2, 2], vec![1.1, 2.1, 3.1, 4.1]).unwrap();
        assert!(!are_tensors_close(&c, &d, 0.1));
    }
}
fn main() {}
