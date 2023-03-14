use std::rc::Rc;

use ndarray::{Dimension, ArrayViewMut, ArrayView, Ix3, Ix1, Array};

use crate::{utils::Shared, gradient::Gradient, autograd::{Forward, Backward}};

type SampleDim<D> = <<D as Dimension>::Smaller as Dimension>::Smaller;

pub trait PoolingMode<D>: Send + Sync + Copy + 'static
where D: Dimension {
    fn pool(
        &self,
        pooled: &mut ArrayViewMut<f32, D>,
        base: &ArrayView<f32, D>,
        pooling: SampleDim<D>,
    );

    fn pool_back(
        &self,
        pooled_grad: &ArrayView<f32, D>,
        base_grad: &mut ArrayViewMut<f32, D>,
        base: &ArrayView<f32, D>,
        pooling: SampleDim<D>,
    );
}

fn max(array: ArrayView<'_, f32, impl Dimension>) -> Option<f32> {
    array.into_iter().cloned().reduce(|x,y| x.max(y))
}

#[derive(Clone, Copy)]
pub struct MaxPool;

unsafe impl Send for MaxPool{}
unsafe impl Sync for MaxPool{}

impl<D: Dimension> PoolingMode<D> for MaxPool {
    fn pool(
        &self,
        pooled: &mut ArrayViewMut<f32, D>,
        base: &ArrayView<f32, D>,
        pooling: SampleDim<D>,
    ) {
        let mut bigger = D::default();
        bigger.as_array_view_mut().assign(&Array::from_iter(Array::from_vec(vec![1,1]).into_iter().chain(pooling.as_array_view().to_owned().into_iter())));
        let x = base.exact_chunks(bigger);
        x.into_iter().zip(pooled.iter_mut()).for_each(|(m, p)| *(p)=max(m).unwrap());
    }

    fn pool_back(
        &self,
        pooled_grad: &ArrayView<f32, D>,
        base_grad: &mut ArrayViewMut<f32, D>,
        base: &ArrayView<f32, D>,
        pooling: SampleDim<D>,
    ) {
        let mut bigger = D::default();
        bigger.as_array_view_mut().assign(&Array::from_iter(Array::from_vec(vec![1,1]).into_iter().chain(pooling.as_array_view().to_owned().into_iter())));
        let bgw = base_grad.exact_chunks_mut(bigger.clone());
        let bw = base.exact_chunks(bigger);
        bgw.into_iter().zip(bw.into_iter()).zip(pooled_grad.iter()).for_each(|((mut m, i), p)| {
            let mut mp : Option<(&mut f32, &f32)> = None;
            for i in m.iter_mut().zip(i.iter()) {
                match &mut mp {
                    Some(mx) => {
                        if i.1>mx.1 {
                            mp = Some(i);
                        }
                    },
                    None => {
                        mp = Some(i)
                    },
                }
            }
            *(mp.unwrap().0) += *p;
        });
    }
}

pub trait Pooling<D>
where D: Dimension,
{
    type Output;

    fn pooling(self, pooling: SampleDim<D>, pooling_type: impl PoolingMode<D>) -> Self::Output;
}

pub(crate) struct Pool<D: Dimension, P: PoolingMode<D>> {
    data: Shared<Array<f32, D>>,
    out: Shared<Array<f32, D>>,
    pooling: SampleDim<D>,
    pooling_mode: P,
}

impl<D: Dimension, P: PoolingMode<D>> Pool<D, P> {
    pub(crate) fn new(data: Shared<Array<f32, D>>,out: Shared<Array<f32, D>>, pooling: SampleDim<D>, pooling_mode: P) ->Self {
        Self{ data, out, pooling, pooling_mode}
    }
}

impl<D: Dimension, P: PoolingMode<D>> Forward for Pool<D, P> {
    fn forward(&self) {
        self.pooling_mode.pool(&mut self.out.borrow_mut().view_mut(), 
        &self.data.borrow_mut().view(), 
        self.pooling.clone())
    }
}

pub(crate) struct PoolBackward<D: Dimension, P: PoolingMode<D>> {
    data: Shared<Array<f32, D>>,
    data_gradient: Rc<Gradient<Array<f32, D>, D>>,
    gradient: Rc<Gradient<Array<f32, D>, D>>,
    pooling: SampleDim<D>,
    pooling_mode: P,
}

impl<D: Dimension, P: PoolingMode<D>> PoolBackward<D, P> {
    pub(crate) fn new(
        data: Shared<Array<f32, D>>,
        data_gradient: Rc<Gradient<Array<f32, D>, D>>, 
        gradient: Rc<Gradient<Array<f32, D>, D>>, 
        pooling: SampleDim<D>, 
        pooling_mode: P) -> Self { 
            Self { data, data_gradient, gradient, pooling, pooling_mode } 
        }
}

impl<D: Dimension, P: PoolingMode<D>> Backward for PoolBackward<D, P> {
    fn backward(&self) {
        self.pooling_mode.pool_back(
            &self.gradient.borrow().view(), 
            &mut self.data_gradient.borrow_mut().view_mut(), 
            &self.data.borrow().view(), 
            self.pooling.clone())
    }
}

#[cfg(test)]
mod test {
    use ndarray::{Array, Ix2, Ix4};

    use crate::{MaxPool, PoolingMode};

    #[test]
    fn max_pool() {
        let v = Array::from_shape_vec(Ix4(1,1,4,4), 
            vec![
                1.0, 2.0, -1.0, 3.0, 
                4.0, 5.0, 0.0, 1.0, 
                2.0, -3.0, 1.0, 0.0,
                1.0, 0.0, 0.0, 0.0,
            ]    
        ).unwrap();
        let mut p = Array::zeros(Ix4(1,1,2,2));
        MaxPool.pool(&mut p.view_mut(), &v.view(), Ix2(2, 2));
        assert_eq!(p, Array::from_shape_vec(Ix4(1,1,2,2), vec![5.0, 3.0, 2.0, 1.0]).unwrap());
    }
}