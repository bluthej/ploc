use numpy::{ndarray::Array, IntoPyArray, PyArrayDyn, PyReadonlyArray2};
use ploc_rs::PointLocator;
use pyo3::prelude::*;

#[pyclass]
struct PyTrapMap(ploc_rs::TrapMap);

#[pyclass(rename_all = "UPPERCASE")]
#[derive(Clone)]
enum Method {
    Sequential,
    Parallel,
    Auto,
}

#[pymethods]
impl PyTrapMap {
    #[new]
    fn new(points: PyReadonlyArray2<'_, f64>, cells: PyReadonlyArray2<'_, isize>) -> Self {
        let points = points.as_array();
        let cells = cells.as_array();
        let nf = cells.shape()[1];
        let cells: Vec<_> = cells.iter().map(|&i| i as usize).collect();
        let points: Vec<[f64; 2]> = points.outer_iter().map(|row| [row[0], row[1]]).collect();
        let mesh = ploc_rs::Mesh::with_stride(points, cells, nf).unwrap();
        Self(ploc_rs::TrapMap::from_mesh(mesh))
    }

    fn locate_many<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray2<'_, f64>,
        method: Option<Method>,
    ) -> Bound<'py, PyArrayDyn<isize>> {
        let method = method.unwrap_or(Method::Sequential);

        let query = query.as_array();
        let query: Vec<[f64; 2]> = query.outer_iter().map(|row| [row[0], row[1]]).collect();

        let res = match method {
            Method::Sequential => self.0.locate_many(&query),
            Method::Parallel => self.0.par_locate_many(&query),
            Method::Auto => todo!("This should determine heuristically whether to use the sequential or parallel version based on the size of the query"),
        };

        Array::from_iter(res.iter().map(|r| r.map_or(-1, |i| i as isize)))
            .into_dyn()
            .into_pyarray_bound(py)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn ploc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrapMap>()?;
    m.add_class::<Method>()?;
    Ok(())
}
