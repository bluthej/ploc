use numpy::{ndarray::Array, PyArrayDyn, PyReadonlyArray2};
use ploc_rs::PointLocator;
use pyo3::prelude::*;

#[pyclass]
struct TrapMap(ploc_rs::TrapMap);

#[pymethods]
impl TrapMap {
    #[new]
    fn new(points: PyReadonlyArray2<'_, f64>, cells: PyReadonlyArray2<'_, isize>) -> Self {
        let points = points.as_array();
        let cells = cells.as_array();
        let nf = cells.shape()[1];
        let cells: Vec<_> = cells.iter().map(|&i| i as usize).collect();
        let points: Vec<[f64; 2]> = points.outer_iter().map(|row| [row[0], row[1]]).collect();
        let mesh = ploc_rs::Mesh::with_stride(points, cells, nf).unwrap();
        Self(ploc_rs::TrapMap::from_mesh(mesh).build())
    }

    fn locate_many<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray2<'_, f64>,
    ) -> Bound<'py, PyArrayDyn<isize>> {
        let query = query.as_array();
        let query: Vec<[f64; 2]> = query.outer_iter().map(|row| [row[0], row[1]]).collect();
        let res = self.0.locate_many(&query);
        let res =
            Array::from_iter(res.iter().map(|r| r.map(|i| i as isize).unwrap_or(-1))).into_dyn();
        PyArrayDyn::from_owned_array_bound(py, res)
    }

    fn par_locate_many<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray2<'_, f64>,
    ) -> Bound<'py, PyArrayDyn<isize>> {
        let query = query.as_array();
        let query: Vec<[f64; 2]> = query.outer_iter().map(|row| [row[0], row[1]]).collect();
        let res = self.0.par_locate_many(&query);
        let res =
            Array::from_iter(res.iter().map(|r| r.map(|i| i as isize).unwrap_or(-1))).into_dyn();
        PyArrayDyn::from_owned_array_bound(py, res)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn ploc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TrapMap>()?;
    Ok(())
}
