use rayon::prelude::*;

/// A trait to locate one or several query points within a mesh.
pub trait PointLocator {
    /// Locates one query point within a mesh.
    ///
    /// Returns [`None`] if the query point does not lie in any cell of the mesh.
    fn locate_one(&self, point: &[f64; 2]) -> Option<usize>;

    /// Locates several query points within a mesh.
    fn locate_many(&self, points: &[[f64; 2]]) -> Vec<Option<usize>> {
        points.iter().map(|point| self.locate_one(point)).collect()
    }

    /// Locates several query points within a mesh in parallel.
    fn par_locate_many(&self, points: &[[f64; 2]]) -> Vec<Option<usize>>
    where
        Self: std::marker::Sync,
    {
        points
            .par_iter()
            .map(|point| self.locate_one(point))
            .collect()
    }
}
