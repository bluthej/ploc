use std::slice::Iter;

use anyhow::{anyhow, Result};

/// An array-based representation of a mesh.
///
/// The internal representation of the mesh is:
/// - An array of vertex coordinates
/// - An array of polygon vertex indices
/// - An array of offsets
///
/// In the offsets array, the `i`th element is the first vertex id of cell number `i`. This means
/// that all the vertices of cell `i` are defined by the range `offsets[i]..offsets[i+1]`. Note that
/// the last element of the offsets array is the length of the polygon vertices array.
///
/// Note: There is an optimization for single cell type meshes whereby we don't actually store the
/// offsets array explicitly. Instead, the stride (i.e. the number of vertices per polygon) is stored,
/// and the offsets for cell `i` are simply `i*stride` and `(i+1)*stride`.
///
/// This representation of a mesh is found in several places, two of which are:
/// - [axom](https://axom.readthedocs.io/en/develop/axom/mint/docs/sphinx/sections/mesh_types.html#mixedcelltopology)
/// - [geogram](https://github.com/BrunoLevy/geogram/wiki/Mesh#triangulated-and-polygonal-meshes)
/// It does have the advantage of allowing both single cell and mixed cell type topologies
/// (see the above link to the axom documentation).
#[derive(Debug, Clone)]
pub(crate) struct Mesh {
    points: Vec<[f32; 2]>,
    cells: Vec<usize>,
    offsets: Offsets,
}

#[derive(Debug, Clone)]
enum Offsets {
    Implicit(usize),
    Explicit(Vec<usize>),
}

pub(crate) struct Cells<'a> {
    cells: &'a [usize],
    offsets: &'a Offsets,
    idx: usize,
}

impl<'a> Iterator for Cells<'a> {
    type Item = &'a [usize];

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.idx;
        let (start, end) = match self.offsets {
            Offsets::Implicit(stride) if idx * stride < self.cells.len() => {
                (idx * stride, (idx + 1) * stride)
            }
            Offsets::Explicit(offsets) if idx + 1 < offsets.len() => {
                (offsets[idx], offsets[idx + 1])
            }
            _ => return None,
        };
        self.idx += 1;
        // This iterator can only be created from a valid `Mesh` so there cannot be bounds issues
        Some(&self.cells[start..end])
    }
}

impl Mesh {
    /// Constructs a new `Mesh` from its array representation.
    pub(crate) fn new(
        points: Vec<[f32; 2]>,
        cells: Vec<usize>,
        offsets: Vec<usize>,
    ) -> Result<Self> {
        Self::check_ids(points.len(), &cells)?;

        if let Some(&last) = offsets.last() {
            if last != cells.len() {
                return Err(anyhow!("The last element of `offsets` should be the length of the cells array, got {} and {}", last, cells.len()));
            }
        } else {
            return Err(anyhow!("The `offsets` array should have a length > 0"));
        }

        Ok(Self {
            points,
            cells,
            offsets: Offsets::Explicit(offsets),
        })
    }

    /// Constructs a new single cell type `Mesh` from its array representation and a stride.
    ///
    /// The offsets are stored implicitly, which leads to less memory usage, but this is only
    /// possible for single cell type topologies.
    pub(crate) fn with_stride(
        points: Vec<[f32; 2]>,
        cells: Vec<usize>,
        stride: usize,
    ) -> Result<Self> {
        Self::check_ids(points.len(), &cells)?;

        if cells.is_empty() && stride > 0 {
            return Err(anyhow!("Cannot have a positive stride with an empty mesh"));
        }

        if cells.len() % stride != 0 {
            return Err(anyhow!(
                "The `stride` should evenly divide the length of the `cells`, got {} and {}.",
                stride,
                cells.len(),
            ));
        }

        Ok(Self {
            points,
            cells,
            offsets: Offsets::Implicit(stride),
        })
    }

    fn check_ids(n: usize, cells: &[usize]) -> Result<()> {
        if cells.iter().any(|&idx| idx >= n) {
            Err(anyhow!(
                "There are vertex ids in `cells` greater than the number of given `points` ({})",
                n
            ))
        } else {
            Ok(())
        }
    }

    pub(crate) fn cell_count(&self) -> usize {
        match &self.offsets {
            Offsets::Implicit(stride) => self.cells.len() / stride,
            Offsets::Explicit(offsets) => offsets.len() - 1,
        }
    }

    pub(crate) fn vertex_count(&self) -> usize {
        self.points.len()
    }

    pub(crate) fn facet_count(&self) -> usize {
        self.cells.len()
    }

    pub(crate) fn cells(&self) -> Cells<'_> {
        Cells {
            cells: &self.cells,
            offsets: &self.offsets,
            idx: 0,
        }
    }

    pub(crate) fn points(&self) -> Iter<[f32; 2]> {
        self.points.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_mesh_with_stride() -> Result<()> {
        let points = vec![[0., 0.], [1., 0.], [0., 1.]];
        let cells = vec![0, 1, 2];
        let mesh = Mesh::with_stride(points, cells, 3)?;

        assert_eq!(mesh.cell_count(), 1);

        Ok(())
    }

    #[test]
    fn create_mesh_with_offsets() -> Result<()> {
        let points = vec![[0., 0.], [1., 0.], [0., 1.]];
        let cells = vec![0, 1, 2];
        let offsets = vec![0, 3];
        let mesh = Mesh::new(points, cells, offsets)?;

        assert_eq!(mesh.cell_count(), 1);

        Ok(())
    }

    #[test]
    fn create_mesh_with_mixed_cell_types() -> Result<()> {
        let points = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0.5, 1.5]];
        let cells = vec![0, 1, 2, 3, 3, 2, 4];
        let offsets = vec![0, 4, 7];
        let mesh = Mesh::new(points, cells, offsets)?;

        assert_eq!(mesh.cell_count(), 2);

        Ok(())
    }

    #[test]
    fn iterate_over_cells_with_single_cell_type() -> Result<()> {
        let points = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
        let cells = vec![0, 1, 3, 1, 2, 3];
        let mesh = Mesh::with_stride(points, cells, 3)?;

        let mut cells = mesh.cells();

        assert_eq!(cells.next(), Some([0, 1, 3].as_slice()));
        assert_eq!(cells.next(), Some([1, 2, 3].as_slice()));
        assert_eq!(cells.next(), None);

        Ok(())
    }

    #[test]
    fn iterate_over_cells_with_mixed_cell_type() -> Result<()> {
        let points = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0.5, 1.5]];
        let cells = vec![0, 1, 2, 3, 3, 2, 4];
        let offsets = vec![0, 4, 7];
        let mesh = Mesh::new(points, cells, offsets)?;

        let mut cells = mesh.cells();

        assert_eq!(cells.next(), Some([0, 1, 2, 3].as_slice()));
        assert_eq!(cells.next(), Some([3, 2, 4].as_slice()));
        assert_eq!(cells.next(), None);

        Ok(())
    }

    #[test]
    fn invalid_cell_array_returns_error() {
        let points = vec![[0., 0.], [1., 0.], [0., 1.]];
        let cells = vec![0, 1, 3];

        let mesh = Mesh::with_stride(points, cells, 3);

        assert!(mesh.is_err());
    }

    #[test]
    fn invalid_stride_returns_error() {
        let points = vec![[0., 0.], [1., 0.], [0., 1.]];
        let cells = vec![0, 1, 2];

        let mesh = Mesh::with_stride(points, cells, 2);

        assert!(mesh.is_err());
    }

    #[test]
    fn last_item_of_offsets_is_not_length_of_cells_return_error() {
        let points = vec![[0., 0.], [1., 0.], [0., 1.]];
        let cells = vec![0, 1, 2];
        let offsets = vec![0, 4];

        let mesh = Mesh::new(points, cells, offsets);

        assert!(mesh.is_err());
    }

    #[test]
    fn empty_offsets_array_return_error() {
        let points = Vec::new();
        let cells = Vec::new();
        let offsets = Vec::new();

        let mesh = Mesh::new(points, cells, offsets);

        assert!(mesh.is_err());
    }

    #[test]
    fn create_empty_mesh() -> Result<()> {
        let points = Vec::new();
        let cells = Vec::new();
        let offsets = vec![0];

        let mesh = Mesh::new(points, cells, offsets)?;

        assert_eq!(mesh.cell_count(), 0);

        Ok(())
    }

    #[test]
    fn empty_mesh_with_non_zero_offset_returns_error() {
        // With offsets
        let mesh = Mesh::new(Vec::new(), Vec::new(), vec![1]);

        assert!(mesh.is_err());

        // With stride
        let mesh = Mesh::with_stride(Vec::new(), Vec::new(), 1);

        assert!(mesh.is_err());
    }
}
