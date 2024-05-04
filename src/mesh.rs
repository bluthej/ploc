pub(crate) struct Mesh {
    points: Vec<[f32; 2]>,
    cells: Vec<usize>,
    offsets: Offsets,
}

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
    pub(crate) fn with_stride(points: Vec<[f32; 2]>, cells: Vec<usize>, stride: usize) -> Self {
        // TODO: check that input is valid
        Self {
            points,
            cells,
            offsets: Offsets::Implicit(stride),
        }
    }

    pub(crate) fn with_offsets(
        points: Vec<[f32; 2]>,
        cells: Vec<usize>,
        offsets: Vec<usize>,
    ) -> Self {
        // TODO: check that input is valid
        Self {
            points,
            cells,
            offsets: Offsets::Explicit(offsets),
        }
    }

    pub(crate) fn cell_count(&self) -> usize {
        match &self.offsets {
            Offsets::Implicit(stride) => self.cells.len() / stride,
            Offsets::Explicit(offsets) => offsets.len() - 1,
        }
    }

    pub(crate) fn cells(&self) -> Cells<'_> {
        Cells {
            cells: &self.cells,
            offsets: &self.offsets,
            idx: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_mesh_with_stride() {
        let points = vec![[0., 0.], [1., 0.], [0., 1.]];
        let cells = vec![0, 1, 2];
        let mesh = Mesh::with_stride(points, cells, 3);

        assert_eq!(mesh.cell_count(), 1);
    }

    #[test]
    fn create_mesh_with_offsets() {
        let points = vec![[0., 0.], [1., 0.], [0., 1.]];
        let cells = vec![0, 1, 2];
        let offsets = vec![0, 3];
        let mesh = Mesh::with_offsets(points, cells, offsets);

        assert_eq!(mesh.cell_count(), 1);
    }

    #[test]
    fn create_mesh_with_mixed_cell_types() {
        let points = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0.5, 1.5]];
        let cells = vec![0, 1, 2, 3, 3, 2, 4];
        let offsets = vec![0, 4, 7];
        let mesh = Mesh::with_offsets(points, cells, offsets);

        assert_eq!(mesh.cell_count(), 2);
    }

    #[test]
    fn iterate_over_cells_with_single_cell_type() {
        let points = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
        let cells = vec![0, 1, 3, 1, 2, 3];
        let mesh = Mesh::with_stride(points, cells, 3);

        let mut cells = mesh.cells();

        assert_eq!(cells.next(), Some([0, 1, 3].as_slice()));
        assert_eq!(cells.next(), Some([1, 2, 3].as_slice()));
        assert_eq!(cells.next(), None);
    }

    #[test]
    fn iterate_over_cells_with_mixed_cell_type() {
        let points = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0.5, 1.5]];
        let cells = vec![0, 1, 2, 3, 3, 2, 4];
        let offsets = vec![0, 4, 7];
        let mesh = Mesh::with_offsets(points, cells, offsets);

        let mut cells = mesh.cells();

        assert_eq!(cells.next(), Some([0, 1, 2, 3].as_slice()));
        assert_eq!(cells.next(), Some([3, 2, 4].as_slice()));
        assert_eq!(cells.next(), None);
    }
}
