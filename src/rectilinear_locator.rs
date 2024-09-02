use anyhow::{anyhow, Result};
use itertools::Itertools;

use crate::point_locator::PointLocator;

/// A point locator for a rectilinear grid.
pub struct RectilinearLocator {
    pub(crate) x: Vec<f64>,
    pub(crate) y: Vec<f64>,
}

impl RectilinearLocator {
    /// Constructs a new `RectilinearLocator`.
    ///
    /// Fails if either the `x` or `y` vector is not sorted and strictly increasing.
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self> {
        for z in [&x, &y] {
            if z.iter().tuple_windows().any(|(z1, z2)| z1 >= z2) {
                return Err(anyhow!("The input values should be strictly increasing."));
            }
        }

        Ok(Self { x, y })
    }
}

impl PointLocator for RectilinearLocator {
    /// Locates a point in a rectilinear mesh.
    ///
    /// The location is performed with two binary searches: one on the x-axis and one on the y-axis.
    ///
    /// The returned cell index is computed assuming that the mesh cells are numbered first from
    /// left to right, and then from bottom to top, i.e. cell `0` is always the bottom-left corner,
    /// then the next cell is its right neighbor if it exists or its top neighbor otherwise.
    /// This continues until the end of the bottom row is reached, and then we continue to the
    /// second row.
    fn locate_one(&self, [xp, yp]: &[f64; 2]) -> Option<usize> {
        let nx = self.x.len();
        let ny = self.y.len();
        let x_idx = self.x.partition_point(|x| x <= xp);
        let y_idx = self.y.partition_point(|y| y <= yp);
        if x_idx > 0 && x_idx < nx && y_idx > 0 && y_idx < ny {
            Some((nx - 1) * (y_idx - 1) + x_idx - 1)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::mesh::Mesh;
    use crate::winding_number::Point;
    use anyhow::Result;
    use itertools::Itertools;
    use proptest::prelude::*;

    use super::*;

    prop_compose! {
        fn coords_in_range(xmin: f64, xmax: f64, ymin: f64, ymax: f64)
                          (x in xmin..xmax, y in ymin..ymax) -> [f64; 2] {
           [x, y]
        }
    }

    #[test]
    fn unsorted_input_values_in_rectilinear_locator_returns_error() {
        // x or y has reversed elements
        assert!(RectilinearLocator::new(vec![0., 2., 1.], vec![0., 1., 2.]).is_err());
        assert!(RectilinearLocator::new(vec![0., 1., 1.], vec![0., 2., 1.]).is_err());
        // x or y has two consecutive equal elements
        assert!(RectilinearLocator::new(vec![0., 1., 1.], vec![0., 1., 2.]).is_err());
        assert!(RectilinearLocator::new(vec![0., 1., 2.], vec![0., 1., 1.]).is_err());
    }

    #[test]
    fn rectilinear_locator_simple_test() -> Result<()> {
        let x = vec![0., 1., 2.];
        let y = vec![0., 1., 2.];
        let points = vec![[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5], [2.5, 2.5]];
        let locator = RectilinearLocator::new(x, y)?;

        let locations = locator.locate_many(&points);

        assert_eq!(locations, vec![Some(0), Some(1), Some(2), Some(3), None]);

        // Check results using the winding number
        let mesh = Mesh::grid(0., 2., 0., 2., 2, 2).unwrap();
        for (point, idx) in points.iter().map(Point::from).zip(&locations) {
            if let Some(idx) = idx {
                let cell = mesh.cell_vertices(*idx).cloned();
                assert!(point.is_inside(cell));
            }
        }

        Ok(())
    }

    #[test]
    fn rectilinear_locator_edge_cases() -> Result<()> {
        let x = vec![0., 1., 2.];
        let y = vec![0., 1., 2.];
        let points = vec![
            // Vertical edges
            [0., 0.5],
            [1., 0.5],
            [2., 0.5], // Right edge counts as outside
            [0., 1.5],
            [1., 1.5],
            [2., 1.5], // Right edge counts as outside
            // Horizontal edges
            [0.5, 0.],
            [0.5, 1.],
            [0.5, 2.], // Top edge counts as outside
            [1.5, 0.],
            [1.5, 1.],
            [1.5, 2.], // Top edge counts as outside
        ];
        let locator = RectilinearLocator::new(x, y)?;

        let locations = locator.locate_many(&points);

        assert_eq!(
            locations,
            vec![
                Some(0),
                Some(1),
                None,
                Some(2),
                Some(3),
                None,
                Some(0),
                Some(2),
                None,
                Some(1),
                Some(3),
                None
            ]
        );

        // Check results using the winding number
        let mesh = Mesh::grid(0., 2., 0., 2., 2, 2).unwrap();
        for (point, idx) in points.iter().map(Point::from).zip(&locations) {
            if let Some(idx) = idx {
                let cell = mesh.cell_vertices(*idx).cloned();
                assert!(point.is_inside(cell));
            }
        }

        Ok(())
    }

    #[test]
    fn rectilinear_locator_corner_cases() -> Result<()> {
        let x = vec![0., 1., 2.];
        let y = vec![0., 1., 2.];
        let points = vec![
            [0., 0.],
            [1., 0.],
            [2., 0.], // Right edge counts as outside
            [0., 1.],
            [1., 1.],
            [2., 1.], // Right edge counts as outside
            [0., 2.], // Top edge counts as outside
            [1., 2.], // Top edge counts as outside
            [2., 2.], // Top/Right edge counts as outside
        ];
        let locator = RectilinearLocator::new(x, y)?;

        let locations = locator.locate_many(&points);

        assert_eq!(
            locations,
            vec![
                Some(0),
                Some(1),
                None,
                Some(2),
                Some(3),
                None,
                None,
                None,
                None
            ]
        );

        // Check results using the winding number
        let mesh = Mesh::grid(0., 2., 0., 2., 2, 2).unwrap();
        for (point, idx) in points.iter().map(Point::from).zip(&locations) {
            if let Some(idx) = idx {
                let cell = mesh.cell_vertices(*idx).cloned();
                assert!(point.is_inside(cell));
            }
        }

        Ok(())
    }

    #[test]
    fn rectilinear_locator_proptest() -> Result<()> {
        let (xmin, xmax) = (0., 10.);
        let (ymin, ymax) = (0., 10.);
        let (nx, ny) = (6, 6); // Use numbers that don't divide the sides evenly on purpose

        // Create rectilinear locator
        let dx = (xmax - xmin) / nx as f64;
        let x = (0..=nx).map(|n| xmin + n as f64 * dx).collect_vec();
        let dy = (ymax - ymin) / ny as f64;
        let y = (0..=ny).map(|n| ymin + n as f64 * dy).collect_vec();
        let locator = RectilinearLocator::new(x, y)?;

        // Create `Mesh` to check the results using the winding number
        let mesh = Mesh::grid(xmin, xmax, ymin, ymax, nx, ny).unwrap();

        // Select the number of points generated. The higher it is, the more time the test takes.
        let np = 20;
        proptest!(|(points in proptest::collection::vec(coords_in_range(xmin, xmax, ymin, ymax), np))| {
            let locations = locator.locate_many(&points);

            // Check results using the winding number
            for (point, idx) in points.iter().map(Point::from).zip(&locations) {
                let Some(idx) = idx else {
                    panic!("All points should be in a cell but {:?} is not", &point);
                };
                let cell = mesh.cell_vertices(*idx).cloned();
                assert!(point.is_inside(cell));
            }
        });

        Ok(())
    }
}
