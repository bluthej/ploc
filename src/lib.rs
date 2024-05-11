#![allow(dead_code)]

mod dag;
mod dcel;
mod mesh;
mod winding_number;

use anyhow::{anyhow, Result};
use dag::Dag;
use dcel::{Dcel, Hedge, HedgeId, VertexId};
use itertools::Itertools;
use mesh::Mesh;
use winding_number::Positioning;

use crate::winding_number::Point;

/// A trait to locate one or several query points within a mesh.
trait PointLocator {
    /// Locates one query point within a mesh.
    ///
    /// Returns [`None`] if the query point does not lie in any cell of the mesh.
    fn locate_one(&self, point: &[f64; 2]) -> Option<usize>;

    /// Locates several query points within a mesh.
    fn locate_many(&self, points: &[[f64; 2]]) -> Vec<Option<usize>> {
        points.iter().map(|point| self.locate_one(point)).collect()
    }
}

struct TrapMap {
    dcel: Dcel,
    dag: Dag<Node>,
    bbox: BoundingBox,
}

// TODO: add the necessary data
enum Node {
    X(VertexId),
    Y(HedgeId),
    Trap(Trapezoid),
}

// TODO: add leftp and rightp
#[derive(Clone)]
struct Trapezoid {
    top: HedgeId,
    bottom: HedgeId,
}

struct BoundingBox {
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
}

impl BoundingBox {
    fn new(xmin: f64, xmax: f64, ymin: f64, ymax: f64) -> Self {
        Self {
            xmin: xmin - 0.1,
            xmax: xmax + 0.1,
            ymin: ymin - 0.1,
            ymax: ymax + 0.1,
        }
    }
}

impl Default for BoundingBox {
    fn default() -> Self {
        Self {
            xmin: 0.,
            xmax: 1.,
            ymin: 0.,
            ymax: 1.,
        }
    }
}

impl TrapMap {
    fn new() -> Self {
        let dcel = Dcel::new();
        Self::from_dcel(dcel)
    }

    fn from_mesh(mesh: Mesh) -> Self {
        let dcel = Dcel::from_mesh(mesh);
        Self::from_dcel(dcel)
    }

    fn from_dcel(dcel: Dcel) -> Self {
        let mut dag = Dag::new();

        let mut dcel = dcel;
        let top = dcel.add_hedge(Hedge::new());
        let bottom = dcel.add_hedge(Hedge::new());

        dag.add(Node::Trap(Trapezoid { top, bottom }));

        let [xmin, xmax, ymin, ymax] = dcel.get_bounds();
        let bbox = BoundingBox::new(xmin, xmax, ymin, ymax);

        Self { dcel, dag, bbox }
    }

    fn x_node_count(&self) -> usize {
        self.dag
            .iter()
            .filter(|&node| matches!(node.data, Node::X(..)))
            .count()
    }

    fn y_node_count(&self) -> usize {
        self.dag
            .iter()
            .filter(|&node| matches!(node.data, Node::Y(..)))
            .count()
    }

    fn trap_count(&self) -> usize {
        self.dag
            .iter()
            .filter(|&node| matches!(node.data, Node::Trap(..)))
            .count()
    }

    fn node_count(&self) -> (usize, usize, usize) {
        self.dag.iter().fold(
            (0, 0, 0),
            |(mut x_count, mut y_count, mut trap_count), node| {
                match node.data {
                    Node::X(..) => x_count += 1,
                    Node::Y(..) => y_count += 1,
                    Node::Trap(..) => trap_count += 1,
                };
                (x_count, y_count, trap_count)
            },
        )
    }

    fn print_stats(&self) {
        let (x_node_count, y_node_count, trap_count) = self.node_count();
        println!(
            "Trapezoidal map counts:\n\t{} X node(s)\n\t{} Y node(s)\n\t{} trapezoid(s)",
            x_node_count, y_node_count, trap_count,
        );
    }

    fn find_trapezoid(&self, point: &[f64; 2]) -> (usize, &Trapezoid) {
        let mut node_id = 0;
        loop {
            let node = &self.dag.get(node_id).unwrap();
            match &node.data {
                Node::Trap(trapezoid) => return (node_id, trapezoid),
                Node::X(vid) => {
                    let vert: &[f64; 2] = self.dcel.get_vertex(*vid);
                    if point[0] < vert[0] {
                        node_id = node.children[0];
                    } else {
                        node_id = node.children[1];
                    }
                }
                Node::Y(hid) => {
                    let hedge = self.dcel.get_hedge(*hid);
                    let twin = self.dcel.get_hedge(hedge.twin);
                    let p1: &[f64; 2] = self.dcel.get_vertex(hedge.origin);
                    let p2: &[f64; 2] = self.dcel.get_vertex(twin.origin);
                    match Point::from(point).position(*p1, *p2) {
                        Positioning::Right => node_id = node.children[1],
                        _ => node_id = node.children[0],
                    }
                }
            }
        }
    }

    fn add_edge(&mut self, hedge_id: HedgeId) {
        self.print_stats();

        let hedge = self.dcel.get_hedge(hedge_id);
        let twin = self.dcel.get_hedge(hedge.twin);
        let p = self.dcel.get_vertex(hedge.origin);

        let (old_nid, _old_trap) = self.find_trapezoid(p);

        let _a_nid = self
            .dag
            .entry(old_nid)
            .prepend(Node::X(hedge.origin))
            .unwrap();
        let p_nid = old_nid;
        let q_nid = self.dag.entry(p_nid).append(Node::X(twin.origin)).unwrap();
        let s_nid = self.dag.entry(q_nid).append(Node::Y(hedge_id)).unwrap();
        let b_trap = Trapezoid {
            top: HedgeId(0),
            bottom: HedgeId(0),
        };
        let c_trap = b_trap.clone();
        let d_trap = b_trap.clone();
        let _b_nid = self.dag.entry(q_nid).append(Node::Trap(b_trap));
        let _c_nid = self.dag.entry(s_nid).append(Node::Trap(c_trap));
        let _d_nid = self.dag.entry(s_nid).append(Node::Trap(d_trap));

        self.print_stats();
    }
}

impl PointLocator for TrapMap {
    fn locate_one(&self, point: &[f64; 2]) -> Option<usize> {
        let (_, trap) = self.find_trapezoid(point);
        self.dcel.get_hedge(trap.bottom).face.as_deref().copied()
    }
}

/// A point locator for a rectilinear grid.
struct RectilinearLocator {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl RectilinearLocator {
    /// Constructs a new `RectilinearLocator`.
    ///
    /// Fails if either the `x` or `y` vector is not sorted and strictly increasing.
    fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self> {
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
    use crate::winding_number::Point;
    use anyhow::Result;
    use itertools::Itertools;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn initialize_empty_trapezoidal_map() {
        let trap_map = TrapMap::new();

        assert_eq!(trap_map.trap_count(), 1);
    }

    #[test]
    fn find_trap_in_empty_trapezoidal_map() {
        let trap_map = TrapMap::new();
        assert_eq!(trap_map.trap_count(), 1);
        assert_eq!(trap_map.x_node_count(), 0);
        assert_eq!(trap_map.y_node_count(), 0);

        let point = [0., 0.];
        let _trap = trap_map.find_trapezoid(&point);
    }

    #[test]
    fn find_face_in_empty_trapezoidal_map() {
        let trap_map = TrapMap::new();

        let point = [0., 0.];

        assert_eq!(trap_map.locate_one(&point), None);
    }

    #[test]
    fn bounding_box() -> Result<()> {
        let points = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
        let cells = vec![0, 1, 2, 3];
        let mesh = Mesh::with_stride(points, cells, 4)?;
        let trap_map = TrapMap::from_mesh(mesh);

        let bbox = trap_map.bbox;

        assert!(bbox.xmin < 0.);
        assert!(bbox.xmax > 1.);
        assert!(bbox.ymin < 0.);
        assert!(bbox.ymax > 1.);

        Ok(())
    }

    #[test]
    fn add_first_edge() -> Result<()> {
        let points = vec![[0., 0.], [1., 0.], [0.5, 0.5]];
        let cells = vec![0, 1, 2];
        let mesh = Mesh::with_stride(points, cells, 3)?;
        let dcel = Dcel::from_mesh(mesh);
        let mut trap_map = TrapMap::from_dcel(dcel);

        // Add the edge
        trap_map.add_edge(HedgeId(0));

        // Check the number of different nodes
        assert_eq!(trap_map.trap_count(), 4);
        assert_eq!(trap_map.x_node_count(), 2);
        assert_eq!(trap_map.y_node_count(), 1);

        // Check that points in the 4 trapezoids are correctly located
        let (a, _) = trap_map.find_trapezoid(&[-0.1, 0.]);
        assert_eq!(a, 1);
        let (b, _) = trap_map.find_trapezoid(&[1.1, 0.]);
        assert_eq!(b, 4);
        let (c, _) = trap_map.find_trapezoid(&[0.5, 0.5]);
        assert_eq!(c, 5);
        let (d, _) = trap_map.find_trapezoid(&[0.5, -0.5]);
        assert_eq!(d, 6);

        Ok(())
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

    prop_compose! {
        fn coords_in_range(xmin: f64, xmax: f64, ymin: f64, ymax: f64)
                          (x in xmin..xmax, y in ymin..ymax) -> [f64; 2] {
           [x, y]
        }
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
