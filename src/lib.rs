#![allow(dead_code)]

mod dag;
mod dcel;
mod mesh;
mod winding_number;

use anyhow::{anyhow, Result};
use dag::Dag;
use dcel::{Dcel, HedgeId, VertexId};
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

#[derive(Clone, Debug)]
struct Trapezoid {
    top: HedgeId,
    bottom: HedgeId,
    rightp: VertexId,
    leftp: VertexId,
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

        let [xmin, xmax, ymin, ymax] = dcel.get_bounds();
        let bbox = BoundingBox::new(xmin, xmax, ymin, ymax);
        let BoundingBox {
            xmin,
            xmax,
            ymin,
            ymax,
        } = bbox;

        let mut dcel = dcel;

        let vertex_count = dcel.vertex_count();
        let hedge_count = dcel.hedge_count();

        // Add the bounding box to the DCEL
        let rec = Mesh::with_stride(
            vec![[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
            vec![0, 1, 2, 3],
            4,
        )
        .unwrap();
        dcel.append(rec);

        dag.add(Node::Trap(Trapezoid {
            top: dcel.get_hedge(HedgeId(hedge_count + 2)).twin,
            bottom: HedgeId(hedge_count),
            leftp: VertexId(vertex_count),
            rightp: VertexId(vertex_count + 1),
        }));

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

    fn build(mut self) -> Self {
        let hedge_count = self.dcel.hedge_count();
        // Last 8 half-edges are for the bounding box, so we always have at least 8 half-edges
        for hid in (0..(hedge_count - 8)).map(HedgeId) {
            if self.dcel.points_right(hid) {
                self.add_edge(hid);
            }
        }
        self
    }

    fn add_edge(&mut self, hedge_id: HedgeId) {
        self.print_stats();

        let hedge = self.dcel.get_hedge(hedge_id);
        let twin = self.dcel.get_hedge(hedge.twin);
        let p = hedge.origin;
        let q = twin.origin;

        let ds = self.follow_segment(hedge_id);
        let old_nid = ds[0];
        let Node::Trap(trap) = &self.dag.get(old_nid).unwrap().data else {
            unreachable!("Has to be a trapezoid")
        };
        let top = trap.top;
        let bottom = trap.bottom;
        let rightp = trap.rightp;
        let leftp = trap.leftp;

        // There are some redundent branches but the code is easier to follow that way
        #[allow(clippy::collapsible_else_if)]
        if p == leftp {
            if q == rightp {
                let s_nid = old_nid;
                // Insert the s Y-node
                let c_nid = self.dag.entry(s_nid).prepend(Node::Y(hedge_id)).unwrap();
                // Update the bottom edge of the C trapezoid
                self.dag.entry(c_nid).and_modify(|node| {
                    let Node::Trap(c) = node else {
                        unreachable!("Has to be a trapezoid")
                    };
                    c.bottom = hedge_id;
                });
                // Append the D trapezoid
                let d = Trapezoid {
                    top: hedge_id,
                    bottom,
                    leftp: p,
                    rightp: q,
                };
                self.dag.entry(s_nid).append(Node::Trap(d)).unwrap();
            } else {
                let q_nid = old_nid;
                // Insert the q X-node
                let s_nid = self.dag.entry(q_nid).prepend(Node::X(q)).unwrap();
                // Append the B trapezoid
                let b = Trapezoid {
                    top,
                    bottom,
                    leftp: q,
                    rightp,
                };
                self.dag.entry(q_nid).append(Node::Trap(b)).unwrap();
                // Insert the s Y-node
                let c_nid = self.dag.entry(s_nid).prepend(Node::Y(hedge_id)).unwrap();
                // Update the bottom edge and rightp of the C trapezoid
                self.dag.entry(c_nid).and_modify(|node| {
                    let Node::Trap(c) = node else {
                        unreachable!("Has to be a trapezoid")
                    };
                    c.bottom = hedge_id;
                    c.rightp = q;
                });
                // Append the D trapezoid
                let d = Trapezoid {
                    top: hedge_id,
                    bottom,
                    leftp,
                    rightp: q,
                };
                self.dag.entry(s_nid).append(Node::Trap(d)).unwrap();
            }
        } else {
            if q == rightp {
                let p_nid = old_nid;
                // Insert the p X-node
                let a_nid = self.dag.entry(p_nid).prepend(Node::X(p)).unwrap();
                // Update the rightp of the A trapezoid
                self.dag.entry(a_nid).and_modify(|node| {
                    let Node::Trap(a) = node else {
                        unreachable!("Has to be a trapezoid")
                    };
                    a.rightp = hedge.origin;
                });
                // Append the s Y-node
                let s_nid = self.dag.entry(p_nid).append(Node::Y(hedge_id)).unwrap();
                // Append the C trapezoid
                let c = Trapezoid {
                    top,
                    bottom: hedge_id,
                    leftp: p,
                    rightp,
                };
                self.dag.entry(s_nid).append(Node::Trap(c)).unwrap();
                // Append the D trapezoid
                let d = Trapezoid {
                    top: hedge_id,
                    bottom,
                    leftp: p,
                    rightp,
                };
                self.dag.entry(s_nid).append(Node::Trap(d)).unwrap();
            } else {
                let p_nid = old_nid;
                // Insert the p X-node
                let a_nid = self.dag.entry(p_nid).prepend(Node::X(p)).unwrap();
                // Update the rightp of the A trapezoid
                self.dag.entry(a_nid).and_modify(|node| {
                    let Node::Trap(a) = node else {
                        unreachable!("Has to be a trapezoid")
                    };
                    a.rightp = hedge.origin;
                });
                // Append the q X-node
                let q_nid = self.dag.entry(p_nid).append(Node::X(rightp)).unwrap();
                // Append the s Y-node
                let s_nid = self.dag.entry(q_nid).append(Node::Y(hedge_id)).unwrap();
                // Append the B trapezoid
                let b = Trapezoid {
                    top,
                    bottom,
                    leftp: q,
                    rightp,
                };
                self.dag.entry(q_nid).append(Node::Trap(b)).unwrap();
                // Append the C trapezoid
                let c = Trapezoid {
                    top,
                    bottom: hedge_id,
                    leftp: p,
                    rightp: q,
                };
                self.dag.entry(s_nid).append(Node::Trap(c)).unwrap();
                // Append the D trapezoid
                let d = Trapezoid {
                    top: hedge_id,
                    bottom,
                    leftp: p,
                    rightp: q,
                };
                self.dag.entry(s_nid).append(Node::Trap(d)).unwrap();
            }
        }

        self.print_stats();
    }

    fn follow_segment(&self, hedge_id: HedgeId) -> Vec<usize> {
        let s = self.dcel.get_hedge(hedge_id);
        let p = s.origin;
        let xy = &self.dcel.get_vertex(p).coords;
        let slope = self.dcel.slope(hedge_id);

        let mut d0 = 0;
        loop {
            let node = &self.dag.get(d0).unwrap();
            match &node.data {
                Node::Trap(..) => break,
                Node::X(vid) => {
                    let vert = &self.dcel.get_vertex(*vid).coords;
                    let left = xy[0] < vert[0];
                    d0 = if left {
                        node.children[0]
                    } else {
                        node.children[1]
                    };
                }
                Node::Y(hid) => {
                    let si = self.dcel.get_hedge(*hid);
                    let pi = si.origin;
                    let qi = self.dcel.get_hedge(si.twin).origin;
                    let above = if p == pi {
                        // s and si share their left endpoint, so we compare the slopes
                        slope > self.dcel.slope(*hid)
                    } else {
                        // s and si share have different left endpoints, so we look at the position
                        // of p with respect to the segment (pi, qi)
                        let xy_pi = self.dcel.get_vertex(pi).coords;
                        let xy_qi = self.dcel.get_vertex(qi).coords;
                        !matches!(Point::from(xy).position(xy_pi, xy_qi), Positioning::Right)
                    };
                    d0 = if above {
                        node.children[0]
                    } else {
                        node.children[1]
                    };
                }
            }
        }

        let res = vec![d0];
        res
    }

    fn find_trapezoid(&self, point: &[f64; 2]) -> (usize, &Trapezoid) {
        let mut node_id = 0;
        loop {
            let node = &self.dag.get(node_id).unwrap();
            match &node.data {
                Node::Trap(trapezoid) => return (node_id, trapezoid),
                Node::X(vid) => {
                    let vert: &[f64; 2] = &self.dcel.get_vertex(*vid).coords;
                    if point[0] < vert[0] {
                        node_id = node.children[0];
                    } else {
                        node_id = node.children[1];
                    }
                }
                Node::Y(hid) => {
                    let hedge = self.dcel.get_hedge(*hid);
                    let twin = self.dcel.get_hedge(hedge.twin);
                    let p1: &[f64; 2] = &self.dcel.get_vertex(hedge.origin).coords;
                    let p2: &[f64; 2] = &self.dcel.get_vertex(twin.origin).coords;
                    match Point::from(point).position(*p1, *p2) {
                        Positioning::Right => node_id = node.children[1],
                        _ => node_id = node.children[0],
                    }
                }
            }
        }
    }
}

impl PointLocator for TrapMap {
    fn locate_one(&self, point: &[f64; 2]) -> Option<usize> {
        if self.dcel.face_count() == 0 {
            return None;
        }

        let bbox_face = self.dcel.face_count() - 1;

        let (_, trap) = self.find_trapezoid(point);
        let face = self.dcel.get_hedge(trap.bottom).face?.get();

        (face < bbox_face).then_some(face)
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
    fn locate_one_in_empty_trapezoidal_map() {
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
    fn add_edges() -> Result<()> {
        let points = vec![[0., 0.], [1., 0.], [0.5, 0.5]];
        let cells = vec![0, 1, 2];
        let mesh = Mesh::with_stride(points, cells, 3)?;
        let dcel = Dcel::from_mesh(mesh);
        let first = HedgeId(0);
        let second = dcel.get_hedge(HedgeId(1)).twin; // Need to get the twin of 1 so that it points to the right
        let third = dcel.get_hedge(HedgeId(2)).twin; // Need to get the twin of 2 so that it points to the right

        let mut trap_map = TrapMap::from_dcel(dcel);

        // Add the first edge
        trap_map.add_edge(first);

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

        // Add the second edge
        trap_map.add_edge(second);

        // Check the number of different nodes
        assert_eq!(trap_map.trap_count(), 6);
        assert_eq!(trap_map.x_node_count(), 3);
        assert_eq!(trap_map.y_node_count(), 2);

        // Add the third edge
        trap_map.add_edge(third);

        // Check the number of different nodes
        assert_eq!(trap_map.trap_count(), 7);
        assert_eq!(trap_map.x_node_count(), 3);
        assert_eq!(trap_map.y_node_count(), 3);

        Ok(())
    }

    #[test]
    fn locate_points_in_single_triangle() -> Result<()> {
        let points = vec![[0., 0.], [1., 0.], [0.5, 0.5]];
        let cells = vec![0, 1, 2];
        let mesh = Mesh::with_stride(points, cells, 3)?;

        let trap_map = TrapMap::from_mesh(mesh).build();

        // Locate a point inside the triangle
        assert_eq!(trap_map.locate_one(&[0.5, 0.1]), Some(0));

        // Locate points outside the triangle
        assert_eq!(trap_map.locate_one(&[0.5, -0.1]), None); // below
        assert_eq!(trap_map.locate_one(&[0.8, 0.8]), None); // above to the right
        assert_eq!(trap_map.locate_one(&[0.2, 0.8]), None); // above to the left
        assert_eq!(trap_map.locate_one(&[1.2, 0.8]), None); // to the right
        assert_eq!(trap_map.locate_one(&[-0.2, 0.8]), None); // to the left

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
