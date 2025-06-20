use itertools::Itertools;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::{cmp::Ordering, collections::HashMap};

use crate::mesh::Mesh;
use crate::point_locator::PointLocator;
use crate::trapezoidal_map::dag::Dag;
use crate::winding_number::{Point, Positioning};

/// Trapezoidal map data structure.
///
/// This is essentially a directed acyclic graph (a.k.a. a DAG)
/// where the nodes can be one of three kinds:
/// - an x-node (associated with a vertex of the mesh)
/// - a y-node (associated with an edge of the mesh)
/// - a trapezoid-node (associated with... a trapezoid!)
///
/// The inner nodes of the DAG can only be x- and y-nodes, while
/// the leaf nodes can only be trapezoid-nodes.
///
/// This data structure is one of four known ones that have
/// the optimal *O*(log(*n*)) search time with *O*(*n*) storage,
/// although in the case of the trapezoidal map those are
/// *expected* results. Note that it can be proven that the
/// probability of a bad maximum query time is very small
/// (see [De Berg et al.]).
///
/// The construction of the trapezoidal map is also very interesting,
/// because it is a *randomized incremental* argorithm. What this
/// means is that the edges of the mesh are added one at a time in
/// random order, and at each step of the process we have a search
/// structure that can perform point location queries. When an edge
/// is added, the intersected trapezoids are found using said search
/// structure, and these trapezoids are divided into sub-trapezoids.
/// This process is expected to take *O*(*n* \* log(*n*)) time.
/// Shuffling the edges is really important for the performance
/// of the resulting data structure!
///
/// [De Berg et al.]: https://doi.org/10.1007/978-3-540-77974-2
#[derive(Debug)]
pub struct TrapMap {
    pub(crate) dag: Dag<Node>,
    pub(crate) vertices: Vec<[f64; 2]>,
    pub(crate) vertex_faces: Vec<usize>,
    pub(crate) bbox: BoundingBox,
}

#[derive(Clone, Debug)]
pub(crate) enum Node {
    X(usize),
    Y(Edge),
    Trap(Trapezoid),
}

impl Node {
    pub(crate) fn get_trap(&self) -> &Trapezoid {
        let Self::Trap(trap) = self else {
            panic!("This is not a Trapezoid")
        };
        trap
    }

    pub(crate) fn get_trap_mut(&mut self) -> &mut Trapezoid {
        let Self::Trap(trap) = self else {
            panic!("This is not a Trapezoid")
        };
        trap
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub(crate) struct Edge {
    p: usize,
    q: usize,
    face_above: Option<usize>,
    face_below: Option<usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct Trapezoid {
    pub(crate) leftp: usize,
    pub(crate) rightp: usize,
    pub(crate) bottom: Edge,
    pub(crate) top: Edge,
    pub(crate) lower_left: Option<usize>,
    pub(crate) upper_left: Option<usize>,
    pub(crate) lower_right: Option<usize>,
    pub(crate) upper_right: Option<usize>,
}

impl Trapezoid {
    pub(crate) fn new(leftp: usize, rightp: usize, bottom: Edge, top: Edge) -> Self {
        Self {
            leftp,
            rightp,
            bottom,
            top,
            lower_left: None,
            upper_left: None,
            lower_right: None,
            upper_right: None,
        }
    }
}

#[derive(Debug)]
pub(crate) struct BoundingBox {
    pub(crate) xmin: f64,
    pub(crate) xmax: f64,
    pub(crate) ymin: f64,
    pub(crate) ymax: f64,
}

impl BoundingBox {
    pub(crate) fn from_mesh(mesh: &Mesh) -> Self {
        let mut xmin = f64::MAX;
        let mut xmax = f64::MIN;
        let mut ymin = f64::MAX;
        let mut ymax = f64::MIN;
        for &[x, y] in mesh.points() {
            if x < xmin {
                xmin = x;
            } else if x > xmax {
                xmax = x;
            }
            if y < ymin {
                ymin = y;
            } else if y > ymax {
                ymax = y;
            }
        }
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

#[derive(Clone, Copy)]
enum Orientation {
    Counterclockwise,
    Clockwise,
}

impl TrapMap {
    pub(crate) fn new() -> Self {
        Self {
            dag: Dag::new(),
            vertices: Vec::new(),
            vertex_faces: Vec::new(),
            bbox: BoundingBox {
                xmin: f64::NAN,
                xmax: f64::NAN,
                ymin: f64::NAN,
                ymax: f64::NAN,
            },
        }
    }

    /// Creates an empty trapezoidal map.
    pub fn empty() -> Self {
        let mut trap_map = Self::new();
        let bbox = BoundingBox::default();
        trap_map.add_bounding_box(bbox);
        trap_map
    }

    /// Creates a trapezoidal map from a [`Mesh`].
    pub fn from_mesh(mesh: Mesh) -> Self {
        let n_edges = mesh.facet_count();
        let n_vertices = mesh.vertex_count();
        let mut edges = Vec::with_capacity(n_edges);
        let mut lefties = HashMap::with_capacity(n_edges);
        let mut righties = HashMap::with_capacity(n_edges);
        let mut vertex_faces = vec![None; n_vertices];
        for (face, cell) in mesh.cells().enumerate() {
            // We need to determine the orientation of the current cell in order to know if the
            // face is to the left or to the right of each of its edges.
            // To do so, we find the leftmost point (and pick the bottommost one in case of ties),
            // and then determine the sign of the angle at that point.
            // See: https://en.wikipedia.org/wiki/Curve_orientation#Orientation_of_a_simple_polygon
            let orientation = {
                let b = cell
                    .iter()
                    .enumerate()
                    .map(|(b, &idx)| (b, mesh.coords(idx)))
                    .min_by(|(_, [x1, y1]), (_, [x2, y2])| {
                        x1.total_cmp(x2).then_with(|| y1.total_cmp(y2))
                    })
                    .map(|(b, _)| b)
                    .unwrap();
                let a = if b == 0 { cell.len() - 1 } else { b - 1 };
                let c = if b == cell.len() - 1 { 0 } else { b + 1 };
                let [xa, ya] = mesh.coords(cell[a]);
                let [xb, yb] = mesh.coords(cell[b]);
                let [xc, yc] = mesh.coords(cell[c]);
                debug_assert!(
                    mesh.coords(cell[b]) <= mesh.coords(cell[a])
                        && mesh.coords(cell[b]) <= mesh.coords(cell[c]),
                    "`b` should be the leftmost bottommost vertex"
                );
                let det = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya);
                if det > 0. {
                    Orientation::Counterclockwise
                } else {
                    Orientation::Clockwise
                }
            };
            for (&p, &q) in cell.iter().circular_tuple_windows() {
                vertex_faces[p].get_or_insert(face);
                let [x1, y1] = mesh.coords(p);
                let [x2, y2] = mesh.coords(q);
                if matches!(
                    x2.total_cmp(&x1).then_with(|| y2.total_cmp(&y1)),
                    Ordering::Greater
                ) {
                    // Remove twin if encountered before
                    let twin_face = lefties.remove(&[q, p]).map(|(face, _, _)| face);

                    let (face_above, face_below) = match orientation {
                        Orientation::Counterclockwise => (Some(face), twin_face),
                        Orientation::Clockwise => (twin_face, Some(face)),
                    };
                    edges.push(Some(Edge {
                        p,
                        q,
                        face_above,
                        face_below,
                    }));

                    // Remember we have visited this righty
                    righties.insert([p, q], edges.len() - 1);
                } else if let Some(&edge_id) = righties.get(&[q, p]) {
                    // This is a lefty and we have seen its righty before => we need to set this
                    // face as below/above the righty
                    let edge = edges[edge_id].as_mut().expect("Edge should be Some");
                    let _ = match orientation {
                        Orientation::Counterclockwise => edge.face_below.insert(face),
                        Orientation::Clockwise => edge.face_above.insert(face),
                    };
                } else {
                    // This is a lefty and we haven't seen its righty twin yet
                    // We don't know yet if the righty twin exists in the mesh, but in case it
                    // doesn't we store this edge along with the corresponding face and the current
                    // index in the list of edges
                    lefties.insert([p, q], (face, edges.len(), orientation));
                    edges.push(None);
                }
            }
        }
        // By the end of the loop we should only have lonely lefties in the hashmap
        // We need to insert their twins in the vec of edges!
        for ([p, q], (face, idx, orientation)) in lefties {
            let (face_above, face_below) = match orientation {
                Orientation::Counterclockwise => (None, Some(face)),
                Orientation::Clockwise => (Some(face), None),
            };
            edges[idx] = Some(Edge {
                p: q,
                q: p,
                face_above,
                face_below,
            });
            debug_assert!(
                vertex_faces[p].is_some(),
                "Every point should have been visited"
            );
        }
        let vertex_faces: Vec<usize> = vertex_faces.iter().map(|f| f.unwrap()).collect();
        // There are still some `None`s in the list of edges, namely for lefties that do have a
        // righty twin but who have been seen first
        let mut edges: Vec<_> = edges.into_iter().flatten().collect();

        let mut trap_map = TrapMap::init_with_mesh(mesh);
        trap_map.vertex_faces = vertex_faces;

        // Mix the edges to get good performance (this is a randomized incremental algorithm after all!)
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        edges.shuffle(&mut rng);

        for edge in edges {
            trap_map.add_edge(edge);
        }

        if cfg!(debug_assertions) {
            trap_map.check();
        }

        trap_map
    }

    pub(crate) fn add_bounding_box(&mut self, bbox: BoundingBox) {
        let nv = self.vertices.len();
        self.vertices.push([bbox.xmin, bbox.ymin]);
        self.vertices.push([bbox.xmax, bbox.ymax]);

        self.bbox = bbox;

        self.dag.add(Node::Trap(Trapezoid::new(
            nv,
            nv + 1,
            Edge {
                p: usize::MAX,
                q: usize::MAX,
                face_above: None,
                face_below: None,
            },
            Edge {
                p: usize::MAX,
                q: usize::MAX,
                face_above: None,
                face_below: None,
            },
        )));
    }

    pub(crate) fn init_with_mesh(mesh: Mesh) -> Self {
        let nv = mesh.vertex_count();
        let vertices: Vec<_> = (0..nv).map(|idx| mesh.coords(idx)).collect();

        let bbox = BoundingBox::from_mesh(&mesh);

        let mut trap_map = TrapMap::new();

        trap_map.vertices = vertices;
        trap_map.add_bounding_box(bbox);

        trap_map
    }

    fn add_edge(&mut self, edge: Edge) {
        // Old trapezoids are replaced by up to 4 new trapezoids:
        // - `left` is the new trapezoid to the left of p, if it exists
        // - `right` is the new trapezoid to the right of q, if it exists
        // - `above` is the new trapezoid above the insterted edge
        // - `below` is the new trapezoid below the insterted edge

        let trap_ids = self.follow_segment(edge);
        let trap_count = trap_ids.len();
        assert!(
            trap_count > 0,
            "Edges should always intersect at least one trapezoid."
        );

        if trap_count == 1 {
            self.add_edge_crossing_one_trapezoid(edge, trap_ids[0]);
        } else {
            self.add_edge_crossing_multiple_trapezoids(edge, trap_ids);
        }
    }

    fn add_edge_crossing_one_trapezoid(&mut self, edge: Edge, old_trap_idx: usize) {
        // The old trapezoid is replaced by either 2, 3 or 4 new trapezoids, depending on whether
        // p and q are different from the the old trapezoid's leftp and rightp respectively or
        // not.

        let Edge { p, q, .. } = edge;
        let old = &self.dag.get(old_trap_idx).unwrap().data.get_trap().clone();

        let p_is_new = p != old.leftp;
        let q_is_new = q != old.rightp;

        // Edge intersects a single trapezoid.
        let below = Trapezoid::new(p, q, old.bottom, edge);
        let above = Trapezoid::new(p, q, edge, old.top);

        // Add new trapezoids to DAG
        let below_idx = self.dag.add(Node::Trap(below));
        let above_idx = self.dag.add(Node::Trap(above));

        // Connect neighbors
        let left_idx = if p_is_new {
            let left = Trapezoid::new(old.leftp, p, old.bottom, old.top);
            let left_idx = self.dag.add(Node::Trap(left));
            self.connect_lower_neighbors(old.lower_left, Some(left_idx));
            self.connect_upper_neighbors(old.upper_left, Some(left_idx));
            self.connect_lower_neighbors(Some(left_idx), Some(below_idx));
            self.connect_upper_neighbors(Some(left_idx), Some(above_idx));
            Some(left_idx)
        } else {
            self.connect_lower_neighbors(old.lower_left, Some(below_idx));
            self.connect_upper_neighbors(old.upper_left, Some(above_idx));
            None
        };
        let right_idx = if q_is_new {
            let right = Trapezoid::new(q, old.rightp, old.bottom, old.top);
            let right_idx = self.dag.add(Node::Trap(right));
            self.connect_lower_neighbors(Some(right_idx), old.lower_right);
            self.connect_upper_neighbors(Some(right_idx), old.upper_right);
            self.connect_lower_neighbors(Some(below_idx), Some(right_idx));
            self.connect_upper_neighbors(Some(above_idx), Some(right_idx));
            Some(right_idx)
        } else {
            self.connect_lower_neighbors(Some(below_idx), old.lower_right);
            self.connect_upper_neighbors(Some(above_idx), old.upper_right);
            None
        };

        self.replace_old_trap_node(
            old_trap_idx,
            left_idx,
            right_idx,
            above_idx,
            below_idx,
            edge,
        );
    }

    fn add_edge_crossing_multiple_trapezoids(&mut self, edge: Edge, trap_ids: Vec<usize>) {
        let Edge { p, q, .. } = edge;

        // First trapezoid.
        //
        // The old trapezoid is replaced by either 2 or 3 new trapezoids, depending on whether
        // p is different from the the old trapezoid's leftp or not.

        let old_trap_idx = trap_ids[0];
        let old = &self.dag.get(old_trap_idx).unwrap().data.get_trap().clone();
        let p_is_new = p != old.leftp;

        let below = Trapezoid::new(p, old.rightp, old.bottom, edge);
        let above = Trapezoid::new(p, old.rightp, edge, old.top);

        // Add new trapezoids to DAG
        let below_idx = self.dag.add(Node::Trap(below));
        let above_idx = self.dag.add(Node::Trap(above));

        // Connect neighbors
        self.connect_lower_neighbors(Some(below_idx), old.lower_right);
        self.connect_upper_neighbors(Some(above_idx), old.upper_right);
        let left_idx = if p_is_new {
            let left = Trapezoid::new(old.leftp, p, old.bottom, old.top);
            let left_idx = self.dag.add(Node::Trap(left));
            self.connect_lower_neighbors(old.lower_left, Some(left_idx));
            self.connect_upper_neighbors(old.upper_left, Some(left_idx));
            self.connect_lower_neighbors(Some(left_idx), Some(below_idx));
            self.connect_upper_neighbors(Some(left_idx), Some(above_idx));
            Some(left_idx)
        } else {
            self.connect_lower_neighbors(old.lower_left, Some(below_idx));
            self.connect_upper_neighbors(old.upper_left, Some(above_idx));
            None
        };

        self.replace_old_trap_node(old_trap_idx, left_idx, None, above_idx, below_idx, edge);

        // Keep track of old, above and below to make connections with the following trapezoids
        let mut left_old = old_trap_idx;
        let mut left_above = above_idx;
        let mut left_below = below_idx;

        for &old_trap_idx in trap_ids[1..trap_ids.len() - 1].iter() {
            // Middle trapezoids.
            // Old trapezoid is neither the first nor last of the 3+ trapezoids that the edge
            // intersects.
            //
            // The old trapezoid is always replaced by exactly 2 new trapezoids.

            let old = &self.dag.get(old_trap_idx).unwrap().data.get_trap().clone();

            let left_below_bottom = self.dag.get(left_below).unwrap().data.get_trap().bottom;
            let below_idx = if left_below_bottom == old.bottom {
                self.dag
                    .entry(left_below)
                    .and_modify(|node| node.get_trap_mut().rightp = old.rightp);
                left_below
            } else {
                self.dag.add(Node::Trap(Trapezoid::new(
                    old.leftp, old.rightp, old.bottom, edge,
                )))
            };

            let left_above_top = self.dag.get(left_above).unwrap().data.get_trap().top;
            let above_idx = if left_above_top == old.top {
                self.dag
                    .entry(left_above)
                    .and_modify(|node| node.get_trap_mut().rightp = old.rightp);
                left_above
            } else {
                self.dag.add(Node::Trap(Trapezoid::new(
                    old.leftp, old.rightp, edge, old.top,
                )))
            };

            // Connect neighbors
            if below_idx != left_below {
                self.connect_upper_neighbors(Some(left_below), Some(below_idx));
                self.connect_lower_neighbors(
                    if old.lower_left == Some(left_old) {
                        Some(left_below)
                    } else {
                        old.lower_left
                    },
                    Some(below_idx),
                );
            }
            if above_idx != left_above {
                self.connect_lower_neighbors(Some(left_above), Some(above_idx));
                self.connect_upper_neighbors(
                    if old.upper_left == Some(left_old) {
                        Some(left_above)
                    } else {
                        old.upper_left
                    },
                    Some(above_idx),
                );
            }
            self.connect_lower_neighbors(Some(below_idx), old.lower_right);
            self.connect_upper_neighbors(Some(above_idx), old.upper_right);

            self.replace_old_trap_node(old_trap_idx, None, None, above_idx, below_idx, edge);

            // Prepare next iteration
            left_old = old_trap_idx;
            left_above = above_idx;
            left_below = below_idx;
        }

        // Last trapezoid.
        //
        // The old trapezoid is replaced by either 2 or 3 new trapezoids, depending on whether
        // q is different from the the old trapezoid's rightp or not.

        let old_trap_idx = trap_ids[trap_ids.len() - 1];
        let old = &self.dag.get(old_trap_idx).unwrap().data.get_trap().clone();
        let q_is_new = q != old.rightp;

        let left_below_bottom = self.dag.get(left_below).unwrap().data.get_trap().bottom;
        let below_idx = if left_below_bottom == old.bottom {
            self.dag
                .entry(left_below)
                .and_modify(|node| node.get_trap_mut().rightp = q);
            left_below
        } else {
            self.dag
                .add(Node::Trap(Trapezoid::new(old.leftp, q, old.bottom, edge)))
        };

        let left_above_top = self.dag.get(left_above).unwrap().data.get_trap().top;
        let above_idx = if left_above_top == old.top {
            self.dag
                .entry(left_above)
                .and_modify(|node| node.get_trap_mut().rightp = q);
            left_above
        } else {
            self.dag
                .add(Node::Trap(Trapezoid::new(old.leftp, q, edge, old.top)))
        };

        // Connect neighbors
        let right_idx = if q_is_new {
            let right = Trapezoid::new(q, old.rightp, old.bottom, old.top);
            let right_idx = self.dag.add(Node::Trap(right));
            self.connect_lower_neighbors(Some(right_idx), old.lower_right);
            self.connect_upper_neighbors(Some(right_idx), old.upper_right);
            self.connect_lower_neighbors(Some(below_idx), Some(right_idx));
            self.connect_upper_neighbors(Some(above_idx), Some(right_idx));
            Some(right_idx)
        } else {
            self.connect_lower_neighbors(Some(below_idx), old.lower_right);
            self.connect_upper_neighbors(Some(above_idx), old.upper_right);
            None
        };
        if below_idx != left_below {
            self.connect_upper_neighbors(Some(left_below), Some(below_idx));
            self.connect_lower_neighbors(
                if old.lower_left == Some(left_old) {
                    Some(left_below)
                } else {
                    old.lower_left
                },
                Some(below_idx),
            );
        }
        if above_idx != left_above {
            self.connect_lower_neighbors(Some(left_above), Some(above_idx));
            self.connect_upper_neighbors(
                if old.upper_left == Some(left_old) {
                    Some(left_above)
                } else {
                    old.upper_left
                },
                Some(above_idx),
            );
        }

        self.replace_old_trap_node(old_trap_idx, None, right_idx, above_idx, below_idx, edge);
    }

    fn replace_old_trap_node(
        &mut self,
        old_trap_idx: usize,
        left_idx: Option<usize>,
        right_idx: Option<usize>,
        above_idx: usize,
        below_idx: usize,
        edge: Edge,
    ) {
        let Edge { p, q, .. } = edge;

        // We need to create a y-node and append the above and below trapezoid-node ids, but
        // before we may need to create 1 or 2 x-nodes.
        let si = match (left_idx, right_idx) {
            (None, None) => {
                // No x-node to add => just create the y-node
                let si = old_trap_idx;
                self.dag.entry(si).and_modify(|node| *node = Node::Y(edge));
                si
            }
            (None, Some(right_idx)) => {
                // One x-node to add with the q endpoint, then create the y-node
                let qi = old_trap_idx;
                self.dag.entry(qi).and_modify(|node| *node = Node::X(q));
                let si = self
                    .dag
                    .entry(qi)
                    .append_new(Node::Y(edge))
                    .expect("This should be a valid node");
                self.dag.entry(qi).append(right_idx);
                si
            }
            (Some(left_idx), None) => {
                // One x-node to add with the p endpoint, then create the y-node
                let pi = old_trap_idx;
                self.dag.entry(pi).and_modify(|node| *node = Node::X(p));
                self.dag.entry(pi).append(left_idx);
                self.dag
                    .entry(pi)
                    .append_new(Node::Y(edge))
                    .expect("This should be a valid node")
            }
            (Some(left_idx), Some(right_idx)) => {
                // Two x-nodes to add (one for each endpoint), then create the y-node
                let pi = old_trap_idx;
                self.dag.entry(pi).and_modify(|node| *node = Node::X(p));
                self.dag.entry(pi).append(left_idx);
                let qi = self
                    .dag
                    .entry(pi)
                    .append_new(Node::X(q))
                    .expect("This should be a valid node");
                let si = self
                    .dag
                    .entry(qi)
                    .append_new(Node::Y(edge))
                    .expect("This should be a valid node");
                self.dag.entry(qi).append(right_idx);
                si
            }
        };

        self.dag.entry(si).append(above_idx);
        self.dag.entry(si).append(below_idx);
    }

    pub(crate) fn connect_lower_neighbors(&mut self, left: Option<usize>, right: Option<usize>) {
        if let Some(idx) = right {
            self.dag
                .entry(idx)
                .and_modify(|node| node.get_trap_mut().lower_left = left);
        }
        if let Some(idx) = left {
            self.dag
                .entry(idx)
                .and_modify(|node| node.get_trap_mut().lower_right = right);
        }
    }

    pub(crate) fn connect_upper_neighbors(&mut self, left: Option<usize>, right: Option<usize>) {
        if let Some(idx) = right {
            self.dag
                .entry(idx)
                .and_modify(|node| node.get_trap_mut().upper_left = left);
        }
        if let Some(idx) = left {
            self.dag
                .entry(idx)
                .and_modify(|node| node.get_trap_mut().upper_right = right);
        }
    }

    pub(crate) fn follow_segment(&self, edge: Edge) -> Vec<usize> {
        let p = self.vertices[edge.p];
        let q = self.vertices[edge.q];

        // Find the first trapezoid intersected by s
        let d0 = self.find_first_trapezoid(edge);

        // Loop to find all the other ones
        let mut res = vec![d0];
        let mut dj = d0;
        let mut trap = self.dag.get(dj).unwrap().data.get_trap();
        let mut rightp = self.vertices[trap.rightp];
        while q > rightp {
            let rightp_above_s = !matches!(Point::from(rightp).position(p, q), Positioning::Right);
            if rightp_above_s {
                dj = trap
                    .lower_right
                    .expect("There should be a lower right trap");
            } else {
                dj = trap
                    .upper_right
                    .expect("There should be an upper right trap");
            }
            trap = self.dag.get(dj).unwrap().data.get_trap();
            rightp = self.vertices[trap.rightp];
            res.push(dj);
        }

        res
    }

    pub(crate) fn find_first_trapezoid(&self, edge: Edge) -> usize {
        let p = edge.p;
        let xy = self.vertices[p];
        let slope = self.slope(edge);

        let mut d0 = 0;
        loop {
            let node = &self.dag.get(d0).unwrap();
            match &node.data {
                Node::Trap(..) => break,
                Node::X(idx) => {
                    let vert = self.vertices[*idx];
                    let left = !(p == *idx || xy > vert);
                    d0 = if left {
                        node.children[0]
                    } else {
                        node.children[1]
                    };
                }
                Node::Y(edge_i) => {
                    let pi = edge_i.p;
                    let qi = edge_i.q;
                    let above = if p == pi {
                        // s and si share their left endpoint, so we compare the slopes
                        slope > self.slope(*edge_i)
                    } else {
                        // s and si share have different left endpoints, so we look at the position
                        // of p with respect to the segment (pi, qi)
                        let xy_pi = self.vertices[pi];
                        let xy_qi = self.vertices[qi];
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
        d0
    }

    pub(crate) fn slope(&self, Edge { p, q, .. }: Edge) -> f64 {
        let [xp, yp] = self.vertices[p];
        let [xq, yq] = self.vertices[q];
        if xp == xq {
            f64::INFINITY
        } else {
            (yq - yp) / (xq - xp)
        }
    }

    pub(crate) fn find_node(&self, point: &[f64; 2]) -> &Node {
        let mut node_id = 0;
        loop {
            let node = &self.dag.get(node_id).unwrap();
            match &node.data {
                Node::Trap(..) => break,
                Node::X(idx) => {
                    let [x, y] = self.vertices[*idx];
                    match point[0].total_cmp(&x).then_with(|| point[1].total_cmp(&y)) {
                        Ordering::Greater => node_id = node.children[1],
                        Ordering::Less => node_id = node.children[0],
                        Ordering::Equal => break,
                    };
                }
                Node::Y(Edge { p, q, .. }) => {
                    let p1 = self.vertices[*p];
                    let p2 = self.vertices[*q];
                    match Point::from(point).position(p1, p2) {
                        Positioning::Right => node_id = node.children[1],
                        Positioning::Left => node_id = node.children[0],
                        Positioning::On => break,
                    }
                }
            }
        }
        &self.dag.get(node_id).unwrap().data
    }

    /// Checks some invariants of the DAG.
    ///
    /// This is meant for debugging purposes.
    ///
    /// # Panics
    ///
    /// Panics if there are isolated nodes in the DAG, or
    /// if there are leaf nodes that are x- or y-nodes.
    pub fn check(&self) {
        // Sanity checks
        for node in self.dag.iter() {
            assert_eq!(
                matches!(node.data, Node::Trap(..)),
                node.children.is_empty(),
                "All leaf nodes should be trapezoids and vice-versa"
            );
        }
    }

    /// Returns the number of x-nodes in the DAG.
    pub fn x_node_count(&self) -> usize {
        self.dag
            .iter()
            .filter(|&node| matches!(node.data, Node::X(..)))
            .count()
    }

    /// Returns the number of y-nodes in the DAG.
    pub fn y_node_count(&self) -> usize {
        self.dag
            .iter()
            .filter(|&node| matches!(node.data, Node::Y { .. }))
            .count()
    }

    /// Returns the number of trapezoid-nodes in the DAG.
    pub fn trap_count(&self) -> usize {
        self.dag
            .iter()
            .filter(|&node| matches!(node.data, Node::Trap(..)))
            .count()
    }

    /// Prints some statistics of the DAG.
    ///
    /// Useful for debugging purposes.
    ///
    /// These statistics are:
    /// - Number of x-, y- and trapezoid-nodes
    /// - Average and max depth
    pub fn print_stats(&self) {
        let (x_node_count, y_node_count, trap_count) = self.node_count();
        println!(
            "Trapezoidal map counts:\n\t{} X node(s)\n\t{} Y node(s)\n\t{} trapezoid(s)",
            x_node_count, y_node_count, trap_count,
        );
        println!();
        let (avg, max) = self.depth_stats();
        println!("Depth:\n\tmax {}\n\taverage {}", max, avg);
    }

    /// Returns the number of nodes in the DAG.
    pub fn node_count(&self) -> (usize, usize, usize) {
        self.dag.iter().fold(
            (0, 0, 0),
            |(mut x_count, mut y_count, mut trap_count), node| {
                match node.data {
                    Node::X(..) => x_count += 1,
                    Node::Y { .. } => y_count += 1,
                    Node::Trap(..) => trap_count += 1,
                };
                (x_count, y_count, trap_count)
            },
        )
    }

    fn depth_stats(&self) -> (f64, usize) {
        let mut trap_count = 0;
        let mut avg = 0;
        let mut max = 0;
        for (idx, node) in self.dag.iter().enumerate() {
            if matches!(node.data, Node::Trap(..)) {
                trap_count += 1;
                let depth = self
                    .dag
                    .depth(idx)
                    .expect("Should be in the DAG and have a depth");
                avg += depth;
                if depth > max {
                    max = depth;
                }
            }
        }
        let avg = avg as f64 / trap_count as f64;
        (avg, max)
    }
}

impl PointLocator for TrapMap {
    fn locate_one(&self, point: &[f64; 2]) -> Option<usize> {
        if self.vertices.is_empty() {
            return None;
        }

        let node = self.find_node(point);
        let face = match node {
            Node::Trap(trap) => {
                debug_assert_eq!(trap.bottom.face_above, trap.top.face_below);
                trap.bottom.face_above.unwrap_or(usize::MAX)
            }
            Node::X(idx) => self.vertex_faces[*idx],
            Node::Y(Edge {
                face_above,
                face_below,
                ..
            }) => face_above.or(*face_below).unwrap_or(usize::MAX),
        };

        (face < usize::MAX).then_some(face)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::winding_number::Point;
    use proptest::prelude::*;
    use rstest::*;

    use super::*;

    #[test]
    fn empty_trapezoidal_map() {
        let trap_map = TrapMap::empty();

        assert_eq!(trap_map.trap_count(), 1);
        assert_eq!(trap_map.locate_one(&[0., 0.]), None);
    }

    #[test]
    fn bounding_box() {
        let points = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
        let cells = vec![0, 1, 2, 3];
        let mesh = Mesh::with_stride(points, cells, 4).unwrap();
        let trap_map = TrapMap::init_with_mesh(mesh);

        let bbox = trap_map.bbox;

        assert!(bbox.xmin < 0.);
        assert!(bbox.xmax > 1.);
        assert!(bbox.ymin < 0.);
        assert!(bbox.ymax > 1.);
    }

    //    2
    //    +
    //   / \
    //  +---+
    //  0   1
    #[fixture]
    #[once]
    fn single_triangle() -> TrapMap {
        let points = vec![[0., 0.], [1., 0.], [0.5, 0.5]];
        let cells = vec![0, 1, 2];
        let mesh = Mesh::with_stride(points, cells, 3).unwrap();
        TrapMap::from_mesh(mesh)
    }

    #[rstest]
    #[case::properly_inside([0.5, 0.1], Some(0))]
    #[case::first_edge([0.5, 0.], Some(0))]
    #[case::second_edge([0.25, 0.25], Some(0))]
    #[case::third_edge([0.75, 0.25], Some(0))]
    #[case::first_vertex([0., 0.], Some(0))]
    #[case::second_vertex([1., 0.], Some(0))]
    #[case::third_vertex([0.5, 0.5], Some(0))]
    #[case::outside_below([0.5, -0.1], None)]
    #[case::outside_above_to_the_right([0.8, 0.8], None)]
    #[case::outside_above_to_the_left([0.2, 0.8], None)]
    #[case::outside_to_the_right([1.2, 0.8], None)]
    #[case::outside_to_the_left([-0.2, 0.8], None)]
    fn locate_points_in_single_triangle(
        single_triangle: &TrapMap,
        #[case] point: [f64; 2],
        #[case] cell_id: Option<usize>,
    ) {
        assert_eq!(single_triangle.locate_one(&point), cell_id);
    }

    //  3  2
    //  +--+
    //  |  |
    //  +--+
    //  0  1
    #[fixture]
    #[once]
    fn single_square() -> TrapMap {
        let points = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
        let cells = vec![0, 1, 2, 3];
        let mesh = Mesh::with_stride(points, cells, 4).unwrap();
        TrapMap::from_mesh(mesh)
    }

    #[rstest]
    #[case::properly_inside([0.5, 0.5], Some(0))]
    #[case::properly_inside([0.1, 0.1], Some(0))]
    #[case::properly_inside([0.1, 0.9], Some(0))]
    #[case::properly_inside([0.9, 0.9], Some(0))]
    #[case::properly_inside([0.9, 0.1], Some(0))]
    #[case::first_edge([0.5, 0.], Some(0))]
    #[case::second_edge([0., 0.5], Some(0))]
    #[case::third_edge([1., 0.5], Some(0))]
    #[case::fourth_edge([0.5, 1.], Some(0))]
    #[case::first_vertex([0., 0.], Some(0))]
    #[case::second_vertex([1., 0.], Some(0))]
    #[case::third_vertex([1., 1.], Some(0))]
    #[case::fourth_vertex([0., 1.], Some(0))]
    #[case::outside_south([0.5, -0.1], None)]
    #[case::outside_south_east([1.5, -0.1], None)]
    #[case::outside_east([1.5, 0.8], None)]
    #[case::outside_north_east([1.5, 1.8], None)]
    #[case::outside_north([0.5, 1.8], None)]
    #[case::outside_north_west([-0.5, 1.8], None)]
    #[case::outside_west([-0.5, 0.8], None)]
    #[case::outside_south_west([-0.5, -0.8], None)]
    fn locate_points_in_single_square(
        single_square: &TrapMap,
        #[case] point: [f64; 2],
        #[case] cell_id: Option<usize>,
    ) {
        assert_eq!(single_square.locate_one(&point), cell_id);
    }

    #[fixture]
    #[once]
    fn grid() -> TrapMap {
        TrapMap::from_mesh(Mesh::grid(0., 1., 0., 1., 2, 2).unwrap())
    }

    #[rstest]
    #[case::bottom_left([0.25, 0.25], Some(0))]
    #[case::bottom_right([0.75, 0.25], Some(1))]
    #[case::top_left([0.25, 0.75], Some(2))]
    #[case::top_right([0.75, 0.75], Some(3))]
    fn locate_points_in_grid(
        grid: &TrapMap,
        #[case] point: [f64; 2],
        #[case] cell_id: Option<usize>,
    ) {
        assert_eq!(grid.locate_one(&point), cell_id);
    }

    //  2  3
    //  +--+
    //  |\1|
    //  |0\|
    //  +--+
    //  0  1
    #[fixture]
    #[once]
    fn two_triangles() -> TrapMap {
        let mesh = Mesh::with_stride(
            vec![[0., 0.], [1., 0.], [0., 1.], [1., 1.]],
            vec![0, 1, 2, 1, 3, 2],
            3,
        )
        .unwrap();
        TrapMap::from_mesh(mesh)
    }

    #[rstest]
    #[case::bottom_right_vertex([1., 0.])]
    #[case::top_left_vertex([0., 1.])]
    fn locate_shared_vertex(two_triangles: &TrapMap, #[case] point: [f64; 2]) {
        // For consistency with matplotlib, points located on vertices should yield the first polygon
        // in which they appear.
        assert_eq!(two_triangles.locate_one(&point), Some(0));
    }

    //  2  3
    //  +--+
    //  |\0|
    //  |1\|
    //  +--+
    //  0  1
    // Since the top triangle is the first one to be added, edge (2, 1) is visited before (1, 2)
    #[fixture]
    #[once]
    fn two_triangles_top_right_first() -> TrapMap {
        let mesh = Mesh::with_stride(
            vec![[0., 0.], [1., 0.], [0., 1.], [1., 1.]],
            vec![1, 3, 2, 0, 1, 2],
            3,
        )
        .unwrap();
        TrapMap::from_mesh(mesh)
    }

    #[rstest]
    #[case::inside([0.1, 0.1], Some(1))]
    #[case::inside([0.9, 0.9], Some(0))]
    fn two_triangles_with_righty_of_twin_visited_first(
        two_triangles_top_right_first: &TrapMap,
        #[case] point: [f64; 2],
        #[case] cell_id: Option<usize>,
    ) {
        assert_eq!(two_triangles_top_right_first.locate_one(&point), cell_id);
    }

    //  5
    //  +          +
    //  |\         |\
    //  | \        |2\
    // 3+--+4      +--+
    //  |\ |\      |\ |\
    //  | \| \     |0\|1\
    //  +--+--+    +--+--+
    //  0  1  2
    #[fixture]
    #[once]
    fn multiply_connected_triangulation() -> TrapMap {
        let mesh = Mesh::with_stride(
            vec![[0., 0.], [1., 0.], [2., 0.], [0., 1.], [1., 1.], [0., 2.]],
            vec![0, 1, 3, 1, 2, 4, 3, 5, 4],
            3,
        )
        .unwrap();
        TrapMap::from_mesh(mesh)
    }

    #[rstest]
    #[case::inside([1. / 3., 1. / 3.], Some(0))]
    #[case::inside([4. / 3., 1. / 3.], Some(1))]
    #[case::inside([1. / 3., 4. / 3.], Some(2))]
    #[case::in_hole([2. / 3., 2. / 3.], None)]
    fn locate_in_multiply_connected_triangulation(
        multiply_connected_triangulation: &TrapMap,
        #[case] point: [f64; 2],
        #[case] cell_id: Option<usize>,
    ) {
        assert_eq!(multiply_connected_triangulation.locate_one(&point), cell_id);
    }

    prop_compose! {
        fn coords_in_range(xmin: f64, xmax: f64, ymin: f64, ymax: f64)
                          (x in xmin..xmax, y in ymin..ymax) -> [f64; 2] {
           [x, y]
        }
    }

    #[test]
    fn grid_proptest() {
        let (xmin, xmax) = (0., 10.);
        let (ymin, ymax) = (0., 10.);
        let (nx, ny) = (6, 6); // Use numbers that don't divide the sides evenly on purpose

        // Create trapezoidal map
        let mesh = Mesh::grid(xmin, xmax, ymin, ymax, nx, ny).unwrap();
        let locator = TrapMap::from_mesh(mesh.clone()); // Clone the mesh to use it for verification with the winding number

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
                prop_assert!(point.is_inside(cell));
            }
        });
    }

    fn barycentric_coordinates() -> impl Strategy<Value = (f64, f64, f64)> {
        let eps = 1e-12;
        (eps..=(1.0 - eps))
            .prop_flat_map(move |wa| (Just(wa), eps..=(1.0 - wa - eps)))
            .prop_flat_map(|(wa, wb)| (Just(wa), Just(wb), Just(1. - wa - wb)))
    }

    proptest! {
        #[test]
        fn triangulation_proptest(points in proptest::collection::vec(coords_in_range(0., 1., 0., 1.), 16)) {
            // We start from points generated in the unit square, and we translate them to end up
            // with points placed on a regular grid, each one being randomly placed in its own cell
            let n = points.len();
            let sqrt_n = (n as f64).sqrt() as usize;
            if sqrt_n * sqrt_n != n {
                panic!("Size of points should be a perfect square");
            }
            let points: Vec<_> = points
                .into_iter()
                .enumerate()
                .map(|(i, [x, y])| {
                    let j = i / sqrt_n;
                    let i = i % sqrt_n;
                    [x + i as f64, y + j as f64]
                })
                .collect();
            let pts: Vec<_> = points
                .iter()
                .map(|&[x, y]| delaunator::Point { x, y })
                .collect();

            let triangulation = delaunator::triangulate(&pts);
            let triangles: Vec<_> = triangulation
                .triangles
                .chunks(3)
                .map(|a| [a[0], a[1], a[2]])
                .collect();

            let mesh = Mesh::with_stride(points.clone(), triangulation.triangles, 3).unwrap();
            let locator = TrapMap::from_mesh(mesh);

            let expected = (0..triangles.len()).map(Some).collect::<Vec<_>>();
            proptest!(|(bcoords in proptest::collection::vec(barycentric_coordinates(), triangles.len()))| {
                // We construct one query point per triangle, using barycentric coordinates.
                // That way we know that the point is supposed to be in the triangle
                let query_points:Vec<_> = triangles.iter().zip(bcoords).map(|(&[a, b, c], (wa, wb, wc))| {
                    let [xa, ya] = points[a];
                    let [xb, yb] = points[b];
                    let [xc, yc] = points[c];
                    let x = wa * xa + wb * xb + wc * xc;
                    let y = wa * ya + wb * yb + wc * yc;
                    [x, y]
                }).collect();

                let locations = locator.locate_many(&query_points);

                prop_assert_eq!(&locations, &expected);
            });
        }
    }
}
