use itertools::Itertools;
use nonmax::NonMaxUsize;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

use crate::mesh::Mesh;
use crate::point_locator::PointLocator;
use crate::trapezoidal_map::dag::Dag;
use crate::winding_number::{Point, Positioning};

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
    pub(crate) p: usize,
    pub(crate) q: usize,
    pub(crate) face: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct Trapezoid {
    pub(crate) leftp: usize,
    pub(crate) rightp: usize,
    pub(crate) bottom: Edge,
    pub(crate) top: Edge,
    pub(crate) lower_left: Option<NonMaxUsize>,
    pub(crate) upper_left: Option<NonMaxUsize>,
    pub(crate) lower_right: Option<NonMaxUsize>,
    pub(crate) upper_right: Option<NonMaxUsize>,
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
    #[allow(unused)]
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

    #[allow(unused)]
    pub(crate) fn empty() -> Self {
        let mut trap_map = Self::new();
        let bbox = BoundingBox::default();
        trap_map.add_bounding_box(bbox);
        trap_map
    }

    pub fn from_mesh(mesh: Mesh) -> Self {
        let n_edges = mesh.facet_count();
        let n_vertices = mesh.vertex_count();
        let mut edges = Vec::with_capacity(n_edges);
        let mut lefties = HashMap::with_capacity(n_edges);
        let mut righties = HashSet::with_capacity(n_edges);
        let mut vertex_faces = vec![None; n_vertices];
        for (face, cell) in mesh.cells().enumerate() {
            for (&p, &q) in cell.iter().circular_tuple_windows() {
                if vertex_faces[p].is_none() {
                    vertex_faces[p] = NonMaxUsize::new(face);
                }
                let edge = Edge { p, q, face };
                let [x1, y1] = mesh.coords(p);
                let [x2, y2] = mesh.coords(q);
                if matches!(
                    x2.total_cmp(&x1).then_with(|| y2.total_cmp(&y1)),
                    Ordering::Greater
                ) {
                    edges.push(Some(edge));
                    // Remove twin if encountered before
                    lefties.remove(&[q, p]);
                    // Remember we have visited this righty
                    righties.insert([p, q]);
                } else if !righties.contains(&[q, p]) {
                    // This is a lefty and we haven't seen its righty twin yet
                    // We don't know yet if the righty twin exists in the mesh, but in case it
                    // doesn't we store this edge along with the corresponding face and the current
                    // index in the list of edges
                    lefties.insert([p, q], [face, edges.len()]);
                    edges.push(None);
                }
            }
        }
        // By the end of the loop we should only have lonely lefties in the hashmap
        // We need to insert their twins in the vec of edges!
        for ([p, q], [face, idx]) in lefties {
            edges[idx] = Some(Edge { p: q, q: p, face });
            if vertex_faces[p].is_none() {
                vertex_faces[p] = NonMaxUsize::new(face);
            }
        }
        let vertex_faces: Vec<usize> = vertex_faces.into_iter().map(|f| f.unwrap().get()).collect();
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

        trap_map
    }

    pub(crate) fn add_bounding_box(&mut self, bbox: BoundingBox) {
        let nv = self.vertices.len();
        self.vertices.push([bbox.xmin, bbox.ymin]);
        self.vertices.push([bbox.xmax, bbox.ymin]);

        self.bbox = bbox;

        self.dag.add(Node::Trap(Trapezoid::new(
            nv,
            nv + 1,
            Edge {
                p: usize::MAX,
                q: usize::MAX,
                face: usize::MAX,
            },
            Edge {
                p: usize::MAX,
                q: usize::MAX,
                face: usize::MAX,
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

    pub(crate) fn add_edge(&mut self, edge: Edge) {
        let Edge { p, q, .. } = edge;

        let mut left_old = None;
        let mut left_below = None;
        let mut left_above = None;

        let trap_ids = self.follow_segment(edge);
        let trap_count = trap_ids.len();
        for (i, &old_trap_idx) in trap_ids.iter().enumerate() {
            let old = &self.dag.get(old_trap_idx).unwrap().data.get_trap().clone();

            // Get the old neighbors
            let old_lower_left = old.lower_left.map(|idx| idx.get());
            let old_upper_left = old.upper_left.map(|idx| idx.get());
            let old_lower_right = old.lower_right.map(|idx| idx.get());
            let old_upper_right = old.upper_right.map(|idx| idx.get());

            let start_trap = i == 0;
            let end_trap = i == trap_count - 1;
            let have_left = start_trap && p != old.leftp;
            let have_right = end_trap && q != old.rightp;

            // Old trapezoid is replaced by up to 4 new trapezoids: left is to the left of the start
            // point p, below/above are below/above the inserted edge, and right is to the right of
            // the end point q.
            //
            // There are 4 different cases here depending on whether the old trapezoid in question
            // is the start and/or end trapezoid of those that intersect the edge inserted.
            // There is some code duplication here but it is much easier to understand this way
            // rather than interleave the 4 different cases with many more if-statements.
            let (left_idx, right_idx, below_idx, above_idx) = if start_trap && end_trap {
                // Edge intersects a single trapezoid.
                let below = Trapezoid::new(p, q, old.bottom, edge);
                let above = Trapezoid::new(p, q, edge, old.top);

                // Add new trapezoids to DAG
                let below_idx = self.dag.add(Node::Trap(below));
                let above_idx = self.dag.add(Node::Trap(above));

                // Connect neighbors
                let left_idx = if have_left {
                    let left = Trapezoid::new(old.leftp, p, old.bottom, old.top);
                    let left_idx = self.dag.add(Node::Trap(left));
                    self.connect_lower_neighbors(old_lower_left, Some(left_idx));
                    self.connect_upper_neighbors(old_upper_left, Some(left_idx));
                    self.connect_lower_neighbors(Some(left_idx), Some(below_idx));
                    self.connect_upper_neighbors(Some(left_idx), Some(above_idx));
                    Some(left_idx)
                } else {
                    self.connect_lower_neighbors(old_lower_left, Some(below_idx));
                    self.connect_upper_neighbors(old_upper_left, Some(above_idx));
                    None
                };

                let right_idx = if have_right {
                    let right = Trapezoid::new(q, old.rightp, old.bottom, old.top);
                    let right_idx = self.dag.add(Node::Trap(right));
                    self.connect_lower_neighbors(Some(right_idx), old_lower_right);
                    self.connect_upper_neighbors(Some(right_idx), old_upper_right);
                    self.connect_lower_neighbors(Some(below_idx), Some(right_idx));
                    self.connect_upper_neighbors(Some(above_idx), Some(right_idx));
                    Some(right_idx)
                } else {
                    self.connect_lower_neighbors(Some(below_idx), old_lower_right);
                    self.connect_upper_neighbors(Some(above_idx), old_upper_right);
                    None
                };

                (left_idx, right_idx, below_idx, above_idx)
            } else if start_trap {
                // Old trapezoid is the first of 2+ trapezoids that the edge intersects.
                let below = Trapezoid::new(p, old.rightp, old.bottom, edge);
                let above = Trapezoid::new(p, old.rightp, edge, old.top);

                // Add new trapezoids to DAG
                let below_idx = self.dag.add(Node::Trap(below));
                let above_idx = self.dag.add(Node::Trap(above));

                // Connect neighbors
                self.connect_lower_neighbors(Some(below_idx), old_lower_right);
                self.connect_upper_neighbors(Some(above_idx), old_upper_right);

                let left_idx = if have_left {
                    let left = Trapezoid::new(old.leftp, p, old.bottom, old.top);
                    let left_idx = self.dag.add(Node::Trap(left));
                    self.connect_lower_neighbors(old_lower_left, Some(left_idx));
                    self.connect_upper_neighbors(old_upper_left, Some(left_idx));
                    self.connect_lower_neighbors(Some(left_idx), Some(below_idx));
                    self.connect_upper_neighbors(Some(left_idx), Some(above_idx));
                    Some(left_idx)
                } else {
                    self.connect_lower_neighbors(old_lower_left, Some(below_idx));
                    self.connect_upper_neighbors(old_upper_left, Some(above_idx));
                    None
                };

                let right_idx = None;

                (left_idx, right_idx, below_idx, above_idx)
            } else if end_trap {
                // Old trapezoid is the last of 2+ trapezoids that the edge intersects.
                let left_below_bottom = self
                    .dag
                    .get(left_below.unwrap())
                    .unwrap()
                    .data
                    .get_trap()
                    .bottom;
                let below_idx = if left_below_bottom == old.bottom {
                    self.dag
                        .entry(left_below.unwrap())
                        .and_modify(|node| node.get_trap_mut().rightp = q);
                    left_below.unwrap()
                } else {
                    self.dag
                        .add(Node::Trap(Trapezoid::new(old.leftp, q, old.bottom, edge)))
                };

                let left_above_top = self
                    .dag
                    .get(left_above.unwrap())
                    .unwrap()
                    .data
                    .get_trap()
                    .top;
                let above_idx = if left_above_top == old.top {
                    self.dag
                        .entry(left_above.unwrap())
                        .and_modify(|node| node.get_trap_mut().rightp = q);
                    left_above.unwrap()
                } else {
                    self.dag
                        .add(Node::Trap(Trapezoid::new(old.leftp, q, edge, old.top)))
                };

                // Connect neighbors
                let right_idx = if have_right {
                    let right = Trapezoid::new(q, old.rightp, old.bottom, old.top);
                    let right_idx = self.dag.add(Node::Trap(right));
                    self.connect_lower_neighbors(Some(right_idx), old_lower_right);
                    self.connect_upper_neighbors(Some(right_idx), old_upper_right);
                    self.connect_lower_neighbors(Some(below_idx), Some(right_idx));
                    self.connect_upper_neighbors(Some(above_idx), Some(right_idx));
                    Some(right_idx)
                } else {
                    self.connect_lower_neighbors(Some(below_idx), old_lower_right);
                    self.connect_upper_neighbors(Some(above_idx), old_upper_right);
                    None
                };

                if below_idx != left_below.unwrap() {
                    self.connect_upper_neighbors(left_below, Some(below_idx));
                    self.connect_lower_neighbors(
                        if old_lower_left == left_old {
                            left_below
                        } else {
                            old_lower_left
                        },
                        Some(below_idx),
                    );
                }

                if above_idx != left_above.unwrap() {
                    self.connect_lower_neighbors(left_above, Some(above_idx));
                    self.connect_upper_neighbors(
                        if old_upper_left == left_old {
                            left_above
                        } else {
                            old_upper_left
                        },
                        Some(above_idx),
                    );
                }

                let left_idx = None;

                (left_idx, right_idx, below_idx, above_idx)
            } else {
                // Middle trapezoid.
                // Old trapezoid is neither the first nor last of the 3+ trapezoids that the edge
                // intersects.
                let left_below_bottom = self
                    .dag
                    .get(left_below.unwrap())
                    .unwrap()
                    .data
                    .get_trap()
                    .bottom;
                let below_idx = if left_below_bottom == old.bottom {
                    self.dag
                        .entry(left_below.unwrap())
                        .and_modify(|node| node.get_trap_mut().rightp = old.rightp);
                    left_below.unwrap()
                } else {
                    self.dag.add(Node::Trap(Trapezoid::new(
                        old.leftp, old.rightp, old.bottom, edge,
                    )))
                };

                let left_above_top = self
                    .dag
                    .get(left_above.unwrap())
                    .unwrap()
                    .data
                    .get_trap()
                    .top;
                let above_idx = if left_above_top == old.top {
                    self.dag
                        .entry(left_above.unwrap())
                        .and_modify(|node| node.get_trap_mut().rightp = old.rightp);
                    left_above.unwrap()
                } else {
                    self.dag.add(Node::Trap(Trapezoid::new(
                        old.leftp, old.rightp, edge, old.top,
                    )))
                };

                // Connect neighbors
                if below_idx != left_below.unwrap() {
                    self.connect_upper_neighbors(left_below, Some(below_idx));
                    self.connect_lower_neighbors(
                        if old_lower_left == left_old {
                            left_below
                        } else {
                            old_lower_left
                        },
                        Some(below_idx),
                    );
                }

                if above_idx != left_above.unwrap() {
                    self.connect_lower_neighbors(left_above, Some(above_idx));
                    self.connect_upper_neighbors(
                        if old_upper_left == left_old {
                            left_above
                        } else {
                            old_upper_left
                        },
                        Some(above_idx),
                    );
                }

                self.connect_lower_neighbors(Some(below_idx), old_lower_right);
                self.connect_upper_neighbors(Some(above_idx), old_upper_right);

                let left_idx = None;
                let right_idx = None;

                (left_idx, right_idx, below_idx, above_idx)
            };

            // Insert new nodes in the DAG and reuse the old trap node
            let si = if let Some(left_idx) = left_idx {
                let pi = old_trap_idx;
                self.dag.entry(pi).and_modify(|node| *node = Node::X(p));
                self.dag.entry(pi).append(left_idx);
                if let Some(right_idx) = right_idx {
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
                } else {
                    self.dag
                        .entry(pi)
                        .append_new(Node::Y(edge))
                        .expect("This should be a valid node")
                }
            } else if let Some(idx) = right_idx {
                let qi = old_trap_idx;
                self.dag.entry(qi).and_modify(|node| *node = Node::X(q));
                let si = self
                    .dag
                    .entry(qi)
                    .append_new(Node::Y(edge))
                    .expect("This should be a valid node");
                self.dag.entry(qi).append(idx);
                si
            } else {
                let si = old_trap_idx;
                self.dag.entry(si).and_modify(|node| *node = Node::Y(edge));
                si
            };
            self.dag.entry(si).append(above_idx);
            self.dag.entry(si).append(below_idx);

            if !end_trap {
                // Prepare for next iteration
                left_old = Some(old_trap_idx);
                left_above = Some(above_idx);
                left_below = Some(below_idx);
            }
        }
    }

    pub(crate) fn connect_lower_neighbors(&mut self, left: Option<usize>, right: Option<usize>) {
        if let Some(idx) = right {
            let left = left.map(|idx| NonMaxUsize::new(idx).unwrap());
            self.dag
                .entry(idx)
                .and_modify(|node| node.get_trap_mut().lower_left = left);
        }
        if let Some(idx) = left {
            let right = right.map(|idx| NonMaxUsize::new(idx).unwrap());
            self.dag
                .entry(idx)
                .and_modify(|node| node.get_trap_mut().lower_right = right);
        }
    }

    pub(crate) fn connect_upper_neighbors(&mut self, left: Option<usize>, right: Option<usize>) {
        if let Some(idx) = right {
            let left = left.map(|idx| NonMaxUsize::new(idx).unwrap());
            self.dag
                .entry(idx)
                .and_modify(|node| node.get_trap_mut().upper_left = left);
        }
        if let Some(idx) = left {
            let right = right.map(|idx| NonMaxUsize::new(idx).unwrap());
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
                    .expect("There should be a lower right trap")
                    .get();
            } else {
                dj = trap
                    .upper_right
                    .expect("There should be an upper right trap")
                    .get();
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

    #[allow(unused)]
    pub(crate) fn check(&self) {
        // Sanity checks
        for node in self.dag.iter() {
            assert!(
                !(node.children.is_empty() && node.parents.is_empty()),
                "There shouldn't be isolated nodes"
            );
            if node.children.is_empty() {
                assert!(
                    matches!(node.data, Node::Trap(..)),
                    "All leaf nodes should be trapezoids"
                );
            }
        }
    }

    #[allow(unused)]
    pub(crate) fn x_node_count(&self) -> usize {
        self.dag
            .iter()
            .filter(|&node| matches!(node.data, Node::X(..)))
            .count()
    }

    #[allow(unused)]
    pub(crate) fn y_node_count(&self) -> usize {
        self.dag
            .iter()
            .filter(|&node| matches!(node.data, Node::Y { .. }))
            .count()
    }

    #[allow(unused)]
    pub(crate) fn trap_count(&self) -> usize {
        self.dag
            .iter()
            .filter(|&node| matches!(node.data, Node::Trap(..)))
            .count()
    }

    pub(crate) fn node_count(&self) -> (usize, usize, usize) {
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

    pub fn depth_stats(&self) -> (f64, usize) {
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
                if trap.bottom.face == usize::MAX || trap.top.face == usize::MAX {
                    usize::MAX
                } else {
                    trap.bottom.face
                }
            }
            Node::X(idx) => self.vertex_faces[*idx],
            Node::Y(Edge { face, .. }) => *face,
        };

        (face < usize::MAX).then_some(face)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::winding_number::Point;
    use anyhow::Result;
    use proptest::prelude::*;

    use super::*;

    prop_compose! {
        fn coords_in_range(xmin: f64, xmax: f64, ymin: f64, ymax: f64)
                          (x in xmin..xmax, y in ymin..ymax) -> [f64; 2] {
           [x, y]
        }
    }

    #[test]
    fn initialize_empty_trapezoidal_map() {
        let trap_map = TrapMap::empty();

        assert_eq!(trap_map.trap_count(), 1);
    }

    #[test]
    fn locate_one_in_empty_trapezoidal_map() {
        let trap_map = TrapMap::empty();

        let point = [0., 0.];

        assert_eq!(trap_map.locate_one(&point), None);
    }

    #[test]
    fn bounding_box() -> Result<()> {
        let points = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
        let cells = vec![0, 1, 2, 3];
        let mesh = Mesh::with_stride(points, cells, 4)?;
        let trap_map = TrapMap::init_with_mesh(mesh);

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
        let first = Edge {
            p: 0,
            q: 1,
            face: 0,
        };
        let second = Edge {
            p: 2,
            q: 1,
            face: 0,
        };
        let third = Edge {
            p: 0,
            q: 2,
            face: 0,
        };

        let mut trap_map = TrapMap::init_with_mesh(mesh);

        // Add the first edge
        trap_map.add_edge(first);

        // Check the number of different nodes
        assert_eq!(trap_map.trap_count(), 4);
        assert_eq!(trap_map.x_node_count(), 2);
        assert_eq!(trap_map.y_node_count(), 1);

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
    fn add_edges_different_order() -> Result<()> {
        let points = vec![[0., 0.], [1., 0.], [0.5, 0.5]];
        let cells = vec![0, 1, 2];
        let mesh = Mesh::with_stride(points, cells, 3)?;
        let first = Edge {
            p: 0,
            q: 2,
            face: 0,
        };
        let second = Edge {
            p: 0,
            q: 1,
            face: 0,
        };
        let third = Edge {
            p: 2,
            q: 1,
            face: 0,
        };

        let mut trap_map = TrapMap::init_with_mesh(mesh);

        // Add the first edge
        trap_map.add_edge(first);

        // Check the number of different nodes
        assert_eq!(trap_map.trap_count(), 4);
        assert_eq!(trap_map.x_node_count(), 2);
        assert_eq!(trap_map.y_node_count(), 1);

        // Add the second edge
        trap_map.add_edge(second);

        // Check the number of different nodes
        assert_eq!(trap_map.trap_count(), 6);
        assert_eq!(trap_map.x_node_count(), 3);
        assert_eq!(trap_map.y_node_count(), 3);

        // Add the third edge
        trap_map.add_edge(third);

        // Check the number of different nodes
        assert_eq!(trap_map.trap_count(), 7);
        assert_eq!(trap_map.x_node_count(), 3);
        assert_eq!(trap_map.y_node_count(), 4);

        Ok(())
    }

    #[test]
    fn add_edges_with_3plus_intersections() -> Result<()> {
        let points = vec![[0., 0.], [1., 0.], [2., 0.], [3., 1.]];
        let cells = vec![0, 1, 2, 3];
        let mesh = Mesh::with_stride(points, cells, 4)?;
        let first = Edge {
            p: 0,
            q: 1,
            face: 0,
        };
        let second = Edge {
            p: 1,
            q: 2,
            face: 0,
        };
        let third = Edge {
            p: 0,
            q: 3,
            face: 0,
        };
        let fourth = Edge {
            p: 2,
            q: 3,
            face: 0,
        };

        let mut trap_map = TrapMap::init_with_mesh(mesh);

        // Add the first edge
        trap_map.add_edge(first);

        // Check the number of different nodes
        assert_eq!(trap_map.trap_count(), 4);
        assert_eq!(trap_map.x_node_count(), 2);
        assert_eq!(trap_map.y_node_count(), 1);

        // Add the second edge
        trap_map.add_edge(second);

        // Check the number of different nodes
        assert_eq!(trap_map.trap_count(), 6);
        assert_eq!(trap_map.x_node_count(), 3);
        assert_eq!(trap_map.y_node_count(), 2);

        // Add the third edge
        trap_map.add_edge(third);

        // Check the number of different nodes
        assert_eq!(trap_map.trap_count(), 8);
        assert_eq!(trap_map.x_node_count(), 4);
        assert_eq!(trap_map.y_node_count(), 5);

        // Add the fourth edge
        trap_map.add_edge(fourth);

        // Check the number of different nodes
        assert_eq!(trap_map.trap_count(), 9);
        assert_eq!(trap_map.x_node_count(), 4);
        assert_eq!(trap_map.y_node_count(), 6);

        Ok(())
    }

    #[test]
    fn locate_points_in_single_triangle() -> Result<()> {
        let points = vec![[0., 0.], [1., 0.], [0.5, 0.5]];
        let cells = vec![0, 1, 2];
        let mesh = Mesh::with_stride(points, cells, 3)?;

        let trap_map = TrapMap::from_mesh(mesh);

        // Locate a point inside the triangle
        assert_eq!(trap_map.locate_one(&[0.5, 0.1]), Some(0));

        // Edge cases
        assert_eq!(trap_map.locate_one(&[0.5, 0.]), Some(0));
        assert_eq!(trap_map.locate_one(&[0.25, 0.25]), Some(0));
        assert_eq!(trap_map.locate_one(&[0.75, 0.25]), Some(0));

        // Corner cases
        assert_eq!(trap_map.locate_one(&[0., 0.]), Some(0));
        assert_eq!(trap_map.locate_one(&[1., 0.]), Some(0));
        assert_eq!(trap_map.locate_one(&[0.5, 0.5]), Some(0));

        // Locate points outside the triangle
        assert_eq!(trap_map.locate_one(&[0.5, -0.1]), None); // below
        assert_eq!(trap_map.locate_one(&[0.8, 0.8]), None); // above to the right
        assert_eq!(trap_map.locate_one(&[0.2, 0.8]), None); // above to the left
        assert_eq!(trap_map.locate_one(&[1.2, 0.8]), None); // to the right
        assert_eq!(trap_map.locate_one(&[-0.2, 0.8]), None); // to the left

        Ok(())
    }

    #[test]
    fn locate_points_in_single_triangle_different_order() -> Result<()> {
        let points = vec![[0., 0.], [1., 0.], [0.5, 0.5]];
        let cells = vec![0, 1, 2];
        let mesh = Mesh::with_stride(points, cells, 3)?;
        let first = Edge {
            p: 0,
            q: 2,
            face: 0,
        };
        let second = Edge {
            p: 0,
            q: 1,
            face: 0,
        };
        let third = Edge {
            p: 2,
            q: 1,
            face: 0,
        };

        let mut trap_map = TrapMap::init_with_mesh(mesh);

        // Add the edges
        trap_map.add_edge(first);
        trap_map.add_edge(second);
        trap_map.add_edge(third);

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
    fn locate_points_with_3plus_intersections() -> Result<()> {
        let points = vec![[0., 0.], [1., 0.], [2., 0.], [3., 1.]];
        let cells = vec![0, 1, 2, 3];
        let mesh = Mesh::with_stride(points, cells, 4)?;
        let first = Edge {
            p: 0,
            q: 1,
            face: 0,
        };
        let second = Edge {
            p: 1,
            q: 2,
            face: 0,
        };
        let third = Edge {
            p: 0,
            q: 3,
            face: 0,
        };
        let fourth = Edge {
            p: 2,
            q: 3,
            face: 0,
        };

        let mut trap_map = TrapMap::init_with_mesh(mesh);

        // Add the edges
        trap_map.add_edge(first);
        trap_map.add_edge(second);
        trap_map.add_edge(third);
        trap_map.add_edge(fourth);

        // Locate a point inside the triangle
        assert_eq!(trap_map.locate_one(&[1., 0.1]), Some(0));

        // Locate points outside the triangle
        assert_eq!(trap_map.locate_one(&[0.5, -0.1]), None); // below
        assert_eq!(trap_map.locate_one(&[0.5, 3.8]), None); // above
        assert_eq!(trap_map.locate_one(&[4.2, 0.8]), None); // to the right
        assert_eq!(trap_map.locate_one(&[-0.2, 0.8]), None); // to the left

        Ok(())
    }

    #[test]
    fn locate_points_in_single_square() -> Result<()> {
        let points = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
        let cells = vec![0, 1, 2, 3];
        let mesh = Mesh::with_stride(points, cells, 4)?;

        let trap_map = TrapMap::from_mesh(mesh);

        // Locate points inside the square
        assert_eq!(trap_map.locate_one(&[0.5, 0.5]), Some(0));
        assert_eq!(trap_map.locate_one(&[0.1, 0.1]), Some(0));
        assert_eq!(trap_map.locate_one(&[0.1, 0.9]), Some(0));
        assert_eq!(trap_map.locate_one(&[0.9, 0.9]), Some(0));
        assert_eq!(trap_map.locate_one(&[0.9, 0.1]), Some(0));

        // Edge cases
        assert_eq!(trap_map.locate_one(&[0.5, 0.]), Some(0));
        assert_eq!(trap_map.locate_one(&[0., 0.5]), Some(0));
        assert_eq!(trap_map.locate_one(&[1., 0.5]), Some(0));
        assert_eq!(trap_map.locate_one(&[0.5, 1.]), Some(0));

        // Corner cases
        assert_eq!(trap_map.locate_one(&[0., 0.]), Some(0));
        assert_eq!(trap_map.locate_one(&[1., 0.]), Some(0));
        assert_eq!(trap_map.locate_one(&[1., 1.]), Some(0));
        assert_eq!(trap_map.locate_one(&[0., 1.]), Some(0));

        // Locate points outside the triangle
        assert_eq!(trap_map.locate_one(&[0.5, -0.1]), None); // south
        assert_eq!(trap_map.locate_one(&[1.5, -0.1]), None); // south-east
        assert_eq!(trap_map.locate_one(&[1.5, 0.8]), None); // east
        assert_eq!(trap_map.locate_one(&[1.5, 1.8]), None); // north-east
        assert_eq!(trap_map.locate_one(&[0.5, 1.8]), None); // north
        assert_eq!(trap_map.locate_one(&[-0.5, 1.8]), None); // north-west
        assert_eq!(trap_map.locate_one(&[-0.5, 0.8]), None); // west
        assert_eq!(trap_map.locate_one(&[-0.5, -0.8]), None); // south-west

        Ok(())
    }

    #[test]
    fn locate_points_in_grid() -> Result<()> {
        let mesh = Mesh::grid(0., 1., 0., 1., 2, 2)?;

        let trap_map = TrapMap::from_mesh(mesh);

        // Locate points in different cells
        assert_eq!(trap_map.locate_one(&[0.25, 0.25]), Some(0));
        assert_eq!(trap_map.locate_one(&[0.75, 0.25]), Some(1));
        assert_eq!(trap_map.locate_one(&[0.25, 0.75]), Some(2));
        assert_eq!(trap_map.locate_one(&[0.75, 0.75]), Some(3));

        Ok(())
    }

    #[test]
    fn locate_vertex() -> Result<()> {
        let mesh = Mesh::with_stride(
            vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
            vec![0, 1, 3, 1, 2, 3],
            3,
        )?;

        let trap_map = TrapMap::from_mesh(mesh);

        // For consistency with matplotlib, points located on vertices should yield the first polygon
        // in which they appear.
        assert_eq!(trap_map.locate_one(&[1., 0.]), Some(0));
        assert_eq!(trap_map.locate_one(&[0., 1.]), Some(0));

        Ok(())
    }

    #[test]
    fn trapezoidal_map_proptest() -> Result<()> {
        let (xmin, xmax) = (0., 10.);
        let (ymin, ymax) = (0., 10.);
        let (nx, ny) = (6, 6); // Use numbers that don't divide the sides evenly on purpose

        // Create trapezoidal map
        let mesh = Mesh::grid(xmin, xmax, ymin, ymax, nx, ny)?;
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
                assert!(point.is_inside(cell));
            }
        });

        Ok(())
    }
}
