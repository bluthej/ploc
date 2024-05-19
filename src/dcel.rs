#![allow(dead_code)]

use std::{
    collections::HashMap,
    fmt::Display,
    ops::{Add, AddAssign, Sub, SubAssign},
};

use anyhow::{anyhow, Result};
use itertools::Itertools;

use crate::mesh::Mesh;

#[derive(Debug)]
pub(crate) struct Dcel {
    vertices: Vec<Vertex>,
    faces: Vec<Face>,
    hedges: Vec<Hedge>,
    contours: Vec<HedgeId>,
}

#[derive(Debug)]
pub(crate) struct Vertex {
    pub(crate) coords: [f64; 2],
    pub(crate) hedge: HedgeId,
}

pub(crate) trait IsRightOf<Rhs = Self> {
    fn is_right_of(&self, other: &Rhs) -> bool;
}

impl IsRightOf<Vertex> for [f64; 2] {
    fn is_right_of(&self, other: &Vertex) -> bool {
        self > &other.coords
    }
}

impl IsRightOf for Vertex {
    fn is_right_of(&self, other: &Self) -> bool {
        self.coords.is_right_of(other)
    }
}

impl Display for Vertex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [x, y] = self.coords;
        write!(f, "({}, {})", x, y)
    }
}

/// A struct for a partially initialized vertex.
///
/// During the construction of the DCEL, we don't know at first which half-edges each vertex belongs
/// to, so we initialize them without a half-edge and update them afterwards.
/// However, since in the end every vertex has an associated half-edge, we don't want to leave an
/// `Option<HedgeId>`, which is why this temporary struct is useful.
struct PartialVertex {
    coords: [f64; 2],
    hedge: Option<HedgeId>,
}

impl PartialVertex {
    fn try_into_vertex(self) -> Result<Vertex> {
        Ok(Vertex {
            coords: self.coords,
            hedge: self.hedge.ok_or(anyhow!("Missing `hedge` field"))?,
        })
    }
}

#[derive(Debug)]
pub(crate) struct Face {
    start: HedgeId,
}

#[derive(Debug, Default)]
pub(crate) struct Hedge {
    pub(crate) origin: VertexId,
    pub(crate) twin: HedgeId,
    pub(crate) face: Option<FaceId>,
    next: HedgeId,
    prev: HedgeId,
}

impl Hedge {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn points_to_the_right(&self, dcel: &Dcel) -> bool {
        let twin = dcel.get_hedge(self.twin);

        let pid = self.origin;
        let p = dcel.get_vertex(pid);
        let qid = twin.origin;
        let q = dcel.get_vertex(qid);

        q.is_right_of(p)
    }
}

/// A struct for a partially initialized half-edge.
///
/// During the construction of the DCEL, we don't know immediately the id of a half-edge's twin
/// or the next/previous half-edge, so we initialize them without these attributes and update them
/// afterwards.
/// However, since in the end every half-edge has them, we don't want to leave `Option`s behind,
/// which is why this temporary struct is useful.
#[derive(Debug)]
struct PartialHedge {
    origin: VertexId,
    twin: Option<HedgeId>,
    face: Option<FaceId>,
    next: Option<HedgeId>,
    prev: Option<HedgeId>,
}

impl PartialHedge {
    fn try_into_hedge(self) -> Result<Hedge> {
        Ok(Hedge {
            origin: self.origin,
            twin: self.twin.ok_or(anyhow!("Missing `twin` field"))?,
            face: self.face,
            next: self.next.ok_or(anyhow!("Missing `next` field"))?,
            prev: self.prev.ok_or(anyhow!("Missing `prev` field"))?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) struct VertexId(pub(crate) usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) struct FaceId(usize);

impl FaceId {
    pub(crate) fn get(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) struct HedgeId(pub(crate) usize);

impl Add<usize> for HedgeId {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl AddAssign<usize> for HedgeId {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs;
    }
}

impl Sub<usize> for HedgeId {
    type Output = Self;

    fn sub(self, rhs: usize) -> Self::Output {
        Self(self.0 - rhs)
    }
}

impl SubAssign<usize> for HedgeId {
    fn sub_assign(&mut self, rhs: usize) {
        self.0 -= rhs;
    }
}

struct FaceVertices<'a> {
    hedge_iterator: FaceHedges<'a>,
}

struct FaceHedges<'a> {
    dcel: &'a Dcel,
    face: FaceId,
    current: HedgeId,
    done: bool,
}

struct ContourHedges<'a> {
    dcel: &'a Dcel,
    contour: usize,
    current: HedgeId,
    done: bool,
}

impl Dcel {
    pub(crate) fn new() -> Self {
        Self {
            vertices: Vec::new(),
            faces: Vec::new(),
            hedges: Vec::new(),
            contours: Vec::new(),
        }
    }

    fn with_capacity(nv: usize, nc: usize, nh: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(nv),
            faces: Vec::with_capacity(nc),
            hedges: Vec::with_capacity(nh),
            contours: Vec::new(),
        }
    }

    fn reserve(&mut self, nv_add: usize, nc_add: usize, nh_add: usize) {
        self.vertices.reserve(nv_add);
        self.faces.reserve(nc_add);
        self.hedges.reserve(nh_add);
    }

    pub(crate) fn get_bounds(&self) -> [f64; 4] {
        let mut xmin = f64::MAX;
        let mut xmax = f64::MIN;
        let mut ymin = f64::MAX;
        let mut ymax = f64::MIN;
        for vertex in &self.vertices {
            let [x, y] = vertex.coords;
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
        [xmin, xmax, ymin, ymax]
    }

    pub(crate) fn get_vertex(&self, VertexId(n): VertexId) -> &Vertex {
        &self.vertices[n]
    }

    fn get_vertex_mut(&mut self, VertexId(n): VertexId) -> &mut Vertex {
        &mut self.vertices[n]
    }

    fn get_face(&self, FaceId(n): FaceId) -> &Face {
        &self.faces[n]
    }

    pub(crate) fn get_hedge(&self, HedgeId(n): HedgeId) -> &Hedge {
        &self.hedges[n]
    }

    fn get_hedge_mut(&mut self, HedgeId(n): HedgeId) -> &mut Hedge {
        &mut self.hedges[n]
    }

    fn face_vertices(&self, face: usize) -> FaceVertices {
        FaceVertices {
            hedge_iterator: self.face_hedges(FaceId(face)),
        }
    }

    fn face_hedges(&self, id: FaceId) -> FaceHedges {
        FaceHedges {
            dcel: self,
            face: id,
            current: self.get_face(id).start,
            done: false,
        }
    }

    fn contour_hedges(&self, contour: usize) -> ContourHedges {
        ContourHedges {
            dcel: self,
            contour,
            current: self.contours[contour],
            done: false,
        }
    }

    fn print_face_hedges(&self, face: usize) {
        for hedge in self.face_hedges(FaceId(face)) {
            let origin = hedge.origin;
            let dest = self.get_hedge(hedge.next).origin;
            let [xo, yo] = self.get_vertex(origin).coords;
            let [xd, yd] = self.get_vertex(dest).coords;
            println!("({}, {}) -> ({}, {})", xo, yo, xd, yd);
        }
    }

    pub(crate) fn add_hedge(&mut self, hedge: Hedge) -> HedgeId {
        let id = self.hedges.len();
        self.hedges.push(hedge);
        HedgeId(id)
    }

    fn add_face(&mut self, face: Face) -> FaceId {
        let id = self.faces.len();
        self.faces.push(face);
        FaceId(id)
    }

    /// Constructs a new [`Dcel`] from a [`Mesh`].
    pub(crate) fn from_mesh(mesh: Mesh) -> Self {
        let nv = mesh.vertex_count();
        let nc = mesh.cell_count();
        let nh = 2 * mesh.facet_count();

        let mut dcel = Self::with_capacity(nv, nc, nh);
        dcel.append(mesh);

        dcel
    }

    fn get_face_coords(&self, face: usize) -> Vec<[f64; 2]> {
        self.face_vertices(face).map(|vert| vert.coords).collect()
    }

    fn get_face_vertex_ids(&self, face: usize) -> Vec<VertexId> {
        self.face_hedges(FaceId(face))
            .map(|hedge| hedge.origin)
            .collect()
    }

    pub(crate) fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub(crate) fn face_count(&self) -> usize {
        self.faces.len()
    }

    pub(crate) fn hedge_count(&self) -> usize {
        self.hedges.len()
    }

    pub(crate) fn slope(&self, hedge_id: HedgeId) -> f64 {
        let (p, q) = self.get_endpoints(hedge_id);
        let [xp, yp] = p.coords;
        let [xq, yq] = q.coords;
        if xp == xq {
            f64::INFINITY
        } else {
            (yq - yp) / (xq - xp)
        }
    }

    pub(crate) fn points_right(&self, hedge_id: HedgeId) -> bool {
        let (p, q) = self.get_endpoints(hedge_id);
        q.is_right_of(p)
    }

    pub(crate) fn get_endpoints(&self, hedge_id: HedgeId) -> (&Vertex, &Vertex) {
        let hedge = self.get_hedge(hedge_id);
        let twin = self.get_hedge(hedge.twin);
        let p = self.get_vertex(hedge.origin);
        let q = self.get_vertex(twin.origin);
        (p, q)
    }

    pub(crate) fn append(&mut self, mesh: Mesh) {
        let nv = mesh.vertex_count();
        let nc = mesh.cell_count();
        let nf = mesh.facet_count();
        let nh = 2 * nf;

        // NOTE: reserve calls `Vec::reserve` which does nothing if the capacity is already enough
        self.reserve(nv, nc, nh);

        // Partially initialized vertices
        let mut vertices: Vec<_> = mesh
            .points()
            .map(|&coords| PartialVertex {
                coords,
                hedge: None,
            })
            .collect();

        let vertex_count = self.vertex_count();
        let face_count = self.face_count();
        let offset = self.hedge_count();
        let mut current_hedge_id = offset;
        let mut edges = HashMap::with_capacity(nf);
        let mut hedges = Vec::with_capacity(nh); // Partially initialized half-edges
        for (idx, cell) in mesh.cells().enumerate() {
            self.add_face(Face {
                start: HedgeId(current_hedge_id),
            });

            let cell_nf = cell.len();
            for (&a, &b) in cell.iter().circular_tuple_windows() {
                if vertices[a].hedge.is_none() {
                    // Only set the half-edge the first time the vertex is seen
                    // This should associate each vertex with the first cell in which it is found,
                    // so that we can match matplotlib's behavior with triangles
                    vertices[a].hedge = Some(HedgeId(current_hedge_id));
                }

                // Determine previous and next half-edge id
                let prev = if a == cell[0] {
                    current_hedge_id + cell_nf - 1
                } else {
                    current_hedge_id - 1
                };
                let next = if b == cell[0] {
                    current_hedge_id + 1 - cell_nf
                } else {
                    current_hedge_id + 1
                };

                // Offset with previous vertex and face counts
                let a = a + vertex_count;
                let b = b + vertex_count;
                let idx = idx + face_count;

                let twin = edges.remove(&(b, a));
                match twin {
                    Some(twin_idx) => {
                        // We have already seen the current half-edge's twin, and we know the current
                        // half-edge is its twin!
                        let twin: &mut PartialHedge = &mut hedges[twin_idx - offset];
                        twin.twin = Some(HedgeId(current_hedge_id));
                    }
                    None => {
                        // Store half-edge id to set the twin id later on
                        edges.insert((a, b), current_hedge_id);
                    }
                }

                hedges.push(PartialHedge {
                    origin: VertexId(a),
                    twin: twin.map(HedgeId),
                    face: Some(FaceId(idx)),
                    next: Some(HedgeId(next)),
                    prev: Some(HedgeId(prev)),
                });

                current_hedge_id += 1;
            }
        }

        // Add vertices (they should all have been visited by now)
        self.vertices.extend(vertices.into_iter().map(|vertex| {
            vertex
                .try_into_vertex()
                .expect("All vertices should be complete at this point")
        }));

        // Find boundary half-edges
        // TODO: only works for one boundary, make it work when there are holes as well
        let first_outer_hedge_id = current_hedge_id;
        self.contours.push(HedgeId(first_outer_hedge_id));
        let start_id = *edges.values().min().unwrap();
        let mut inner_hedge_id = start_id;
        'outer: loop {
            // These are wrong for the first and last half-edge of the contour, but this is fixed
            // when exiting the loop
            let next = current_hedge_id + 1;
            let prev = current_hedge_id - 1;
            let next_inner_hedge_id = hedges[inner_hedge_id - offset]
                .next
                .expect("There should always be a next hedge here")
                .0;
            hedges.push(PartialHedge {
                origin: hedges[next_inner_hedge_id - offset].origin,
                twin: Some(HedgeId(inner_hedge_id)),
                face: None,
                next: Some(HedgeId(next)),
                prev: Some(HedgeId(prev)),
            });
            hedges[inner_hedge_id - offset].twin = Some(HedgeId(current_hedge_id));
            current_hedge_id += 1;
            loop {
                // Iterate over the current origin's `umbrella` to find the next hedge
                inner_hedge_id = hedges[inner_hedge_id - offset]
                    .prev
                    .expect("There should always be a previous hedge here")
                    .0;
                if inner_hedge_id == start_id {
                    // Means we are back at the start, time to fix the first and last half-edge
                    let last_outer_hedge_id = current_hedge_id - 1;
                    hedges[first_outer_hedge_id - offset].prev = Some(HedgeId(last_outer_hedge_id));
                    hedges[last_outer_hedge_id - offset].next = Some(HedgeId(first_outer_hedge_id));
                    break 'outer;
                }
                if let Some(HedgeId(twin)) = hedges[inner_hedge_id - offset].twin {
                    inner_hedge_id = twin;
                } else {
                    break;
                }
            }
        }

        // Finalize hedges
        self.hedges.extend(hedges.into_iter().map(|hedge| {
            hedge
                .try_into_hedge()
                .expect("All hedges should be complete at this point")
        }));
    }
}

impl<'a> Iterator for FaceVertices<'a> {
    type Item = &'a Vertex;

    fn next(&mut self) -> Option<Self::Item> {
        self.hedge_iterator
            .next()
            .map(|hedge| self.hedge_iterator.dcel.get_vertex(hedge.origin))
    }
}

impl<'a> Iterator for FaceHedges<'a> {
    type Item = &'a Hedge;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let hedge = &self.dcel.get_hedge(self.current);
        self.current = hedge.next;
        let start = self.dcel.get_face(self.face).start;
        if self.current == start {
            self.done = true;
        }
        Some(hedge)
    }
}

impl<'a> Iterator for ContourHedges<'a> {
    type Item = &'a Hedge;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let hedge = self.dcel.get_hedge(self.current);
        self.current = hedge.next;
        let start = self.dcel.contours[self.contour];
        if self.current == start {
            self.done = true;
        }
        Some(hedge)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_triangle() -> Mesh {
        //
        //                             Half-edges
        //   2                |
        //   +                |        +
        //   |\               |        |\
        //   | \              |        | \
        //   |  \             |        |  \
        //   |   \            |       4|  1\5
        //   |    \           |        |2   \
        //   |  0  \          |        |     \
        //   |      \         |        |   0  \
        //   +-------+        |        +-------+
        //   0       1        |            3
        //
        let points = vec![[0., 0.], [1., 0.], [0., 1.]];
        let cells = vec![0, 1, 2];
        Mesh::with_stride(points, cells, 3).expect("This should be a valid input")
    }

    fn two_triangles() -> Mesh {
        //
        //                             Half-edges
        //   3       2        |            8
        //   +-------+        |        +-------+
        //   |\      |        |        |\  4   |
        //   | \  1  |        |        | \     |
        //   |  \    |        |        |  \   3|
        //   |   \   |        |       7|  1\5  |9
        //   |    \  |        |        |2   \  |
        //   |  0  \ |        |        |     \ |
        //   |      \|        |        |   0  \|
        //   +-------+        |        +-------+
        //   0       1        |            6
        //
        let points = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
        let cells = vec![
            0, 1, 3, // first tri
            1, 2, 3, // second tri
        ];
        Mesh::with_stride(points, cells, 3).expect("This should be a valid input")
    }

    fn four_quadrangles() -> Mesh {
        //
        //                                   Half-edges
        //         7              |           19   20
        // 6 +-----+-----+ 8      |        +-----+-----+
        //   |     |     |        |        |  10 | 14  |
        //   |  2  |  3  |        |      18|11  9|15 13|21
        //   |     |     |        |        |  8  | 12  |
        // 3 +-----4-----+ 5      |        +-----+-----+
        //   |     |     |        |        |  2  |  6  |
        //   |  0  |  1  |        |      17|3   1|7   5|22
        //   |     |     |        |        |  0  |  4  |
        //   +-----+-----+        |        +-----+-----+
        //   0     1     2        |          16     23
        //

        let points = vec![
            [0., 0.],
            [1., 0.],
            [2., 0.],
            [0., 1.],
            [1., 1.],
            [2., 1.],
            [0., 2.],
            [1., 2.],
            [2., 2.],
        ];
        let cells = vec![
            0, 1, 4, 3, // first quad
            1, 2, 5, 4, // second quad
            3, 4, 7, 6, // third quad
            4, 5, 8, 7, // fourth quad
        ];
        Mesh::with_stride(points, cells, 4).expect("This should be a valid input")
    }

    #[test]
    fn create_one_triangle_dcel() {
        let mesh = one_triangle();

        let dcel = Dcel::from_mesh(mesh);

        let vertex_ids = dcel.get_face_vertex_ids(0);
        let expected_vertex_ids = [0, 1, 2].into_iter().map(VertexId).collect::<Vec<_>>();
        assert_eq!(vertex_ids, expected_vertex_ids);
        let verts = dcel.get_face_coords(0);
        let expected_coords = vec![[0., 0.], [1., 0.], [0., 1.]];
        assert_eq!(verts, expected_coords);
    }

    #[test]
    fn create_two_triangle_dcel() {
        let mesh = two_triangles();

        let dcel = Dcel::from_mesh(mesh);

        // Check vertex ids
        let expected_vertex_ids: Vec<_> = [0, 1, 3].into_iter().map(VertexId).collect();
        assert_eq!(dcel.get_face_vertex_ids(0), expected_vertex_ids);
        let expected_vertex_ids: Vec<_> = [1, 2, 3].into_iter().map(VertexId).collect();
        assert_eq!(dcel.get_face_vertex_ids(1), expected_vertex_ids);

        // Check vertex coordinates
        let verts = dcel.get_face_coords(0);
        assert_eq!(verts, vec![[0., 0.], [1., 0.], [0., 1.]]);
        let verts = dcel.get_face_coords(1);
        assert_eq!(verts, vec![[1., 0.], [1., 1.], [0., 1.]]);

        // Check face ids
        for face in [0, 1] {
            for hedge in dcel.face_hedges(FaceId(face)) {
                assert_eq!(hedge.face, Some(FaceId(face)));
            }
        }

        // Check next and prev
        let nexts: Vec<_> = dcel
            .face_hedges(FaceId(0))
            .map(|hedge| hedge.next)
            .collect();
        let expected_nexts: Vec<_> = [1, 2, 0].into_iter().map(HedgeId).collect();
        assert_eq!(nexts, expected_nexts);
        let prevs: Vec<_> = dcel
            .face_hedges(FaceId(0))
            .map(|hedge| hedge.prev)
            .collect();
        let expected_prevs: Vec<_> = [2, 0, 1].into_iter().map(HedgeId).collect();
        assert_eq!(prevs, expected_prevs);
        let nexts: Vec<_> = dcel
            .face_hedges(FaceId(1))
            .map(|hedge| hedge.next)
            .collect();
        let expected_nexts: Vec<_> = [4, 5, 3].into_iter().map(HedgeId).collect();
        assert_eq!(nexts, expected_nexts);
        let prevs: Vec<_> = dcel
            .face_hedges(FaceId(1))
            .map(|hedge| hedge.prev)
            .collect();
        let expected_prevs: Vec<_> = [5, 3, 4].into_iter().map(HedgeId).collect();
        assert_eq!(prevs, expected_prevs);

        // Check contour
        let origins: Vec<_> = dcel.contour_hedges(0).map(|hedge| hedge.origin).collect();
        let expected_origins: Vec<_> = [1, 0, 3, 2].into_iter().map(VertexId).collect();
        assert_eq!(origins, expected_origins);
    }

    #[test]
    fn create_four_quadrangle_dcel() {
        let mesh = four_quadrangles();

        let dcel = Dcel::from_mesh(mesh);

        // Check vertex ids
        let expected_vertex_ids: Vec<_> = [0, 1, 4, 3].into_iter().map(VertexId).collect();
        assert_eq!(dcel.get_face_vertex_ids(0), expected_vertex_ids);
        let expected_vertex_ids: Vec<_> = [1, 2, 5, 4].into_iter().map(VertexId).collect();
        assert_eq!(dcel.get_face_vertex_ids(1), expected_vertex_ids);
        let expected_vertex_ids: Vec<_> = [3, 4, 7, 6].into_iter().map(VertexId).collect();
        assert_eq!(dcel.get_face_vertex_ids(2), expected_vertex_ids);
        let expected_vertex_ids: Vec<_> = [4, 5, 8, 7].into_iter().map(VertexId).collect();
        assert_eq!(dcel.get_face_vertex_ids(3), expected_vertex_ids);

        // Check vertex coordinates
        let verts = dcel.get_face_coords(0);
        assert_eq!(verts, vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]]);
        let verts = dcel.get_face_coords(1);
        assert_eq!(verts, vec![[1., 0.], [2., 0.], [2., 1.], [1., 1.]]);

        // Check face ids
        for face in [0, 1] {
            for hedge in dcel.face_hedges(FaceId(face)) {
                assert_eq!(hedge.face, Some(FaceId(face)));
            }
        }

        // Check next and prev
        let nexts: Vec<_> = dcel
            .face_hedges(FaceId(0))
            .map(|hedge| hedge.next)
            .collect();
        let expected_nexts: Vec<_> = [1, 2, 3, 0].into_iter().map(HedgeId).collect();
        assert_eq!(nexts, expected_nexts);
        let prevs: Vec<_> = dcel
            .face_hedges(FaceId(0))
            .map(|hedge| hedge.prev)
            .collect();
        let expected_prevs: Vec<_> = [3, 0, 1, 2].into_iter().map(HedgeId).collect();
        assert_eq!(prevs, expected_prevs);

        let nexts: Vec<_> = dcel
            .face_hedges(FaceId(1))
            .map(|hedge| hedge.next)
            .collect();
        let expected_nexts: Vec<_> = [5, 6, 7, 4].into_iter().map(HedgeId).collect();
        assert_eq!(nexts, expected_nexts);
        let prevs: Vec<_> = dcel
            .face_hedges(FaceId(1))
            .map(|hedge| hedge.prev)
            .collect();
        let expected_prevs: Vec<_> = [7, 4, 5, 6].into_iter().map(HedgeId).collect();
        assert_eq!(prevs, expected_prevs);

        let nexts: Vec<_> = dcel
            .face_hedges(FaceId(2))
            .map(|hedge| hedge.next)
            .collect();
        let expected_nexts: Vec<_> = [9, 10, 11, 8].into_iter().map(HedgeId).collect();
        assert_eq!(nexts, expected_nexts);
        let prevs: Vec<_> = dcel
            .face_hedges(FaceId(2))
            .map(|hedge| hedge.prev)
            .collect();
        let expected_prevs: Vec<_> = [11, 8, 9, 10].into_iter().map(HedgeId).collect();
        assert_eq!(prevs, expected_prevs);

        let nexts: Vec<_> = dcel
            .face_hedges(FaceId(3))
            .map(|hedge| hedge.next)
            .collect();
        let expected_nexts: Vec<_> = [13, 14, 15, 12].into_iter().map(HedgeId).collect();
        assert_eq!(nexts, expected_nexts);
        let prevs: Vec<_> = dcel
            .face_hedges(FaceId(3))
            .map(|hedge| hedge.prev)
            .collect();
        let expected_prevs: Vec<_> = [15, 12, 13, 14].into_iter().map(HedgeId).collect();
        assert_eq!(prevs, expected_prevs);

        // Check contour
        let origins: Vec<_> = dcel.contour_hedges(0).map(|hedge| hedge.origin).collect();
        let expected_origins: Vec<_> = [1, 0, 3, 6, 7, 8, 5, 2].into_iter().map(VertexId).collect();
        assert_eq!(origins, expected_origins);
    }

    #[test]
    fn twins() {
        // Triangles
        let mesh = two_triangles();

        let dcel = Dcel::from_mesh(mesh);

        // Check individual twins
        let twins: Vec<_> = dcel.hedges.iter().map(|hedge| hedge.twin).collect();
        let expected_twins: Vec<_> = [6, 5, 7, 9, 8, 1, 0, 2, 4, 3]
            .into_iter()
            .map(HedgeId)
            .collect();
        assert_eq!(twins, expected_twins);

        // Check that the twins' origin is the same as the next hedge's origin
        for (i, _) in dcel.faces.iter().enumerate() {
            for hedge in dcel.face_hedges(FaceId(i)) {
                assert_eq!(
                    dcel.get_hedge(hedge.twin).origin,
                    dcel.get_hedge(hedge.next).origin
                );
            }
        }

        // Quadrangles
        let mesh = four_quadrangles();

        let dcel = Dcel::from_mesh(mesh);

        // Check individual twins
        let twins: Vec<_> = dcel.hedges.iter().map(|hedge| hedge.twin).collect();
        let expected_twins: Vec<_> = [
            16, 7, 8, 17, 23, 22, 12, 1, 2, 15, 19, 18, 6, 21, 20, 9, 0, 3, 11, 10, 14, 13, 5, 4,
        ]
        .into_iter()
        .map(HedgeId)
        .collect();
        assert_eq!(twins, expected_twins);

        // Check that the twins' origin is the same as the next hedge's origin
        for (i, _) in dcel.faces.iter().enumerate() {
            for hedge in dcel.face_hedges(FaceId(i)) {
                assert_eq!(
                    dcel.get_hedge(hedge.twin).origin,
                    dcel.get_hedge(hedge.next).origin
                );
            }
        }
    }

    #[test]
    fn bounds() {
        let mesh = two_triangles();

        let dcel = Dcel::from_mesh(mesh);

        let [xmin, xmax, ymin, ymax] = dcel.get_bounds();

        assert_eq!(xmin, 0.);
        assert_eq!(xmax, 1.);
        assert_eq!(ymin, 0.);
        assert_eq!(ymax, 1.);
    }

    #[test]
    fn right_of() {
        let v0 = Vertex {
            coords: [0., 0.],
            hedge: HedgeId(0),
        };

        assert!([1., 0.].is_right_of(&v0));
        assert!(![-1., 0.].is_right_of(&v0));
        assert!([0., 1.].is_right_of(&v0));
        assert!(![0., -1.].is_right_of(&v0));
        assert!(!v0.is_right_of(&v0));
    }

    #[test]
    fn points_to_the_right() {
        let points = vec![[0., 0.], [1., 0.], [0.5, 0.5]];
        let cells = vec![0, 1, 2];
        let dcel = Dcel::from_mesh(Mesh::with_stride(points, cells, 3).unwrap());

        assert!(dcel.get_hedge(HedgeId(0)).points_to_the_right(&dcel));
        assert!(!dcel.get_hedge(HedgeId(1)).points_to_the_right(&dcel));
        assert!(!dcel.get_hedge(HedgeId(2)).points_to_the_right(&dcel));
        assert!(!dcel.get_hedge(HedgeId(3)).points_to_the_right(&dcel));
        assert!(dcel.get_hedge(HedgeId(4)).points_to_the_right(&dcel));
        assert!(dcel.get_hedge(HedgeId(5)).points_to_the_right(&dcel));
    }

    #[test]
    fn append_mesh() {
        let points = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
        let cells = vec![0, 1, 2, 3];
        let mesh = Mesh::with_stride(points, cells, 4).expect("This should be a valid input");

        // Create empty DCEL
        let mut dcel = Dcel::new();

        // Append the mesh
        dcel.append(mesh);

        assert_eq!(dcel.face_count(), 1);
        assert_eq!(dcel.vertex_count(), 4);

        let vertex_ids = dcel.get_face_vertex_ids(0);
        let expected_vertex_ids = [0, 1, 2, 3].into_iter().map(VertexId).collect::<Vec<_>>();
        assert_eq!(vertex_ids, expected_vertex_ids);
        let verts = dcel.get_face_coords(0);
        let expected_coords = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
        assert_eq!(verts, expected_coords);
    }

    #[test]
    fn append_mesh_to_existing_dcel() {
        let points = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
        let cells = vec![0, 1, 2, 3];
        let mesh = Mesh::with_stride(points, cells, 4).expect("This should be a valid input");

        // Create DCEL with first mesh
        let mut dcel = Dcel::from_mesh(mesh);

        // Create second mesh
        let points = vec![[2., 0.], [3., 0.], [3., 1.], [2., 1.]];
        let cells = vec![0, 1, 2, 3];
        let mesh = Mesh::with_stride(points, cells, 4).expect("This should be a valid input");

        // Append the mesh
        dcel.append(mesh);

        assert_eq!(dcel.face_count(), 2);
        assert_eq!(dcel.vertex_count(), 8);

        let vertex_ids = dcel.get_face_vertex_ids(1);
        let expected_vertex_ids = [4, 5, 6, 7].into_iter().map(VertexId).collect::<Vec<_>>();
        assert_eq!(vertex_ids, expected_vertex_ids);
        let verts = dcel.get_face_coords(1);
        let expected_coords = vec![[2., 0.], [3., 0.], [3., 1.], [2., 1.]];
        assert_eq!(verts, expected_coords);
    }
}
