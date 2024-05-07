#![allow(dead_code)]

use std::{
    collections::HashMap,
    fmt::Display,
    ops::{Add, AddAssign, Sub, SubAssign},
};

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
    pub(crate) coords: [f32; 2],
    hedge: HedgeId,
}

trait IsRightOf<Rhs = Self> {
    fn is_right_of(&self, other: &Rhs) -> bool;
}

impl IsRightOf<Vertex> for [f32; 2] {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) struct VertexId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) struct FaceId(usize);

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

pub(crate) enum Offsets {
    Implicit { stride: usize, n_cells: usize },
    Explicit(Vec<usize>),
}

impl Offsets {
    fn iter(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        match self {
            Offsets::Implicit { stride, n_cells } => {
                Box::new((0..=(stride * n_cells)).step_by(*stride))
            }
            Offsets::Explicit(offsets) => Box::new(offsets.iter().copied()),
        }
    }

    fn n_cells(&self) -> usize {
        match self {
            Offsets::Implicit { stride: _, n_cells } => *n_cells,
            Offsets::Explicit(offsets) => offsets.len() - 1,
        }
    }

    fn get(&self, idx: usize) -> Option<usize> {
        match self {
            Offsets::Implicit { stride, n_cells } => (idx <= *n_cells).then_some(stride * idx),
            Offsets::Explicit(offsets) => offsets.get(idx).cloned(),
        }
    }
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

    fn with_capacity(nv: usize, nc: usize, nf: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(nv),
            faces: Vec::with_capacity(nc),
            hedges: Vec::with_capacity(nf),
            contours: Vec::new(),
        }
    }

    pub(crate) fn get_bounds(&self) -> [f32; 4] {
        let mut xmin = f32::MAX;
        let mut xmax = f32::MIN;
        let mut ymin = f32::MAX;
        let mut ymax = f32::MIN;
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

    fn add_vertices<'a, I>(&mut self, vertices: I)
    where
        I: IntoIterator<Item = &'a [f32; 2]>,
    {
        for coords in vertices.into_iter() {
            self.vertices.push(Vertex {
                coords: *coords,
                hedge: HedgeId(usize::MAX),
            });
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
        let nf = mesh.facet_count();

        let mut dcel = Self::with_capacity(nv, nc, nf);
        dcel.add_vertices(mesh.points());

        let mut current_hedge_id = HedgeId(0);
        let mut edges = HashMap::with_capacity(nf);
        for (idx, cell) in mesh.cells().enumerate() {
            dcel.add_face(Face {
                start: current_hedge_id,
            });

            let cell_nf = cell.len();
            for (&a, &b) in cell.iter().circular_tuple_windows() {
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

                let twin = if let Some(twin) = edges.remove(&(b, a)) {
                    // We have already seen the current half-edge's twin, and we know the current
                    // half-edge is its twin!
                    dcel.get_hedge_mut(twin).twin = current_hedge_id;
                    twin
                } else {
                    // Store half-edge id to set the twin id later on
                    edges.insert((a, b), current_hedge_id);
                    HedgeId(usize::MAX)
                };

                dcel.add_hedge(Hedge {
                    origin: VertexId(a),
                    twin,
                    face: Some(FaceId(idx)),
                    next,
                    prev,
                });
                dcel.get_vertex_mut(VertexId(a)).hedge = current_hedge_id;

                current_hedge_id += 1;
            }
        }

        // Find boundary half-edges
        // TODO: only works for one boundary, make it work when there are holes as well
        let first_outer_hedge_id = current_hedge_id;
        dcel.contours.push(first_outer_hedge_id);
        let start_id = HedgeId(edges.values().map(|hedge_id| hedge_id.0).min().unwrap());
        let mut inner_hedge_id = start_id;
        'outer: loop {
            // These are wrong for the first and last half-edge of the contour, but this is fixed
            // when exiting the loop
            let next = current_hedge_id + 1;
            let prev = current_hedge_id - 1;
            dcel.add_hedge(Hedge {
                origin: dcel.get_hedge(dcel.get_hedge(inner_hedge_id).next).origin,
                twin: inner_hedge_id,
                face: None,
                next,
                prev,
            });
            dcel.get_hedge_mut(inner_hedge_id).twin = current_hedge_id;
            current_hedge_id += 1;
            loop {
                // Iterate over the current origin's `umbrella` to find the next hedge
                inner_hedge_id = dcel.get_hedge(inner_hedge_id).prev;
                if inner_hedge_id == start_id {
                    // Means we are back at the start, time to fix the first and last half-edge
                    let last_outer_hedge_id = current_hedge_id - 1;
                    dcel.get_hedge_mut(first_outer_hedge_id).prev = last_outer_hedge_id;
                    dcel.get_hedge_mut(last_outer_hedge_id).next = first_outer_hedge_id;
                    break 'outer;
                }
                if dcel.get_hedge(inner_hedge_id).twin.0 < usize::MAX {
                    inner_hedge_id = dcel.get_hedge(inner_hedge_id).twin;
                } else {
                    break;
                }
            }
        }

        dcel
    }

    fn get_face_coords(&self, face: usize) -> Vec<[f32; 2]> {
        self.face_vertices(face).map(|vert| vert.coords).collect()
    }

    fn get_face_vertex_ids(&self, face: usize) -> Vec<VertexId> {
        self.face_hedges(FaceId(face))
            .map(|hedge| hedge.origin)
            .collect()
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
}
