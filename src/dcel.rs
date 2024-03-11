use std::{
    collections::HashMap,
    fmt::Display,
    ops::{Add, AddAssign, Sub, SubAssign},
};

#[derive(Debug)]
pub(crate) struct Dcel {
    vertices: Vec<Vertex>,
    faces: Vec<Face>,
    hedges: Vec<Hedge>,
    contours: Vec<HedgeId>,
}

#[derive(Debug)]
pub(crate) struct Vertex {
    coords: [f32; 2],
    hedge: HedgeId,
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

#[derive(Debug)]
pub(crate) struct Hedge {
    origin: VertexId,
    twin: HedgeId,
    face: Option<FaceId>,
    next: HedgeId,
    prev: HedgeId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VertexId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FaceId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HedgeId(usize);

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

struct FaceVerticesIterator<'a> {
    hedge_iterator: FaceHedgesIterator<'a>,
}

struct FaceHedgesIterator<'a> {
    dcel: &'a Dcel,
    face: FaceId,
    current: HedgeId,
    done: bool,
}

struct ContourHedgesIterator<'a> {
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

    fn with_capacity(n_vertices: usize, n_faces: usize, n_sides: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(n_vertices),
            faces: Vec::with_capacity(n_faces),
            hedges: Vec::with_capacity(n_faces * n_sides),
            contours: Vec::new(),
        }
    }

    fn get_vertex(&self, VertexId(n): VertexId) -> &Vertex {
        &self.vertices[n]
    }

    fn get_vertex_mut(&mut self, VertexId(n): VertexId) -> &mut Vertex {
        &mut self.vertices[n]
    }

    fn get_face(&self, FaceId(n): FaceId) -> &Face {
        &self.faces[n]
    }

    fn get_hedge(&self, HedgeId(n): HedgeId) -> &Hedge {
        &self.hedges[n]
    }

    fn get_hedge_mut(&mut self, HedgeId(n): HedgeId) -> &mut Hedge {
        &mut self.hedges[n]
    }

    fn iter_face_vertices(&self, face: usize) -> FaceVerticesIterator {
        FaceVerticesIterator {
            hedge_iterator: self.iter_face_hedges(FaceId(face)),
        }
    }

    fn iter_face_hedges(&self, id: FaceId) -> FaceHedgesIterator {
        FaceHedgesIterator {
            dcel: self,
            face: id,
            current: self.get_face(id).start,
            done: false,
        }
    }

    fn iter_contour_hedges(&self, contour: usize) -> ContourHedgesIterator {
        ContourHedgesIterator {
            dcel: self,
            contour,
            current: self.contours[contour],
            done: false,
        }
    }

    fn print_face_hedges(&self, face: usize) {
        for hedge in self.iter_face_hedges(FaceId(face)) {
            let origin = hedge.origin;
            let dest = self.get_hedge(hedge.next).origin;
            let [xo, yo] = self.get_vertex(origin).coords;
            let [xd, yd] = self.get_vertex(dest).coords;
            println!("({}, {}) -> ({}, {})", xo, yo, xd, yd);
        }
    }

    fn add_vertices(&mut self, vertices: &[[f32; 2]]) {
        for &coords in vertices {
            self.vertices.push(Vertex {
                coords,
                hedge: HedgeId(usize::MAX),
            });
        }
    }

    fn add_hedge(&mut self, hedge: Hedge) -> HedgeId {
        let id = self.hedges.len();
        self.hedges.push(hedge);
        HedgeId(id)
    }

    fn add_face(&mut self, face: Face) -> FaceId {
        let id = self.faces.len();
        self.faces.push(face);
        FaceId(id)
    }

    fn from_polygon_soup<const N: usize>(vertices: &[[f32; 2]], polygons: &[[usize; N]]) -> Self {
        let mut dcel = Self::with_capacity(vertices.len(), polygons.len(), N);
        dcel.add_vertices(vertices);

        let mut current_hedge_id = HedgeId(0);
        let mut edges = HashMap::with_capacity(polygons.len() * N);
        for (face, polygon) in polygons.iter().enumerate() {
            dcel.add_face(Face {
                start: current_hedge_id,
            });

            for (iloc, &vert) in polygon.iter().enumerate() {
                // Determine previous and next half-edge id
                let (prev, next) = match iloc {
                    0 => (current_hedge_id + N - 1, current_hedge_id + 1),
                    ivert if ivert == N - 1 => (current_hedge_id - 1, current_hedge_id + 1 - N),
                    _ => (current_hedge_id - 1, current_hedge_id + 1),
                };

                // Determine indices of edge vertices
                let dest = if iloc == polygon.len() - 1 {
                    polygon[0]
                } else {
                    polygon[iloc + 1]
                };
                let twin = if let Some(twin) = edges.remove(&(dest, vert)) {
                    // We have already seen the current half-edge's twin, and we know the current
                    // half-edge is its twin!
                    dcel.get_hedge_mut(twin).twin = current_hedge_id;
                    twin
                } else {
                    // Store half-edge id to set the twin id later on
                    edges.insert((vert, dest), current_hedge_id);
                    HedgeId(usize::MAX)
                };

                dcel.add_hedge(Hedge {
                    origin: VertexId(vert),
                    twin,
                    face: Some(FaceId(face)),
                    next,
                    prev,
                });
                dcel.get_vertex_mut(VertexId(vert)).hedge = current_hedge_id;
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
        self.iter_face_vertices(face)
            .map(|vert| vert.coords)
            .collect()
    }

    fn get_face_vertex_ids(&self, face: usize) -> Vec<VertexId> {
        self.iter_face_hedges(FaceId(face))
            .map(|hedge| hedge.origin)
            .collect()
    }
}

impl<'a> Iterator for FaceVerticesIterator<'a> {
    type Item = &'a Vertex;

    fn next(&mut self) -> Option<Self::Item> {
        self.hedge_iterator
            .next()
            .map(|hedge| self.hedge_iterator.dcel.get_vertex(hedge.origin))
    }
}

impl<'a> Iterator for FaceHedgesIterator<'a> {
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

impl<'a> Iterator for ContourHedgesIterator<'a> {
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

    fn two_triangles() -> (Vec<[f32; 2]>, Vec<[usize; 3]>) {
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
        let vertices = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
        let polygons = vec![[0, 1, 3], [1, 2, 3]];
        (vertices, polygons)
    }

    fn four_quadrangles() -> (Vec<[f32; 2]>, Vec<[usize; 4]>) {
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

        let vertices = vec![
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
        let polygons = vec![[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]];
        (vertices, polygons)
    }

    #[test]
    fn create_one_triangle_dcel_from_polygon_soup() {
        let vertices = vec![[0., 0.], [1., 0.], [0., 1.]];
        let polygons = vec![[0, 1, 2]];

        let dcel = Dcel::from_polygon_soup(&vertices, &polygons);

        let vertex_ids = dcel.get_face_vertex_ids(0);
        let expected_vertex_ids = polygons[0].into_iter().map(VertexId).collect::<Vec<_>>();
        assert_eq!(vertex_ids, expected_vertex_ids);
        let verts = dcel.get_face_coords(0);
        assert_eq!(verts, vertices);
    }

    #[test]
    fn create_two_triangle_dcel_from_polygon_soup() {
        let (vertices, polygons) = two_triangles();

        let dcel = Dcel::from_polygon_soup(&vertices, &polygons);

        // Check vertex ids
        for face in [0, 1] {
            let vertex_ids = dcel.get_face_vertex_ids(face);
            let expected_vertex_ids = polygons[face].into_iter().map(VertexId).collect::<Vec<_>>();
            assert_eq!(vertex_ids, expected_vertex_ids);
        }

        // Check vertex coordinates
        let verts = dcel.get_face_coords(0);
        assert_eq!(verts, vec![[0., 0.], [1., 0.], [0., 1.]]);
        let verts = dcel.get_face_coords(1);
        assert_eq!(verts, vec![[1., 0.], [1., 1.], [0., 1.]]);

        // Check face ids
        for face in [0, 1] {
            for hedge in dcel.iter_face_hedges(FaceId(face)) {
                assert_eq!(hedge.face, Some(FaceId(face)));
            }
        }

        // Check next and prev
        let nexts: Vec<_> = dcel
            .iter_face_hedges(FaceId(0))
            .map(|hedge| hedge.next)
            .collect();
        let expected_nexts: Vec<_> = [1, 2, 0].into_iter().map(HedgeId).collect();
        assert_eq!(nexts, expected_nexts);
        let prevs: Vec<_> = dcel
            .iter_face_hedges(FaceId(0))
            .map(|hedge| hedge.prev)
            .collect();
        let expected_prevs: Vec<_> = [2, 0, 1].into_iter().map(HedgeId).collect();
        assert_eq!(prevs, expected_prevs);
        let nexts: Vec<_> = dcel
            .iter_face_hedges(FaceId(1))
            .map(|hedge| hedge.next)
            .collect();
        let expected_nexts: Vec<_> = [4, 5, 3].into_iter().map(HedgeId).collect();
        assert_eq!(nexts, expected_nexts);
        let prevs: Vec<_> = dcel
            .iter_face_hedges(FaceId(1))
            .map(|hedge| hedge.prev)
            .collect();
        let expected_prevs: Vec<_> = [5, 3, 4].into_iter().map(HedgeId).collect();
        assert_eq!(prevs, expected_prevs);

        // Check contour
        let origins: Vec<_> = dcel
            .iter_contour_hedges(0)
            .map(|hedge| hedge.origin)
            .collect();
        let expected_origins: Vec<_> = [1, 0, 3, 2].into_iter().map(VertexId).collect();
        assert_eq!(origins, expected_origins);
    }

    #[test]
    fn create_four_quadrangles_from_polygon_soup() {
        let (vertices, polygons) = four_quadrangles();

        let dcel = Dcel::from_polygon_soup(&vertices, &polygons);

        // Check vertex ids
        for face in [0, 1] {
            let vertex_ids = dcel.get_face_vertex_ids(face);
            let expected_vertex_ids: Vec<_> = polygons[face].into_iter().map(VertexId).collect();
            assert_eq!(vertex_ids, expected_vertex_ids);
        }

        // Check vertex coordinates
        let verts = dcel.get_face_coords(0);
        assert_eq!(verts, vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]]);
        let verts = dcel.get_face_coords(1);
        assert_eq!(verts, vec![[1., 0.], [2., 0.], [2., 1.], [1., 1.]]);

        // Check face ids
        for face in [0, 1] {
            for hedge in dcel.iter_face_hedges(FaceId(face)) {
                assert_eq!(hedge.face, Some(FaceId(face)));
            }
        }

        // Check next and prev
        let nexts: Vec<_> = dcel
            .iter_face_hedges(FaceId(0))
            .map(|hedge| hedge.next)
            .collect();
        let expected_nexts: Vec<_> = [1, 2, 3, 0].into_iter().map(HedgeId).collect();
        assert_eq!(nexts, expected_nexts);
        let prevs: Vec<_> = dcel
            .iter_face_hedges(FaceId(0))
            .map(|hedge| hedge.prev)
            .collect();
        let expected_prevs: Vec<_> = [3, 0, 1, 2].into_iter().map(HedgeId).collect();
        assert_eq!(prevs, expected_prevs);

        let nexts: Vec<_> = dcel
            .iter_face_hedges(FaceId(1))
            .map(|hedge| hedge.next)
            .collect();
        let expected_nexts: Vec<_> = [5, 6, 7, 4].into_iter().map(HedgeId).collect();
        assert_eq!(nexts, expected_nexts);
        let prevs: Vec<_> = dcel
            .iter_face_hedges(FaceId(1))
            .map(|hedge| hedge.prev)
            .collect();
        let expected_prevs: Vec<_> = [7, 4, 5, 6].into_iter().map(HedgeId).collect();
        assert_eq!(prevs, expected_prevs);

        let nexts: Vec<_> = dcel
            .iter_face_hedges(FaceId(2))
            .map(|hedge| hedge.next)
            .collect();
        let expected_nexts: Vec<_> = [9, 10, 11, 8].into_iter().map(HedgeId).collect();
        assert_eq!(nexts, expected_nexts);
        let prevs: Vec<_> = dcel
            .iter_face_hedges(FaceId(2))
            .map(|hedge| hedge.prev)
            .collect();
        let expected_prevs: Vec<_> = [11, 8, 9, 10].into_iter().map(HedgeId).collect();
        assert_eq!(prevs, expected_prevs);

        let nexts: Vec<_> = dcel
            .iter_face_hedges(FaceId(3))
            .map(|hedge| hedge.next)
            .collect();
        let expected_nexts: Vec<_> = [13, 14, 15, 12].into_iter().map(HedgeId).collect();
        assert_eq!(nexts, expected_nexts);
        let prevs: Vec<_> = dcel
            .iter_face_hedges(FaceId(3))
            .map(|hedge| hedge.prev)
            .collect();
        let expected_prevs: Vec<_> = [15, 12, 13, 14].into_iter().map(HedgeId).collect();
        assert_eq!(prevs, expected_prevs);

        // Check contour
        let origins: Vec<_> = dcel
            .iter_contour_hedges(0)
            .map(|hedge| hedge.origin)
            .collect();
        let expected_origins: Vec<_> = [1, 0, 3, 6, 7, 8, 5, 2].into_iter().map(VertexId).collect();
        assert_eq!(origins, expected_origins);
    }

    #[test]
    fn twins() {
        // Triangles
        let (vertices, polygons) = two_triangles();

        let dcel = Dcel::from_polygon_soup(&vertices, &polygons);

        // Check individual twins
        let twins: Vec<_> = dcel.hedges.iter().map(|hedge| hedge.twin).collect();
        let expected_twins: Vec<_> = [6, 5, 7, 9, 8, 1, 0, 2, 4, 3]
            .into_iter()
            .map(HedgeId)
            .collect();
        assert_eq!(twins, expected_twins);

        // Check that the twins' origin is the same as the next hedge's origin
        for (i, _) in dcel.faces.iter().enumerate() {
            for hedge in dcel.iter_face_hedges(FaceId(i)) {
                assert_eq!(
                    dcel.get_hedge(hedge.twin).origin,
                    dcel.get_hedge(hedge.next).origin
                );
            }
        }

        // Quadrangles
        let (vertices, polygons) = four_quadrangles();

        let dcel = Dcel::from_polygon_soup(&vertices, &polygons);

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
            for hedge in dcel.iter_face_hedges(FaceId(i)) {
                assert_eq!(
                    dcel.get_hedge(hedge.twin).origin,
                    dcel.get_hedge(hedge.next).origin
                );
            }
        }
    }
}
