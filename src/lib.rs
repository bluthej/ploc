use std::{collections::HashMap, fmt::Display};

#[derive(Debug)]
struct Dcel {
    vertices: Vec<Vertex>,
    faces: Vec<Face>,
    hedges: Vec<Hedge>,
    contours: Vec<Face>,
}

#[derive(Debug)]
struct Vertex {
    coords: [f32; 2],
    hedge: usize,
}

impl Display for Vertex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [x, y] = self.coords;
        write!(f, "({}, {})", x, y)
    }
}

#[derive(Debug)]
struct Face {
    start: usize,
}

#[derive(Debug)]
struct Hedge {
    origin: usize,
    twin: usize,
    face: usize,
    next: usize,
    prev: usize,
}

struct FaceVerticesIterator<'a> {
    hedge_iterator: FaceHedgesIterator<'a>,
}

struct FaceHedgesIterator<'a> {
    dcel: &'a Dcel,
    face: usize,
    current: usize,
    done: bool,
}

struct ContourHedgesIterator<'a> {
    dcel: &'a Dcel,
    contour: usize,
    current: usize,
    done: bool,
}

impl Dcel {
    fn iter_face_vertices(&self, face: usize) -> FaceVerticesIterator {
        FaceVerticesIterator {
            hedge_iterator: self.iter_face_hedges(face),
        }
    }

    fn iter_face_hedges(&self, face: usize) -> FaceHedgesIterator {
        FaceHedgesIterator {
            dcel: self,
            face,
            current: self.faces[face].start,
            done: false,
        }
    }

    fn iter_contour_hedges(&self, contour: usize) -> ContourHedgesIterator {
        ContourHedgesIterator {
            dcel: self,
            contour,
            current: self.contours[contour].start,
            done: false,
        }
    }

    fn print_face_hedges(&self, face: usize) {
        for hedge in self.iter_face_hedges(face) {
            let origin = hedge.origin;
            let dest = self.hedges[hedge.next].origin;
            let [xo, yo] = self.vertices[origin].coords;
            let [xd, yd] = self.vertices[dest].coords;
            println!("({}, {}) -> ({}, {})", xo, yo, xd, yd);
        }
    }

    fn from_polygon_soup<const N: usize>(vertices: &[[f32; 2]], polygons: &[[usize; N]]) -> Self {
        let mut verts = Vec::with_capacity(vertices.len());
        for &coords in vertices {
            verts.push(Vertex {
                coords,
                hedge: usize::MAX,
            });
        }

        let mut faces = Vec::with_capacity(polygons.len());
        let mut hedges: Vec<Hedge> = Vec::new();
        let mut current_hedge_id = 0;
        let mut edges = HashMap::with_capacity(polygons.len() * N);
        for (face, polygon) in polygons.iter().enumerate() {
            faces.push(Face {
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
                    let hedge: &mut Hedge = &mut hedges[twin];
                    hedge.twin = current_hedge_id;
                    // NOTE: We have to do this little dance for some reason, this will not compile:
                    // hedges[twin].twin = current_hedge_id;
                    twin
                } else {
                    // Store half-edge id to set the twin id later on
                    edges.insert((vert, dest), current_hedge_id);
                    usize::MAX
                };

                hedges.push(Hedge {
                    origin: vert,
                    twin,
                    face,
                    next,
                    prev,
                });
                verts[vert].hedge = current_hedge_id;
                current_hedge_id += 1;
            }
        }

        // Find boundary half-edges
        // TODO: only works for one boundary, make it work when there are holes as well
        let mut contours = Vec::new();
        let first_outer_hedge_id = current_hedge_id;
        contours.push(Face {
            start: first_outer_hedge_id,
        });
        let start_id = *edges.values().min().unwrap();
        let mut inner_hedge_id = start_id;
        'outer: loop {
            // These are wrong for the first and last half-edge of the contour, but this is fixed
            // when exiting the loop
            let next = current_hedge_id + 1;
            let prev = current_hedge_id - 1;
            hedges.push(Hedge {
                origin: hedges[hedges[inner_hedge_id].next].origin,
                twin: inner_hedge_id,
                face: usize::MAX,
                next,
                prev,
            });
            hedges[inner_hedge_id].twin = current_hedge_id;
            current_hedge_id += 1;
            loop {
                // Iterate over the current origin's `umbrella` to find the next hedge
                inner_hedge_id = hedges[inner_hedge_id].prev;
                if inner_hedge_id == start_id {
                    // Means we are back at the start, time to fix the first and last half-edge
                    let last_outer_hedge_id = current_hedge_id - 1;
                    hedges[first_outer_hedge_id].prev = last_outer_hedge_id;
                    hedges[last_outer_hedge_id].next = first_outer_hedge_id;
                    break 'outer;
                }
                if hedges[inner_hedge_id].twin < usize::MAX {
                    inner_hedge_id = hedges[inner_hedge_id].twin;
                } else {
                    break;
                }
            }
        }

        Self {
            vertices: verts,
            faces,
            hedges,
            contours,
        }
    }

    fn get_face_coords(&self, face: usize) -> Vec<[f32; 2]> {
        self.iter_face_vertices(face)
            .map(|vert| vert.coords)
            .collect()
    }

    fn get_face_vertex_ids(&self, face: usize) -> Vec<usize> {
        self.iter_face_hedges(face)
            .map(|hedge| hedge.origin)
            .collect()
    }
}

impl<'a> Iterator for FaceVerticesIterator<'a> {
    type Item = &'a Vertex;

    fn next(&mut self) -> Option<Self::Item> {
        self.hedge_iterator
            .next()
            .map(|hedge| &self.hedge_iterator.dcel.vertices[hedge.origin])
    }
}

impl<'a> Iterator for FaceHedgesIterator<'a> {
    type Item = &'a Hedge;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let hedge = &self.dcel.hedges[self.current];
        self.current = hedge.next;
        let start = self.dcel.faces[self.face].start;
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
        let hedge = &self.dcel.hedges[self.current];
        self.current = hedge.next;
        let start = self.dcel.contours[self.contour].start;
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
        assert_eq!(vertex_ids, polygons[0]);
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
            assert_eq!(vertex_ids, polygons[face]);
        }

        // Check vertex coordinates
        let verts = dcel.get_face_coords(0);
        assert_eq!(verts, vec![[0., 0.], [1., 0.], [0., 1.]]);
        let verts = dcel.get_face_coords(1);
        assert_eq!(verts, vec![[1., 0.], [1., 1.], [0., 1.]]);

        // Check face ids
        for face in [0, 1] {
            for hedge in dcel.iter_face_hedges(face) {
                assert_eq!(hedge.face, face);
            }
        }

        // Check next and prev
        let nexts: Vec<_> = dcel.iter_face_hedges(0).map(|hedge| hedge.next).collect();
        assert_eq!(nexts, vec![1, 2, 0]);
        let prevs: Vec<_> = dcel.iter_face_hedges(0).map(|hedge| hedge.prev).collect();
        assert_eq!(prevs, vec![2, 0, 1]);
        let nexts: Vec<_> = dcel.iter_face_hedges(1).map(|hedge| hedge.next).collect();
        assert_eq!(nexts, vec![4, 5, 3]);
        let prevs: Vec<_> = dcel.iter_face_hedges(1).map(|hedge| hedge.prev).collect();
        assert_eq!(prevs, vec![5, 3, 4]);

        // Check contour
        let origins: Vec<_> = dcel
            .iter_contour_hedges(0)
            .map(|hedge| hedge.origin)
            .collect();
        assert_eq!(origins, vec![1, 0, 3, 2]);
    }

    #[test]
    fn create_four_quadrangles_from_polygon_soup() {
        let (vertices, polygons) = four_quadrangles();

        let dcel = Dcel::from_polygon_soup(&vertices, &polygons);

        // Check vertex ids
        for face in [0, 1] {
            let vertex_ids = dcel.get_face_vertex_ids(face);
            assert_eq!(vertex_ids, polygons[face]);
        }

        // Check vertex coordinates
        let verts = dcel.get_face_coords(0);
        assert_eq!(verts, vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]]);
        let verts = dcel.get_face_coords(1);
        assert_eq!(verts, vec![[1., 0.], [2., 0.], [2., 1.], [1., 1.]]);

        // Check face ids
        for face in [0, 1] {
            for hedge in dcel.iter_face_hedges(face) {
                assert_eq!(hedge.face, face);
            }
        }

        // Check next and prev
        let nexts: Vec<_> = dcel.iter_face_hedges(0).map(|hedge| hedge.next).collect();
        assert_eq!(nexts, vec![1, 2, 3, 0]);
        let prevs: Vec<_> = dcel.iter_face_hedges(0).map(|hedge| hedge.prev).collect();
        assert_eq!(prevs, vec![3, 0, 1, 2]);

        let nexts: Vec<_> = dcel.iter_face_hedges(1).map(|hedge| hedge.next).collect();
        assert_eq!(nexts, vec![5, 6, 7, 4]);
        let prevs: Vec<_> = dcel.iter_face_hedges(1).map(|hedge| hedge.prev).collect();
        assert_eq!(prevs, vec![7, 4, 5, 6]);

        let nexts: Vec<_> = dcel.iter_face_hedges(2).map(|hedge| hedge.next).collect();
        assert_eq!(nexts, vec![9, 10, 11, 8]);
        let prevs: Vec<_> = dcel.iter_face_hedges(2).map(|hedge| hedge.prev).collect();
        assert_eq!(prevs, vec![11, 8, 9, 10]);

        let nexts: Vec<_> = dcel.iter_face_hedges(3).map(|hedge| hedge.next).collect();
        assert_eq!(nexts, vec![13, 14, 15, 12]);
        let prevs: Vec<_> = dcel.iter_face_hedges(3).map(|hedge| hedge.prev).collect();
        assert_eq!(prevs, vec![15, 12, 13, 14]);

        // Check contour
        let origins: Vec<_> = dcel
            .iter_contour_hedges(0)
            .map(|hedge| hedge.origin)
            .collect();
        assert_eq!(origins, vec![1, 0, 3, 6, 7, 8, 5, 2]);
    }

    #[test]
    fn twins() {
        // Triangles
        let (vertices, polygons) = two_triangles();

        let dcel = Dcel::from_polygon_soup(&vertices, &polygons);

        // Check individual twins
        let twins: Vec<_> = dcel.hedges.iter().map(|hedge| hedge.twin).collect();
        assert_eq!(twins, vec![6, 5, 7, 9, 8, 1, 0, 2, 4, 3]);

        // Check that the twins' origin is the same as the next hedge's origin
        for (i, _) in dcel.faces.iter().enumerate() {
            for hedge in dcel.iter_face_hedges(i) {
                assert_eq!(
                    dcel.hedges[hedge.twin].origin,
                    dcel.hedges[hedge.next].origin
                );
            }
        }

        // Quadrangles
        let (vertices, polygons) = four_quadrangles();

        let dcel = Dcel::from_polygon_soup(&vertices, &polygons);

        // Check individual twins
        let twins: Vec<_> = dcel.hedges.iter().map(|hedge| hedge.twin).collect();
        assert_eq!(
            twins,
            vec![
                16, 7, 8, 17, 23, 22, 12, 1, 2, 15, 19, 18, 6, 21, 20, 9, 0, 3, 11, 10, 14, 13, 5,
                4
            ]
        );

        // Check that the twins' origin is the same as the next hedge's origin
        for (i, _) in dcel.faces.iter().enumerate() {
            for hedge in dcel.iter_face_hedges(i) {
                assert_eq!(
                    dcel.hedges[hedge.twin].origin,
                    dcel.hedges[hedge.next].origin
                );
            }
        }
    }
}
