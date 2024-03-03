use std::fmt::Display;

#[derive(Debug)]
struct Dcel {
    vertices: Vec<Vertex>,
    faces: Vec<Face>,
    hedges: Vec<Hedge>,
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
    dcel: &'a Dcel,
    face: usize,
    current: usize,
    done: bool,
}

struct FaceHedgesIterator<'a> {
    dcel: &'a Dcel,
    face: usize,
    current: usize,
    done: bool,
}

impl Dcel {
    fn iter_face_vertices(&self, face: usize) -> FaceVerticesIterator {
        FaceVerticesIterator {
            dcel: self,
            face,
            current: self.faces[face].start,
            done: false,
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

    fn print_face_hedges(&self, face: usize) {
        for hedge in self.iter_face_hedges(face) {
            let origin = hedge.origin;
            let dest = self.hedges[hedge.next].origin;
            let [xo, yo] = self.vertices[origin].coords;
            let [xd, yd] = self.vertices[dest].coords;
            println!("({}, {}) -> ({}, {})", xo, yo, xd, yd);
        }
    }

    fn from_polygon_soup(vertices: &[[f32; 2]], polygons: &[[usize; 3]]) -> Self {
        let mut verts = Vec::with_capacity(vertices.len());
        for &coords in vertices {
            verts.push(Vertex {
                coords,
                hedge: usize::MAX,
            });
        }

        let mut faces = Vec::with_capacity(polygons.len());
        let mut hedges = Vec::new();
        let mut current_hedge_id = 0;
        for (face, polygon) in polygons.iter().enumerate() {
            faces.push(Face {
                start: current_hedge_id,
            });

            let n_verts = polygon.len();
            for (iloc, &vert) in polygon.iter().enumerate() {
                let (prev, next) = match iloc {
                    0 => (current_hedge_id + n_verts - 1, current_hedge_id + 1),
                    ivert if ivert == n_verts - 1 => {
                        (current_hedge_id - 1, current_hedge_id + 1 - n_verts)
                    }
                    _ => (current_hedge_id - 1, current_hedge_id + 1),
                };
                hedges.push(Hedge {
                    origin: vert,
                    twin: usize::MAX,
                    face,
                    next,
                    prev,
                });
                verts[vert].hedge = current_hedge_id;
                current_hedge_id += 1;
            }
        }

        Self {
            vertices: verts,
            faces,
            hedges,
        }
    }
}

impl<'a> Iterator for FaceVerticesIterator<'a> {
    type Item = &'a Vertex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let hedge = &self.dcel.hedges[self.current];
        let vertex_id = hedge.origin;
        self.current = hedge.next;
        let start = self.dcel.faces[self.face].start;
        if self.current == start {
            self.done = true;
        }
        Some(&self.dcel.vertices[vertex_id])
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_one_triangle_dcel_from_polygon_soup() {
        let vertices = vec![[0., 0.], [1., 0.], [0., 1.]];
        let polygons = vec![[0, 1, 2]];

        let dcel = Dcel::from_polygon_soup(&vertices, &polygons);

        let verts: Vec<_> = dcel.iter_face_vertices(0).map(|vert| vert.coords).collect();
        assert_eq!(verts, vertices);
        let hedges: Vec<_> = dcel.iter_face_hedges(0).map(|hedge| hedge.origin).collect();
        assert_eq!(hedges, &polygons[0]);
    }

    #[test]
    fn create_two_triangle_dcel_from_polygon_soup() {
        let vertices = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
        let polygons = vec![[0, 1, 3], [1, 2, 3]];

        let dcel = Dcel::from_polygon_soup(&vertices, &polygons);

        // Check vertices
        let verts: Vec<_> = dcel.iter_face_vertices(0).map(|vert| vert.coords).collect();
        assert_eq!(verts, vec![[0., 0.], [1., 0.], [0., 1.]]);
        let verts: Vec<_> = dcel.iter_face_vertices(1).map(|vert| vert.coords).collect();
        assert_eq!(verts, vec![[1., 0.], [1., 1.], [0., 1.]]);

        // Check half-edges
        let hedges: Vec<_> = dcel.iter_face_hedges(0).map(|hedge| hedge.origin).collect();
        assert_eq!(hedges, vec![0, 1, 3]);
        let hedges: Vec<_> = dcel.iter_face_hedges(1).map(|hedge| hedge.origin).collect();
        assert_eq!(hedges, vec![1, 2, 3]);

        // Check face ids
        for face in [0, 1] {
            for hedge in dcel.iter_face_hedges(face) {
                assert_eq!(hedge.face, face);
            }
        }

        // Check next and prev
        let nexts: Vec<_> = dcel.iter_face_hedges(0).map(|hedge| hedge.next).collect();
        assert_eq!(nexts, vec![1, 2, 0]);
        let nexts: Vec<_> = dcel.iter_face_hedges(1).map(|hedge| hedge.next).collect();
        assert_eq!(nexts, vec![4, 5, 3]);
        let prevs: Vec<_> = dcel.iter_face_hedges(0).map(|hedge| hedge.prev).collect();
        assert_eq!(prevs, vec![2, 0, 1]);
        let prevs: Vec<_> = dcel.iter_face_hedges(1).map(|hedge| hedge.prev).collect();
        assert_eq!(prevs, vec![5, 3, 4]);
    }
}
