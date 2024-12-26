use std::collections::HashMap;

use crate::{winding_number::Point, Mesh};

pub struct QuadTree<'a, const C: usize> {
    arena: Vec<Node>,
    mesh: &'a Mesh,
    // Hash-map for fast point location. Key is: (level, i, j). Value is a node id.
    map: HashMap<(usize, usize, usize), usize>,
    depth: usize,
}

struct Node {
    xc: f64,
    yc: f64,
    h: f64,
    // Children
    sw: Option<usize>,
    se: Option<usize>,
    nw: Option<usize>,
    ne: Option<usize>,
    // Conflict list
    conflicts: Vec<usize>,
}

impl<'a, const C: usize> QuadTree<'a, C> {
    pub fn new(mesh: &'a Mesh) -> Self {
        let root = Node {
            xc: 0.,
            yc: 0.,
            h: 1.,
            sw: None,
            se: None,
            nw: None,
            ne: None,
            conflicts: (0..mesh.cell_count()).collect(),
        };
        let arena = vec![root];
        let map = HashMap::new();
        let mut quad_tree = Self {
            arena,
            mesh,
            map,
            depth: 0,
        };
        quad_tree.split(0);
        quad_tree.build_map();
        quad_tree.depth = quad_tree
            .map
            .keys()
            .map(|&(level, ..)| level)
            .max()
            .expect("The map should never be empty.");
        quad_tree
    }

    fn split(&mut self, id: usize) {
        let current = &self.arena[id];
        let xc = current.xc;
        let yc = current.yc;
        let h = current.h;
        let h2 = h / 2.;

        // South-west
        let con = self.get_conflicts(&self.arena[id].conflicts, xc, yc, h2);
        if !con.is_empty() {
            let conflict_count = con.len();
            let child_id = self.arena.len();
            let new_node = Node {
                xc,
                yc,
                h: h2,
                sw: None,
                se: None,
                nw: None,
                ne: None,
                conflicts: con,
            };
            self.arena.push(new_node);
            self.arena[id].sw = Some(child_id);
            if conflict_count > C {
                self.split(child_id);
            }
        }

        // South-east
        let con = self.get_conflicts(&self.arena[id].conflicts, xc + h2, yc, h2);
        if !con.is_empty() {
            let conflict_count = con.len();
            let child_id = self.arena.len();
            let new_node = Node {
                xc: xc + h2,
                yc,
                h: h2,
                sw: None,
                se: None,
                nw: None,
                ne: None,
                conflicts: con,
            };
            self.arena.push(new_node);
            self.arena[id].se = Some(child_id);
            if conflict_count > C {
                self.split(child_id);
            }
        }

        // North-west
        let con = self.get_conflicts(&self.arena[id].conflicts, xc, yc + h2, h2);
        if !con.is_empty() {
            let conflict_count = con.len();
            let child_id = self.arena.len();
            let new_node = Node {
                xc,
                yc: yc + h2,
                h: h2,
                sw: None,
                se: None,
                nw: None,
                ne: None,
                conflicts: con,
            };
            self.arena.push(new_node);
            self.arena[id].nw = Some(child_id);
            if conflict_count > C {
                self.split(child_id);
            }
        }

        // North-east
        let con = self.get_conflicts(&self.arena[id].conflicts, xc + h2, yc + h2, h2);
        if !con.is_empty() {
            let conflict_count = con.len();
            let child_id = self.arena.len();
            let new_node = Node {
                xc: xc + h2,
                yc: yc + h2,
                h: h2,
                sw: None,
                se: None,
                nw: None,
                ne: None,
                conflicts: con,
            };
            self.arena.push(new_node);
            self.arena[id].ne = Some(child_id);
            if conflict_count > C {
                self.split(child_id);
            }
        }
    }

    fn get_conflicts(&self, conflicts: &[usize], xc: f64, yc: f64, h: f64) -> Vec<usize> {
        conflicts
            .iter()
            .filter(|&&cell_id| {
                self.mesh
                    .cell_vertices(cell_id)
                    .any(|&[x, y]| x >= xc && x <= xc + h && y >= yc && y <= yc + h)
            })
            .cloned()
            .collect()
    }

    fn locate(&self, &[x, y]: &[f64; 2]) -> Option<usize> {
        let root = &self.arena[0];
        if x < root.xc || x > root.xc + root.h || y < root.yc || y > root.yc + root.h {
            return None;
        }
        let mut id = 0;
        while let Some(child_id) = self.child(id, &[x, y]) {
            id = child_id;
        }
        let point = Point::from([x, y]);
        self.arena[id]
            .conflicts
            .iter()
            .find(|&&cell_id| point.is_inside(self.mesh.cell_vertices(cell_id).copied()))
            .copied()
    }

    fn build_map(&mut self) {
        self.map.reserve(self.arena.len());
        for (id, node) in self.arena.iter().enumerate() {
            let level = (1. / node.h).log2() as usize;
            let i = (node.xc / node.h).floor() as usize;
            let j = (node.yc / node.h).floor() as usize;
            self.map.insert((level, i, j), id);
        }
    }

    pub fn fast_locate(&self, q: &[f64; 2]) -> Option<usize> {
        let &[x, y] = q;
        let root = &self.arena[0];
        if x < root.xc || x > root.xc + root.h || y < root.yc || y > root.yc + root.h {
            return None;
        }
        let id = self.bisect(q, 0, self.depth);
        self.arena[id]
            .conflicts
            .iter()
            .find(|&&cell_id| {
                Point::from([x, y]).is_inside(self.mesh.cell_vertices(cell_id).copied())
            })
            .copied()
    }

    fn bisect(&self, q: &[f64; 2], l: usize, h: usize) -> usize {
        let m = ((l + h) as f64 / 2.).floor() as usize;
        let Some(v) = self.get_node(q, m) else {
            return self.bisect(q, l, m - 1);
        };
        if self.child(v, q).is_none() {
            return v;
        };
        self.bisect(q, m + 1, h)
    }

    fn get_node(&self, q: &[f64; 2], level: usize) -> Option<usize> {
        let h = 1. / (2usize.pow(level as u32) as f64);
        let i = (q[0] / h).floor() as usize;
        let j = (q[1] / h).floor() as usize;
        self.map.get(&(level, i, j)).cloned()
    }

    fn child(&self, id: usize, q: &[f64; 2]) -> Option<usize> {
        let x = q[0];
        let y = q[1];
        let xc = self.arena[id].xc;
        let yc = self.arena[id].yc;
        let h2 = self.arena[id].h / 2.;
        #[allow(clippy::collapsible_else_if)]
        if y <= yc + h2 {
            if x <= xc + h2 {
                self.arena[id].sw
            } else {
                self.arena[id].se
            }
        } else {
            if x <= xc + h2 {
                self.arena[id].nw
            } else {
                self.arena[id].ne
            }
        }
    }

    pub fn locate_many(&self, query: &[[f64; 2]]) -> Vec<Option<usize>> {
        query.iter().map(|q| self.locate(q)).collect()
    }

    pub fn fast_locate_many(&self, query: &[[f64; 2]]) -> Vec<Option<usize>> {
        query.iter().map(|q| self.fast_locate(q)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stuff() {
        let mesh = Mesh::grid(0., 1., 0., 1., 10, 10).unwrap();
        let quad_tree = QuadTree::<9>::new(&mesh);
        for j in 1..10 {
            let y = 0.05 + (j as f64) * 0.1;
            for i in 1..10 {
                let x = 0.05 + (i as f64) * 0.1;
                assert_eq!(quad_tree.locate(&[x, y]), Some(j * 10 + i));
            }
        }
    }

    #[test]
    fn fast_stuff() {
        let mesh = Mesh::grid(0., 1., 0., 1., 10, 10).unwrap();
        let quad_tree = QuadTree::<9>::new(&mesh);
        for j in 1..10 {
            let y = 0.05 + (j as f64) * 0.1;
            for i in 1..10 {
                let x = 0.05 + (i as f64) * 0.1;
                assert_eq!(quad_tree.locate(&[x, y]), Some(j * 10 + i));
            }
        }
    }
}
