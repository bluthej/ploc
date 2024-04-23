#![allow(dead_code)]

mod dcel;
mod winding_number;

use dcel::{Dcel, FaceId, Hedge, HedgeId};
use indextree::Arena;

struct TrapMap {
    dcel: Dcel,
    tree: Arena<Node>,
    root: indextree::NodeId,
    bbox: BoundingBox,
}

// TODO: add the necessary data
enum Node {
    X,
    Y,
    Trap(Trapezoid),
}

// TODO: add leftp and rightp
#[derive(Clone)]
struct Trapezoid {
    top: HedgeId,
    bottom: HedgeId,
}

struct BoundingBox {
    xmin: f32,
    xmax: f32,
    ymin: f32,
    ymax: f32,
}

impl BoundingBox {
    fn new() -> Self {
        Self::default()
    }

    fn from_bounds(xmin: f32, xmax: f32, ymin: f32, ymax: f32) -> Self {
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
        Self::with_dcel(dcel)
    }

    fn from_polygon_soup<const N: usize>(vertices: &[[f32; 2]], polygons: &[[usize; N]]) -> Self {
        let dcel = Dcel::from_polygon_soup(vertices, polygons);
        Self::with_dcel(dcel)
    }

    fn with_dcel(dcel: Dcel) -> Self {
        let mut tree = Arena::new();

        let mut dcel = dcel;
        let top = dcel.add_hedge(Hedge::new());
        let bottom = dcel.add_hedge(Hedge::new());

        let root = tree.new_node(Node::Trap(Trapezoid { top, bottom }));

        let [xmin, xmax, ymin, ymax] = dcel.get_bounds();
        let bbox = BoundingBox::from_bounds(xmin, xmax, ymin, ymax);

        Self {
            dcel,
            tree,
            root,
            bbox,
        }
    }

    fn count_x_nodes(&self) -> usize {
        let mut count = 0;
        for node in self.tree.iter() {
            if matches!(node.get(), Node::X) {
                count += 1;
            }
        }
        count
    }

    fn count_y_nodes(&self) -> usize {
        let mut count = 0;
        for node in self.tree.iter() {
            if matches!(node.get(), Node::Y) {
                count += 1;
            }
        }
        count
    }

    fn count_traps(&self) -> usize {
        let mut count = 0;
        for node in self.tree.iter() {
            if matches!(node.get(), Node::Trap(..)) {
                count += 1;
            }
        }
        count
    }

    fn count_nodes(&self) -> (usize, usize, usize) {
        let mut trap_count = 0;
        let mut x_node_count = 0;
        let mut y_node_count = 0;
        for node in self.tree.iter() {
            match node.get() {
                Node::X => x_node_count += 1,
                Node::Y => y_node_count += 1,
                Node::Trap(_) => trap_count += 1,
            }
        }
        (x_node_count, y_node_count, trap_count)
    }

    fn print_stats(&self) {
        let (x_node_count, y_node_count, trap_count) = self.count_nodes();
        println!(
            "Trapezoidal map counts:\n\t{} X node(s)\n\t{} Y node(s)\n\t{} trapezoid(s)",
            x_node_count, y_node_count, trap_count
        );
    }

    fn find_face(&self, point: &[f32; 2]) -> Option<FaceId> {
        let (_, trap) = self.find_trapezoid(point);
        self.dcel.get_hedge(trap.bottom).face
    }

    fn find_trapezoid(&self, _point: &[f32; 2]) -> (indextree::NodeId, &Trapezoid) {
        let node_id = self.root;
        loop {
            match self
                .tree
                .get(node_id)
                .expect("Node ids should always exist")
                .get()
            {
                Node::Trap(trapezoid) => return (node_id, trapezoid),
                _ => todo!("Handle X and Y nodes later"),
            }
        }
    }

    fn add_edge(&mut self, hedge_id: HedgeId) {
        self.print_stats();
        let hedge = self.dcel.get_hedge(hedge_id);
        let p = self.dcel.get_vertex(hedge.origin);
        let (old_nid, _old_trap) = self.find_trapezoid(&p.coords);

        let p_nid = self.tree.new_node(Node::X);
        old_nid.prepend(p_nid, &mut self.tree);
        let q_nid = p_nid.append_value(Node::X, &mut self.tree);

        let s_nid = q_nid.append_value(Node::Y, &mut self.tree);
        let b_trap = Trapezoid {
            top: HedgeId(0),
            bottom: HedgeId(0),
        };
        let c_trap = b_trap.clone();
        let d_trap = b_trap.clone();
        let _b_nid = q_nid.append_value(Node::Trap(b_trap), &mut self.tree);

        let _c_nid = s_nid.append_value(Node::Trap(c_trap), &mut self.tree);
        let _d_nid = s_nid.append_value(Node::Trap(d_trap), &mut self.tree);
        self.print_stats();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialize_empty_trapezoidal_map() {
        let trap_map = TrapMap::new();

        assert_eq!(trap_map.count_traps(), 1);
    }

    #[test]
    fn find_trap_in_empty_trapezoidal_map() {
        let trap_map = TrapMap::new();
        assert_eq!(trap_map.count_traps(), 1);
        assert_eq!(trap_map.count_x_nodes(), 0);
        assert_eq!(trap_map.count_y_nodes(), 0);

        let point = [0., 0.];
        let _trap = trap_map.find_trapezoid(&point);
    }

    #[test]
    fn find_face_in_empty_trapezoidal_map() {
        let trap_map = TrapMap::new();

        let point = [0., 0.];

        assert_eq!(trap_map.find_face(&point), None);
    }

    #[test]
    fn bounding_box() {
        let vertices = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
        let polygons = vec![[0, 1, 2, 3]];
        let trap_map = TrapMap::from_polygon_soup(&vertices, &polygons);

        let bbox = trap_map.bbox;

        assert!(bbox.xmin < 0.);
        assert!(bbox.xmax > 1.);
        assert!(bbox.ymin < 0.);
        assert!(bbox.ymax > 1.);
    }

    #[test]
    fn add_first_edge() {
        let vertices = vec![[0., 0.], [1., 0.], [0.5, 0.5]];
        let polygons = vec![[0, 1, 2]];
        let dcel = Dcel::from_polygon_soup(&vertices, &polygons);
        let mut trap_map = TrapMap::with_dcel(dcel);

        trap_map.add_edge(HedgeId(0));

        assert_eq!(trap_map.count_traps(), 4);
        assert_eq!(trap_map.count_x_nodes(), 2);
        assert_eq!(trap_map.count_y_nodes(), 1);
    }
}
