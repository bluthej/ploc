mod dcel;

use dcel::{Dcel, FaceId, Hedge};
use indextree::Arena;

struct TrapMap {
    dcel: Dcel,
    tree: Arena<Node>,
    root: indextree::NodeId,
}

// TODO: add the necessary data
enum Node {
    X,
    Y,
    Trap(Trapezoid),
}

struct Trapezoid {
    top: Hedge,
    bottom: Hedge,
}

impl TrapMap {
    fn new(dcel: Dcel) -> Self {
        let mut tree = Arena::new();
        let top = Hedge::default();
        let bottom = Hedge::default();
        let root = tree.new_node(Node::Trap(Trapezoid { top, bottom }));
        Self { dcel, tree, root }
    }

    fn count_traps(&self) -> usize {
        self.tree.count()
    }

    fn find_face(&self, point: &[f64; 2]) -> Option<FaceId> {
        self.find_trapezoid(point).and_then(|trap| trap.bottom.face)
    }

    fn find_trapezoid(&self, _point: &[f64; 2]) -> Option<&Trapezoid> {
        self.tree.get(self.root).and_then(|node| match node.get() {
            Node::Trap(trapezoid) => Some(trapezoid),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialize_empty_trapezoidal_map() {
        let dcel = Dcel::new();

        let trap_map = TrapMap::new(dcel);

        assert_eq!(trap_map.count_traps(), 1);
    }

    #[test]
    fn find_trap_in_empty_trapezoidal_map() {
        let dcel = Dcel::new();
        let trap_map = TrapMap::new(dcel);
        assert_eq!(trap_map.count_traps(), 1);

        let point = [0., 0.];
        let trap = trap_map.find_trapezoid(&point);
    }

    #[test]
    fn find_face_in_empty_trapezoidal_map() {
        let dcel = Dcel::new();
        let trap_map = TrapMap::new(dcel);

        let point = [0., 0.];

        assert_eq!(trap_map.find_face(&point), None);
    }
}
