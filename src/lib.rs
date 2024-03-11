mod dcel;

use dcel::{Dcel, FaceId};
use indextree::Arena;

struct TrapMap {
    dcel: Dcel,
    tree: Arena<Node>,
}

// TODO: add the necessary data
enum Node {
    X,
    Y,
    Trap,
}

impl TrapMap {
    fn new(dcel: Dcel) -> Self {
        let mut tree = Arena::new();
        tree.new_node(Node::Trap);
        Self { dcel, tree }
    }

    fn count_traps(&self) -> usize {
        self.tree.count()
    }

    fn find_face(&self, point: &[f64; 2]) -> Option<FaceId> {
        // TODO: actually implement it
        None
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
    fn find_point_in_empty_trapezoidal_map() {
        let dcel = Dcel::new();
        let trap_map = TrapMap::new(dcel);

        let point = [0., 0.];

        assert_eq!(trap_map.find_face(&point), None);
    }
}
