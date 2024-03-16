mod dcel;

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

    fn count_traps(&self) -> usize {
        self.tree.count()
    }

    fn find_face(&self, point: &[f64; 2]) -> Option<FaceId> {
        self.find_trapezoid(point)
            .and_then(|trap| self.dcel.get_hedge(trap.bottom).face)
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
        let trap_map = TrapMap::new();

        assert_eq!(trap_map.count_traps(), 1);
    }

    #[test]
    fn find_trap_in_empty_trapezoidal_map() {
        let trap_map = TrapMap::new();
        assert_eq!(trap_map.count_traps(), 1);

        let point = [0., 0.];
        let trap = trap_map.find_trapezoid(&point);

        assert!(trap.is_some());
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
}
