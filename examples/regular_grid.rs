use ploc::{Mesh, PointLocator, QuadTree, TrapMap};
use rand::prelude::*;

fn main() {
    let (xmin, xmax) = (0., 1.);
    let (ymin, ymax) = (0., 1.);
    let n = 500;

    // Create trapezoidal maps
    let mesh = Mesh::grid(xmin, xmax, ymin, ymax, n, n).unwrap();
    let trap_map = TrapMap::from_mesh(mesh.clone());

    let quad_tree = QuadTree::<9>::new(&mesh);

    let mut rng = rand::thread_rng();
    let query: Vec<_> = (0..420_000)
        .map(|_| [rng.gen::<f64>() * xmax, rng.gen::<f64>() * ymax])
        .collect();

    let res = trap_map.locate_many(&query);
    // println!("Trapezoidal map: {:?}", res);

    let qt_res = quad_tree.locate_many(&query);
    // println!("Quad tree: {:?}", res);
    assert_eq!(res, qt_res);
}
