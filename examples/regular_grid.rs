use ploc::{Mesh, NewTrapMap, PointLocator};
use rand::prelude::*;

fn main() {
    let (xmin, xmax) = (0., 10.);
    let (ymin, ymax) = (0., 10.);
    let n = 200;

    // Create trapezoidal maps
    let mesh = Mesh::grid(xmin, xmax, ymin, ymax, n, n).unwrap();
    let trap_map = NewTrapMap::from_mesh(mesh);

    let mut rng = rand::thread_rng();
    let query: Vec<_> = (0..420_000)
        .map(|_| [rng.gen::<f64>() * xmax, rng.gen::<f64>() * ymax])
        .collect();

    let _res = trap_map.locate_many(&query);
    // println!("{:?}", res);
}
