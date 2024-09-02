use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ploc::{Mesh, PointLocator, TrapMap};
use rand::prelude::*;

pub fn create_trap_map(c: &mut Criterion) {
    let (xmin, xmax) = (0., 10.);
    let (ymin, ymax) = (0., 10.);

    for n in [5, 50, 200] {
        // Create trapezoidal maps
        let mesh = Mesh::grid(xmin, xmax, ymin, ymax, n, n).unwrap();

        c.bench_with_input(
            BenchmarkId::new("Create trapezoidal maps", n),
            &mesh,
            |b, m| {
                b.iter(|| TrapMap::from_mesh(m.clone()));
            },
        );
    }
}

pub fn locate_points(c: &mut Criterion) {
    let (xmin, xmax) = (0., 10.);
    let (ymin, ymax) = (0., 10.);

    for n in [5, 50, 200] {
        // Create trapezoidal maps
        let mesh = Mesh::grid(xmin, xmax, ymin, ymax, n, n).unwrap();
        let trap_map = TrapMap::from_mesh(mesh);

        let mut rng = rand::thread_rng();
        let query: Vec<_> = (0..42_000)
            .map(|_| [rng.gen::<f64>() * xmax, rng.gen::<f64>() * ymax])
            .collect();

        c.bench_with_input(BenchmarkId::new("Locate points", n), &query, |b, q| {
            b.iter(|| trap_map.locate_many(q));
        });
    }
}

criterion_group!(benches, create_trap_map, locate_points);
criterion_main!(benches);
