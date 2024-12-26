use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::izip;
use ploc::{Mesh, PointLocator, QuadTree, TrapMap};
use rand::prelude::*;

fn bench(c: &mut Criterion) {
    // Discretization parameter
    let ns: Vec<_> = (5..=405).step_by(40).collect();

    let (xmin, xmax) = (0., 1.);
    let (ymin, ymax) = (0., 1.);

    // Create trapezoidal maps
    let trap_maps: Vec<TrapMap> = ns
        .iter()
        .map(|&n| {
            let mesh = Mesh::grid(xmin, xmax, ymin, ymax, n, n).unwrap();
            TrapMap::from_mesh(mesh)
        })
        .collect();

    // Create quadtrees
    let meshes: Vec<_> = ns
        .iter()
        .map(|&n| Mesh::grid(xmin, xmax, ymin, ymax, n, n).unwrap())
        .collect();
    let quad_trees: Vec<_> = meshes.iter().map(QuadTree::<9>::new).collect();

    // Random number generator
    let mut rng = rand::thread_rng();

    let mut group = c.benchmark_group("Trapezoidal map VS Quadtree");
    for (n, trap_map, quad_tree) in izip!(ns, trap_maps, quad_trees) {
        let query: Vec<_> = (0..420)
            .map(|_| [rng.gen::<f64>() * xmax, rng.gen::<f64>() * ymax])
            .collect();
        group.bench_with_input(BenchmarkId::new("TrapMap", n), &query, |b, q| {
            b.iter(|| {
                trap_map.locate_many(q);
            })
        });
        group.bench_with_input(BenchmarkId::new("QuadTree slow", n), &query, |b, q| {
            b.iter(|| {
                quad_tree.locate_many(q);
            })
        });
        group.bench_with_input(BenchmarkId::new("QuadTree", n), &query, |b, q| {
            b.iter(|| {
                quad_tree.fast_locate_many(q);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
