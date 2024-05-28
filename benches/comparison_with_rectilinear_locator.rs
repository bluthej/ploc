use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::{izip, Itertools};
use ploc::{Mesh, PointLocator, RectilinearLocator, TrapMap};
use rand::prelude::*;

fn bench_rect(c: &mut Criterion) {
    // Discretization parameter
    let ns: Vec<_> = (5..=105).step_by(20).collect();

    let (xmin, xmax) = (0., 10.);
    let (ymin, ymax) = (0., 10.);

    // Create trapezoidal maps
    let trap_maps: Vec<TrapMap> = ns
        .iter()
        .map(|&n| {
            let mesh = Mesh::grid(xmin, xmax, ymin, ymax, n, n).unwrap();
            TrapMap::from_mesh(mesh)
        })
        .collect();

    // Create rectilinear locator
    let rect_locators: Vec<_> = ns
        .iter()
        .map(|&n| {
            let dx = (xmax - xmin) / n as f64;
            let x = (0..=n).map(|n| xmin + n as f64 * dx).collect_vec();
            let dy = (ymax - ymin) / n as f64;
            let y = (0..=n).map(|n| ymin + n as f64 * dy).collect_vec();
            RectilinearLocator::new(x, y).unwrap()
        })
        .collect();

    // Random number generator
    let mut rng = rand::thread_rng();

    let mut group = c.benchmark_group("Rectilinear mesh");
    for (n, trap_map, rect_locator) in izip!(ns, trap_maps, rect_locators) {
        let query: Vec<_> = (0..420)
            .map(|_| [rng.gen::<f64>() * xmax, rng.gen::<f64>() * ymax])
            .collect();
        group.bench_with_input(BenchmarkId::new("TrapMap", n), &query, |b, q| {
            b.iter(|| {
                trap_map.locate_many(q);
            })
        });
        group.bench_with_input(BenchmarkId::new("RectilinearLocator", n), &query, |b, q| {
            b.iter(|| {
                rect_locator.locate_many(q);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_rect);
criterion_main!(benches);
