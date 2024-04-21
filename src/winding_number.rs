use itertools::Itertools;

/// A point of the 2D plane.
#[derive(Debug, Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

/// Positioning of a `Point` with respect to a line.
#[derive(Debug, PartialEq)]
enum Positioning {
    Left,
    On,
    Right,
}

impl Point {
    /// Test if a point is Left|On|Right of an infinite 2D line defined by two points.
    fn position(&self, p1: &Point, p2: &Point) -> Positioning {
        let p0 = self;
        match ((p2.x - p1.x) * (p0.y - p1.y) - (p0.x - p1.x) * (p2.y - p1.y)).total_cmp(&0.) {
            std::cmp::Ordering::Greater => Positioning::Left,
            std::cmp::Ordering::Less => Positioning::Right,
            std::cmp::Ordering::Equal => Positioning::On,
        }
    }

    /// Compute the winding number for a [`Point`] in a polygon (defined by a slice of [`Point`]s).
    ///
    /// This number can be:
    /// - `0` if the [`Point`] is not inside the polygon
    /// - `> 0` if the [`Point`] is inside the polygon and the polygon "winds" at least once around the [`Point`] counter-clockwise
    /// - `< 0` if the [`Point`] is inside the polygon and the polygon "winds" at least once around the [`Point`] clockwise
    ///
    /// For more information, see <https://web.archive.org/web/20130126163405/http://geomalgorithms.com/a03-_inclusion.html>.
    fn wn(&self, poly: &[Point]) -> isize {
        let mut wn = 0;
        for (a, b) in poly.iter().circular_tuple_windows() {
            if a.y <= self.y {
                // `a` is below self
                if b.y > self.y {
                    // an upward crossing
                    if matches!(self.position(a, b), Positioning::Left) {
                        wn += 1;
                    }
                }
            } else {
                // `a` is above self
                if b.y <= self.y {
                    // a downward crossing
                    if matches!(self.position(a, b), Positioning::Right) {
                        wn -= 1;
                    }
                }
            }
        }
        wn
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positioning() {
        let p1 = Point { x: 0., y: 0. };
        let p2 = Point { x: 1., y: 1. };

        assert_eq!(
            Point { x: 0., y: 0.5 }.position(&p1, &p2),
            Positioning::Left
        );
        assert_eq!(Point { x: 0.5, y: 0.5 }.position(&p1, &p2), Positioning::On);
        assert_eq!(
            Point { x: 1., y: 0.5 }.position(&p1, &p2),
            Positioning::Right
        );
    }

    #[test]
    fn winding_number_square() {
        let poly: Vec<_> = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
            .iter()
            .map(|&[x, y]| Point { x, y })
            .collect();

        assert_eq!(Point { x: 0.5, y: 0.5 }.wn(&poly), 1);
        assert_eq!(Point { x: 1.5, y: 0.5 }.wn(&poly), 0);
        assert_eq!(Point { x: 0.5, y: 1.5 }.wn(&poly), 0);
    }

    #[test]
    fn winding_number_self_overlapping_polygon() {
        let poly: Vec<_> = [
            [0., 0.],
            [1., 0.],
            [1., 0.8],
            [0.2, 0.8],
            [0.2, 0.5],
            [0.8, 0.5],
            [0.8, 0.2],
            [0.5, 0.2],
            [0.5, 1.],
            [0., 1.],
        ]
        .iter()
        .map(|&[x, y]| Point { x, y })
        .collect();

        assert_eq!(Point { x: 0.1, y: 0.1 }.wn(&poly), 1);
        assert_eq!(Point { x: 0.6, y: 0.3 }.wn(&poly), 0);
        assert_eq!(Point { x: 0.4, y: 0.6 }.wn(&poly), 2);
    }
}
