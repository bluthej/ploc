use itertools::Itertools;

/// A point of the 2D plane.
#[derive(Debug, Clone, Copy)]
pub struct Point {
    x: f64,
    y: f64,
}

impl From<&Point> for [f64; 2] {
    fn from(val: &Point) -> Self {
        [val.x, val.y]
    }
}

impl From<Point> for [f64; 2] {
    fn from(val: Point) -> Self {
        (&val).into()
    }
}

impl From<&[f64; 2]> for Point {
    fn from(value: &[f64; 2]) -> Self {
        Self {
            x: value[0],
            y: value[1],
        }
    }
}

impl From<[f64; 2]> for Point {
    fn from(value: [f64; 2]) -> Self {
        Self::from(&value)
    }
}

/// Positioning of a `Point` with respect to a line.
#[derive(Debug, PartialEq)]
pub(crate) enum Positioning {
    Left,
    On,
    Right,
}

impl Point {
    /// Tests if a point is Left|On|Right of an infinite 2D line defined by two points.
    pub(crate) fn position<T>(&self, p1: T, p2: T) -> Positioning
    where
        T: Into<[f64; 2]>,
    {
        let Self { x: x0, y: y0 } = self;
        let [x1, y1] = p1.into();
        let [x2, y2] = p2.into();
        match ((x2 - x1) * (y0 - y1) - (x0 - x1) * (y2 - y1)).total_cmp(&0.) {
            std::cmp::Ordering::Greater => Positioning::Left,
            std::cmp::Ordering::Less => Positioning::Right,
            std::cmp::Ordering::Equal => Positioning::On,
        }
    }

    /// Computes the winding number for a [`Point`] in a polygon (defined by a slice of [`Point`]s).
    ///
    /// This number can be:
    /// - `0` if the [`Point`] is not inside the polygon
    /// - `> 0` if the [`Point`] is inside the polygon and the polygon "winds" at least once around the [`Point`] counter-clockwise
    /// - `< 0` if the [`Point`] is inside the polygon and the polygon "winds" at least once around the [`Point`] clockwise
    ///
    /// For more information, see <https://web.archive.org/web/20130126163405/http://geomalgorithms.com/a03-_inclusion.html>.
    pub fn wn<I>(&self, poly: I) -> isize
    where
        I: IntoIterator,
        <I as IntoIterator>::IntoIter: Clone,
        <I as IntoIterator>::IntoIter: ExactSizeIterator,
        <I as IntoIterator>::Item: Into<[f64; 2]>,
        <I as IntoIterator>::Item: Clone,
    {
        let mut wn = 0;
        for (a, b) in poly.into_iter().circular_tuple_windows() {
            let [_, ya] = a.clone().into();
            let [_, yb] = b.clone().into();
            if ya <= self.y {
                // `a` is below self
                if yb > self.y {
                    // an upward crossing
                    if matches!(self.position(a, b), Positioning::Left) {
                        wn += 1;
                    }
                }
            } else {
                // `a` is above self
                if yb <= self.y {
                    // a downward crossing
                    if matches!(self.position(a, b), Positioning::Right) {
                        wn -= 1;
                    }
                }
            }
        }
        wn
    }

    /// Returns `true` if the point is inside the input polygon.
    pub fn is_inside<I>(&self, poly: I) -> bool
    where
        I: IntoIterator,
        <I as IntoIterator>::IntoIter: Clone,
        <I as IntoIterator>::IntoIter: ExactSizeIterator,
        <I as IntoIterator>::Item: Into<[f64; 2]>,
        <I as IntoIterator>::Item: Clone,
    {
        self.wn(poly) != 0
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
        //
        //            2
        //
        //
        //     +------6------+
        //     |             |
        //     |             |
        //     |             |
        //     3      0      5      1
        //     |             |
        //     |             |
        //     |             |
        //     +------4------+
        //
        let poly: Vec<_> = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
            .iter()
            .map(|&[x, y]| Point { x, y })
            .collect();

        let p0 = Point { x: 0.5, y: 0.5 };
        let p1 = Point { x: 1.5, y: 0.5 };
        let p2 = Point { x: 0.5, y: 1.5 };
        let p3 = Point { x: 0., y: 0.5 };
        let p4 = Point { x: 0.5, y: 0. };
        let p5 = Point { x: 1.0, y: 0.5 };
        let p6 = Point { x: 0.5, y: 1. };
        assert_eq!(p0.wn(&poly), 1);
        assert_eq!(p1.wn(&poly), 0);
        assert_eq!(p2.wn(&poly), 0);
        assert_eq!(p3.wn(&poly), 1); // Left edges are included
        assert_eq!(p4.wn(&poly), 1); // Bottom edges are included
        assert_eq!(p5.wn(&poly), 0); // Right edges are not included
        assert_eq!(p6.wn(&poly), 0); // Top edges are not included
    }

    #[test]
    fn winding_number_square_with_array() {
        let poly = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]];

        let p0 = Point { x: 0.5, y: 0.5 };
        let p1 = Point { x: 1.5, y: 0.5 };
        let p2 = Point { x: 0.5, y: 1.5 };
        let p3 = Point { x: 0., y: 0.5 };
        let p4 = Point { x: 0.5, y: 0. };
        let p5 = Point { x: 1.0, y: 0.5 };
        let p6 = Point { x: 0.5, y: 1. };
        assert_eq!(p0.wn(poly), 1);
        assert_eq!(p1.wn(poly), 0);
        assert_eq!(p2.wn(poly), 0);
        assert_eq!(p3.wn(poly), 1); // Left edges are included
        assert_eq!(p4.wn(poly), 1); // Bottom edges are included
        assert_eq!(p5.wn(poly), 0); // Right edges are not included
        assert_eq!(p6.wn(poly), 0); // Top edges are not included
    }

    #[test]
    fn winding_number_self_overlapping_polygon() {
        //
        // Think of the following polygon like an "L" with an outgrowth that goes up and to the
        // right, so that there is a band that covers the vertical part of the "L" in which points
        // are "twice" inside the polygon.
        //
        //     +------------+
        //     |            |
        //     |  +----------------------+
        //     |  |         |            |
        //     |  | inside  |            |
        //     |  | twice   |            |
        //     |  |      2  |            |
        //     |  +-------------------+  |
        //     |            |         |  |
        //     |            | outside |  |
        //     |            |    1    |  |
        //     |            +---------+  |
        //     | 0                       |
        //     +-------------------------+
        //
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

        let p0 = Point { x: 0.1, y: 0.1 };
        let p1 = Point { x: 0.6, y: 0.3 };
        let p2 = Point { x: 0.4, y: 0.6 };
        assert_eq!(p0.wn(&poly), 1);
        assert_eq!(p1.wn(&poly), 0);
        assert_eq!(p2.wn(&poly), 2);
    }
}
