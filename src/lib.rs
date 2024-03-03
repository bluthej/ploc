use std::ptr;
use std::{fmt::Display, marker::PhantomData, ptr::NonNull};

#[derive(Debug)]
struct Face {
    boundary: Link,
    len: usize,
}

type Link = Option<NonNull<Edge>>;

#[derive(Debug)]
struct Edge {
    next: Link,
    prev: Link,
    origin: Vertex,
}

impl Display for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [xo, yo] = self.origin.coords;
        if let Some(next) = self.next {
            let [xn, yn] = unsafe { (*next.as_ptr()).origin.coords };
            write!(f, "({}, {})->({}, {})", xo, yo, xn, yn)
        } else {
            write!(f, "({}, {})->(None)", xo, yo)
        }
    }
}

#[derive(Debug)]
struct Vertex {
    coords: [f32; 2],
    edge: Link,
}

impl Vertex {
    fn new_with_coords(x: f32, y: f32) -> Self {
        Self {
            coords: [x, y],
            edge: None,
        }
    }
}

impl Display for Vertex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [x, y] = self.coords;
        write!(f, "({}, {})", x, y)
    }
}

struct VertexIter<'a> {
    current: Link,
    len: usize,
    _boo: PhantomData<&'a Edge>,
}

struct EdgeIter<'a> {
    current: Link,
    len: usize,
    _boo: PhantomData<&'a Edge>,
}

impl Face {
    fn new() -> Self {
        Self {
            boundary: None,
            len: 0,
        }
    }

    fn push(&mut self, vert: Vertex) {
        // SAFETY: it's a linked-list, what do you want?
        unsafe {
            let new = NonNull::new_unchecked(Box::into_raw(Box::new(Edge {
                next: None,
                prev: None,
                origin: vert,
            })));
            if let Some(old) = self.boundary {
                // Put the new front before the old one
                (*old.as_ptr()).next = Some(new);
                (*new.as_ptr()).prev = Some(old);
            } else {
                // If there's no front, then we're the empty list and need
                // to set the back too.
                self.boundary = Some(new);
            }
            // These things always happen!
            self.boundary = Some(new);
            self.len += 1;
        }
    }

    fn close(&mut self) {
        unsafe {
            let last = self.boundary.unwrap();
            let mut boundary = last;
            while let Some(prev) = (*boundary.as_ptr()).prev {
                boundary = prev;
            }
            (*boundary.as_ptr()).prev = Some(last);
            (*last.as_ptr()).next = Some(boundary);
            self.boundary = Some(boundary);
        }
    }

    pub fn vertex_iter(&self) -> VertexIter {
        VertexIter {
            current: self.boundary,
            len: self.len,
            _boo: PhantomData,
        }
    }

    pub fn edge_iter(&self) -> EdgeIter {
        EdgeIter {
            current: self.boundary,
            len: self.len,
            _boo: PhantomData,
        }
    }
}

impl Drop for Face {
    fn drop(&mut self) {
        unsafe {
            if let Some(first) = self.boundary {
                if let Some(last) = (*first.as_ptr()).prev {
                    (*last.as_ptr()).next = None;
                    (*first.as_ptr()).prev = None;
                }
                let mut current = first;
                while let Some(next) = (*current.as_ptr()).next {
                    (*current.as_ptr()).next = None;
                    (*next.as_ptr()).prev = None;
                    current = next;
                }
            };
        }
    }
}

impl Default for Face {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Iterator for VertexIter<'a> {
    type Item = &'a Vertex;

    fn next(&mut self) -> Option<Self::Item> {
        // While self.front == self.back is a tempting condition to check here,
        // it won't do the right for yielding the last element! That sort of
        // thing only works for arrays because of "one-past-the-end" pointers.
        if self.len > 0 {
            // We could unwrap front, but this is safer and easier
            self.current.map(|edge| unsafe {
                self.len -= 1;
                self.current = (*edge.as_ptr()).next;
                &(*edge.as_ptr()).origin
            })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a> Iterator for EdgeIter<'a> {
    type Item = &'a Edge;

    fn next(&mut self) -> Option<Self::Item> {
        // While self.front == self.back is a tempting condition to check here,
        // it won't do the right for yielding the last element! That sort of
        // thing only works for arrays because of "one-past-the-end" pointers.
        if self.len > 0 {
            // We could unwrap front, but this is safer and easier
            self.current.map(|edge| unsafe {
                self.len -= 1;
                self.current = (*edge.as_ptr()).next;
                &(*edge.as_ptr())
            })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_face() {
        let mut face = Face::new();
        face.push(Vertex::new_with_coords(0., 0.));
        face.push(Vertex::new_with_coords(1., 0.));
        face.push(Vertex::new_with_coords(0., 1.));
        face.close();

        assert_eq!(face.len, 3);

        for vertex in face.vertex_iter() {
            println!("{}", vertex);
        }

        for edge in face.edge_iter() {
            println!("{}", edge);
        }
    }
}
