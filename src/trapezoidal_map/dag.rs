use smallvec::SmallVec;
use std::{collections::HashSet, slice::Iter};

/// A Directed Acyclic Graph (DAG).
///
/// It turns out this is the kind of graph you need to represent a trapezoidal map. A tree won't cut
/// it because some nodes need to have multiple parents. This happens notably when a new edge crosses
/// multiple trapezoids, in which case these trapezoids are replaces with Y nodes that may share a
/// common child trapezoid node.
///
/// These kinds of graphs are generally implemented using pointers, but this is not very practical
/// or idiomatic in Rust, so here it is implemented using an arena. This means that it is actually
/// backed by a simple [`Vec`], and we keep track of the nodes using [`usize`]s. This is not the best
/// approach when you need to be able to efficiently remove nodes from the arena, but in the present
/// case we never actually remove nodes.
///
/// Note: This implementation is influenced by:
/// - [This blog post](https://dev.to/deciduously/no-more-tears-no-more-knots-arena-allocated-trees-in-rust-44k6)
/// - [indextree](https://crates.io/crates/indextree) (I was using it before I realized I needed a
///   graph instead of a tree)
#[derive(Debug, Default)]
pub(crate) struct Dag<T> {
    arena: Vec<Node<T>>,
}

impl<T> Dag<T> {
    /// Constructs a new empty DAG.
    pub(crate) fn new() -> Self {
        Dag { arena: Vec::new() }
    }

    /// Add a new node to the DAG. Returns the index of the node.
    pub(crate) fn add(&mut self, data: T) -> usize {
        let idx = self.arena.len();
        self.arena.push(Node::new(data));
        idx
    }

    /// Get a shared reference to the node with index `idx`, if it exists.
    pub(crate) fn get(&self, idx: usize) -> Option<&Node<T>> {
        self.arena.get(idx)
    }

    /// Get an exclusive reference to the node with index `idx`, if it exists.
    fn get_mut(&mut self, idx: usize) -> Option<&mut Node<T>> {
        self.arena.get_mut(idx)
    }

    /// An iterator over the DAG's nodes.
    pub(crate) fn iter(&self) -> Iter<'_, Node<T>> {
        self.arena.iter()
    }

    /// Gets the given index’ corresponding entry in the DAG for in-place manipulation.
    pub(crate) fn entry(&mut self, idx: usize) -> Entry<'_, T> {
        Entry { idx, dag: self }
    }

    pub(crate) fn depth(&self, idx: usize) -> Option<usize> {
        if idx >= self.arena.len() {
            return None;
        }

        let mut to_visit = HashSet::new();
        to_visit.insert(0);
        let mut buf = HashSet::new();
        let mut depth = 0;
        while !to_visit.contains(&idx) {
            for id in to_visit.drain() {
                buf.extend(self.get(id).expect("Should be valid").children.iter());
            }
            std::mem::swap(&mut to_visit, &mut buf);
            depth += 1;
        }
        Some(depth)
    }
}

/// A node of the DAG.
#[derive(Debug, Default)]
pub(crate) struct Node<T> {
    pub(crate) data: T,
    pub(crate) children: SmallVec<[usize; 2]>,
}

impl<T> Node<T> {
    fn new(data: T) -> Self {
        Node {
            data,
            children: SmallVec::new(),
        }
    }
}

/// A view into a single entry in a DAG, which may or may not exist yet.
pub(crate) struct Entry<'a, T> {
    idx: usize,
    dag: &'a mut Dag<T>,
}

impl<T> Entry<'_, T> {
    /// Creates and appends a new [`Node`] with given data to the entry, if it exists.
    pub(crate) fn append_new(&mut self, data: T) -> Option<usize> {
        let dag = &mut self.dag;
        let new_idx = dag.add(data);
        self.append(new_idx)
    }

    /// Appends an existing [`Node`] to the entry, if it exists.
    pub(crate) fn append(&mut self, idx: usize) -> Option<usize> {
        if self.dag.get(idx).is_some() {
            self.dag.arena[self.idx].children.push(idx);
            Some(idx)
        } else {
            None
        }
    }

    pub(crate) fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut T),
    {
        if let Some(node) = self.dag.get_mut(self.idx) {
            f(&mut node.data);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl<T> Dag<T> {
        /// Returns the number of nodes in the DAG.
        pub fn count(&self) -> usize {
            self.arena.len()
        }
    }

    #[test]
    fn create_empty_dag() {
        let dag = Dag::<usize>::new();

        assert_eq!(dag.count(), 0);
    }

    #[test]
    fn create_node() {
        let node = Node::new(0);

        assert_eq!(node.children.len(), 0);
    }

    #[test]
    fn add_node_to_dag() {
        let mut dag = Dag::new();

        let idx_42 = dag.add(42);
        assert_eq!(idx_42, 0);
        assert_eq!(dag.count(), 1);
        assert_eq!(dag.depth(idx_42), Some(0));

        let idx_314 = dag.entry(idx_42).append_new(314).unwrap();
        assert_eq!(idx_314, 1);
        assert_eq!(dag.count(), 2);
        assert_eq!(dag.depth(idx_314), Some(1));

        assert_eq!(dag.depth(2), None);
    }

    #[test]
    fn dag_iter() {
        let mut dag = Dag::new();
        dag.add(42);
        dag.add(314);

        let values: Vec<usize> = dag.iter().map(|node| node.data).collect();

        assert_eq!(&values, &[42, 314]);
    }
}
