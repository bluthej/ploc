use std::slice::Iter;

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
/// graph instead of a tree)
#[derive(Debug, Default)]
pub(crate) struct Dag<T> {
    arena: Vec<Node<T>>,
}

impl<T> Dag<T> {
    /// Constructs a new empty DAG.
    pub(crate) fn new() -> Self {
        Dag { arena: Vec::new() }
    }

    /// Returns the number of nodes in the DAG.
    fn count(&self) -> usize {
        self.arena.len()
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

    pub(crate) fn entry(&mut self, idx: usize) -> Entry<'_, T> {
        Entry { idx, dag: self }
    }

    pub(crate) fn entries<'a>(&'a mut self, idxs: &'a [usize]) -> Entries<'a, T> {
        Entries { idxs, dag: self }
    }
}

/// A node of the DAG.
#[derive(Debug, Default)]
pub(crate) struct Node<T> {
    pub(crate) data: T,
    parents: Vec<usize>,
    children: Vec<usize>,
}

impl<T> Node<T> {
    fn new(data: T) -> Self {
        Node {
            data,
            parents: Vec::new(),
            children: Vec::new(),
        }
    }
}

pub(crate) struct Entry<'a, T> {
    idx: usize,
    dag: &'a mut Dag<T>,
}

impl<T> Entry<'_, T> {
    pub(crate) fn append(&mut self, data: T) -> Option<usize> {
        let dag = &mut self.dag;
        let idx = self.idx;
        let new_idx = dag.add(data);
        if let Some(node) = dag.arena.get_mut(idx) {
            node.children.push(new_idx);
            dag.arena[new_idx].parents.push(idx);
            Some(new_idx)
        } else {
            None
        }
    }

    pub(crate) fn prepend(&mut self, data: T) -> Option<usize> {
        let dag = &mut self.dag;
        let idx = self.idx;
        let new_idx = dag.add(data);
        // Store old node's parents
        if let Some(old_node) = dag.arena.get_mut(idx) {
            let old_parents = std::mem::take(&mut old_node.parents);
            // Swap the node indices so that the new node takes the old node's place
            dag.arena.swap(idx, new_idx);
            // Set the parents and children
            dag.arena[idx].parents = old_parents;
            dag.arena[idx].children.push(new_idx);
            dag.arena[new_idx].parents.push(idx);
            Some(new_idx)
        } else {
            None
        }
    }
}

pub(crate) struct Entries<'a, T> {
    idxs: &'a [usize],
    dag: &'a mut Dag<T>,
}

impl<T> Entries<'_, T> {
    pub(crate) fn append(&mut self, data: T) -> Option<usize> {
        let dag = &mut self.dag;
        let idxs = self.idxs;
        let new_idx = dag.add(data);
        for &idx in idxs {
            if let Some(node) = dag.arena.get_mut(idx) {
                node.children.push(new_idx);
                dag.arena[new_idx].parents.push(idx);
            } else {
                return None;
            }
        }
        Some(new_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{anyhow, Result};

    #[test]
    fn create_empty_dag() {
        let dag = Dag::<usize>::new();

        assert_eq!(dag.count(), 0);
    }

    #[test]
    fn create_node() {
        let node = Node::new(0);

        assert_eq!(node.parents.len(), 0);
        assert_eq!(node.children.len(), 0);
    }

    #[test]
    fn add_node_to_dag() {
        let mut dag = Dag::new();

        let idx = dag.add(42);
        assert_eq!(idx, 0);
        assert_eq!(dag.count(), 1);

        let idx = dag.add(314);
        assert_eq!(idx, 1);
        assert_eq!(dag.count(), 2);
    }

    #[test]
    fn dag_iter() {
        let mut dag = Dag::new();
        dag.add(42);
        dag.add(314);

        let values: Vec<usize> = dag.iter().map(|node| node.data).collect();

        assert_eq!(&values, &[42, 314]);
    }

    #[test]
    fn append_node() -> Result<()> {
        let mut dag = Dag::new();
        let idx0 = dag.add(42);

        let idx = dag
            .entry(idx0)
            .prepend(314)
            .ok_or(anyhow!("Missing entry"))?;

        assert_eq!(idx, 1);
        assert_eq!(dag.get(idx0).unwrap().children, &[idx]);
        assert!(dag.get(idx0).unwrap().parents.is_empty());
        assert_eq!(dag.get(idx).unwrap().parents, &[idx0]);
        assert!(dag.get(idx).unwrap().children.is_empty());

        Ok(())
    }

    #[test]
    fn prepend_node() -> Result<()> {
        let mut dag = Dag::new();
        let idx0 = dag.add(42);

        let idx = dag
            .entry(idx0)
            .prepend(314)
            .ok_or(anyhow!("Missing entry"))?;

        assert_eq!(idx, 1);
        assert_eq!(dag.get(idx0).unwrap().data, 314);
        assert_eq!(dag.get(idx0).unwrap().children, &[idx]);
        assert!(dag.get(idx0).unwrap().parents.is_empty());
        assert_eq!(dag.get(idx).unwrap().data, 42);
        assert_eq!(dag.get(idx).unwrap().parents, &[idx0]);
        assert!(dag.get(idx).unwrap().children.is_empty());

        Ok(())
    }

    #[test]
    fn prepend_node_with_multiple_parents() -> Result<()> {
        let mut dag = Dag::new();
        let idx0 = dag.add(42);
        let idx1 = dag.add(4);
        let idx2 = dag
            .entries(&[idx0, idx1])
            .append(16)
            .ok_or(anyhow!("Missing entry"))?;

        let idx = dag
            .entry(idx2)
            .prepend(314)
            .ok_or(anyhow!("Missing entry"))?;

        assert_eq!(idx, 3);
        assert_eq!(dag.get(idx2).unwrap().data, 314);
        assert_eq!(dag.get(idx2).unwrap().children, &[idx]);
        assert_eq!(dag.get(idx2).unwrap().parents, &[idx0, idx1]);
        assert_eq!(dag.get(idx).unwrap().data, 16);
        assert_eq!(dag.get(idx).unwrap().parents, &[idx2]);
        assert!(dag.get(idx).unwrap().children.is_empty());

        Ok(())
    }
}
