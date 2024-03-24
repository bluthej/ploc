use anyhow::{anyhow, Result};
use std::slice::Iter;

#[derive(Debug, Default)]
struct Graph<T> {
    arena: Vec<Node<T>>,
}

#[derive(Debug, Default)]
struct Node<T> {
    data: T,
    parents: Vec<usize>,
    children: Vec<usize>,
}

impl<T> Graph<T> {
    fn new() -> Self {
        Graph { arena: Vec::new() }
    }

    fn count(&self) -> usize {
        self.arena.len()
    }

    fn add(&mut self, data: T) -> usize {
        let idx = self.arena.len();
        self.arena.push(Node::new(data));
        idx
    }

    fn get(&self, idx: usize) -> Option<&Node<T>> {
        self.arena.get(idx)
    }

    fn get_mut(&mut self, idx: usize) -> Option<&mut Node<T>> {
        self.arena.get_mut(idx)
    }

    fn iter(&self) -> Iter<'_, Node<T>> {
        self.arena.iter()
    }

    fn append_to(&mut self, data: T, idx: usize) -> Result<usize> {
        self.append_to_many(data, &[idx])
    }

    fn append_to_many(&mut self, data: T, idxs: &[usize]) -> Result<usize> {
        let new_idx = self.add(data);
        for &idx in idxs {
            self.arena
                .get_mut(idx)
                .ok_or(anyhow!("Node with index {} does not exist.", idx))?
                .children
                .push(new_idx);
            self.arena[new_idx].parents.push(idx);
        }
        Ok(new_idx)
    }

    fn insert_before(&mut self, data: T, idx: usize) -> Result<usize> {
        let new_idx = self.add(data);
        // Store old node's parents
        let old_node = self
            .arena
            .get_mut(idx)
            .ok_or(anyhow!("Node with index {} does not exist.", idx))?;
        let old_parents = std::mem::take(&mut old_node.parents);
        // Swap the node indices so that the new node takes the old node's place
        self.arena.swap(idx, new_idx);
        // Set the parents and children
        self.arena[idx].parents = old_parents;
        self.arena[idx].children.push(new_idx);
        self.arena[new_idx].parents.push(idx);
        Ok(new_idx)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_empty_graph() {
        let graph = Graph::<usize>::new();

        assert_eq!(graph.count(), 0);
    }

    #[test]
    fn create_node() {
        let node = Node::new(0);

        assert_eq!(node.parents.len(), 0);
        assert_eq!(node.children.len(), 0);
    }

    #[test]
    fn add_node_to_graph() {
        let mut graph = Graph::new();

        let idx = graph.add(42);
        assert_eq!(idx, 0);
        assert_eq!(graph.count(), 1);

        let idx = graph.add(314);
        assert_eq!(idx, 1);
        assert_eq!(graph.count(), 2);
    }

    #[test]
    fn graph_iter() {
        let mut graph = Graph::<usize>::new();
        graph.add(42);
        graph.add(314);

        let values: Vec<usize> = graph.iter().map(|node| node.data).collect();

        assert_eq!(&values, &[42, 314]);
    }

    #[test]
    fn append_node() -> Result<()> {
        let mut graph = Graph::<usize>::new();
        let idx0 = graph.add(42);

        let idx = graph.append_to(314, idx0)?;

        assert_eq!(idx, 1);
        assert_eq!(graph.get(idx0).unwrap().children, &[idx]);
        assert!(graph.get(idx0).unwrap().parents.is_empty());
        assert_eq!(graph.get(idx).unwrap().parents, &[idx0]);
        assert!(graph.get(idx).unwrap().children.is_empty());

        Ok(())
    }

    #[test]
    fn prepend_node() -> Result<()> {
        let mut graph = Graph::<usize>::new();
        let idx0 = graph.add(42);

        let idx = graph.insert_before(314, idx0)?;

        assert_eq!(idx, 1);
        assert_eq!(graph.get(idx0).unwrap().data, 314);
        assert_eq!(graph.get(idx0).unwrap().children, &[idx]);
        assert!(graph.get(idx0).unwrap().parents.is_empty());
        assert_eq!(graph.get(idx).unwrap().data, 42);
        assert_eq!(graph.get(idx).unwrap().parents, &[idx0]);
        assert!(graph.get(idx).unwrap().children.is_empty());

        Ok(())
    }

    #[test]
    fn prepend_node_with_multiple_parents() -> Result<()> {
        let mut graph = Graph::<usize>::new();
        let idx0 = graph.add(42);
        let idx1 = graph.add(4);
        let idx2 = graph.append_to_many(16, &[idx0, idx1])?;

        let idx = graph.insert_before(314, idx2)?;

        assert_eq!(idx, 3);
        assert_eq!(graph.get(idx2).unwrap().data, 314);
        assert_eq!(graph.get(idx2).unwrap().children, &[idx]);
        assert_eq!(graph.get(idx2).unwrap().parents, &[idx0, idx1]);
        assert_eq!(graph.get(idx).unwrap().data, 16);
        assert_eq!(graph.get(idx).unwrap().parents, &[idx2]);
        assert!(graph.get(idx).unwrap().children.is_empty());

        Ok(())
    }
}
