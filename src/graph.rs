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

    fn get(&self, idx: usize) -> &Node<T> {
        &self.arena[idx]
    }

    fn get_mut(&mut self, idx: usize) -> &mut Node<T> {
        &mut self.arena[idx]
    }

    fn iter(&self) -> Iter<'_, Node<T>> {
        self.arena.iter()
    }

    fn insert_after(&mut self, data: T, idx: usize) -> usize {
        let new_idx = self.add(data);
        self.arena[idx].children.push(new_idx);
        self.arena[new_idx].parents.push(idx);
        new_idx
    }

    fn insert_after_many(&mut self, data: T, idxs: &[usize]) -> usize {
        let new_idx = self.add(data);
        for &idx in idxs {
            self.arena[idx].children.push(new_idx);
            self.arena[new_idx].parents.push(idx);
        }
        new_idx
    }

    fn insert_before(&mut self, data: T, idx: usize) -> usize {
        let new_idx = self.add(data);
        let old_parents = std::mem::take(&mut self.arena[idx].parents);
        self.arena.swap(idx, new_idx);
        self.arena[idx].parents = old_parents;
        self.arena[idx].children.push(new_idx);
        self.arena[new_idx].parents.push(idx);
        new_idx
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
    fn append_node() {
        let mut graph = Graph::<usize>::new();
        let idx0 = graph.add(42);

        let idx = graph.insert_after(314, idx0);

        assert_eq!(idx, 1);
        assert_eq!(graph.get(idx0).children, &[idx]);
        assert!(graph.get(idx0).parents.is_empty());
        assert_eq!(graph.get(idx).parents, &[idx0]);
        assert!(graph.get(idx).children.is_empty());
    }

    #[test]
    fn prepend_node() {
        let mut graph = Graph::<usize>::new();
        let idx0 = graph.add(42);

        let idx = graph.insert_before(314, idx0);

        assert_eq!(idx, 1);
        assert_eq!(graph.get(idx0).data, 314);
        assert_eq!(graph.get(idx0).children, &[idx]);
        assert!(graph.get(idx0).parents.is_empty());
        assert_eq!(graph.get(idx).data, 42);
        assert_eq!(graph.get(idx).parents, &[idx0]);
        assert!(graph.get(idx).children.is_empty());
    }

    #[test]
    fn prepend_node_with_multiple_parents() {
        let mut graph = Graph::<usize>::new();
        let idx0 = graph.add(42);
        let idx1 = graph.add(4);
        let idx2 = graph.insert_after_many(16, &[idx0, idx1]);

        let idx = graph.insert_before(314, idx2);

        assert_eq!(idx, 3);
        assert_eq!(graph.get(idx2).data, 314);
        assert_eq!(graph.get(idx2).children, &[idx]);
        assert_eq!(graph.get(idx2).parents, &[idx0, idx1]);
        assert_eq!(graph.get(idx).data, 16);
        assert_eq!(graph.get(idx).parents, &[idx2]);
        assert!(graph.get(idx).children.is_empty());
    }
}
