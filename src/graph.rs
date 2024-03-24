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
        let node = Node::<usize>::new(0);

        assert_eq!(node.parents.len(), 0);
        assert_eq!(node.children.len(), 0);
    }

    #[test]
    fn add_node_to_graph() {
        let mut graph = Graph::<usize>::new();

        let idx = graph.add(42);
        assert_eq!(idx, 0);
        assert_eq!(graph.count(), 1);

        let idx = graph.add(314);
        assert_eq!(idx, 1);
        assert_eq!(graph.count(), 2);
    }
}
