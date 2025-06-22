//! Ploc is a library for efficient point location queries.
//!
//! It uses efficient data structures like trapezoidal maps,
//! as well as parallelism to make queries on many points as
//! fast as possible.
//!
//! # Basic usage
//!
//! A [`PointLocator`] trait is provided for any data structure that
//! can answer point location queries, the most simple one being one
//! that performs a linear search through the mesh to find the cells
//! that contain the query points. The main implementor provided by
//! Ploc is the trapezoidal map, or [`TrapMap`] for short.
//!
//! A trapezoidal map is constructed from a [`Mesh`] and then queried
//! as follows:
//!
//! ```
//! use ploc::{Mesh, PointLocator, TrapMap};
//!
//! // Create a `Mesh`
//! let (xmin, xmax) = (0., 2.);
//! let (ymin, ymax) = (0., 2.);
//! let mesh = Mesh::grid(xmin, xmax, ymin, ymax, 2, 2).unwrap();
//!
//! // Create a trapezoidal map
//! let trap_map = TrapMap::from_mesh(mesh);
//!
//! // Make a query
//! let query = vec![[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]];
//! let locations = trap_map.locate_many(&query);
//! assert_eq!(locations, vec![Some(0), Some(1), Some(2), Some(3)]);
//! ```

#![deny(missing_docs)]
// Ploc types in rustdoc of other crates get linked to here.
#![doc(html_root_url = "https://docs.rs/ploc/0.1.2")]

mod mesh;
mod point_locator;
mod rectilinear_locator;
mod trapezoidal_map;
mod winding_number;

pub use mesh::{CellVertices, Mesh};
pub use point_locator::PointLocator;
pub use rectilinear_locator::RectilinearLocator;
pub use trapezoidal_map::trap_map::TrapMap;
pub use winding_number::Point;
