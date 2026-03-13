use uuid::Uuid;

#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    center: [f32; 3],
    half_size: f32,
}

impl BoundingBox {
    fn contains(&self, point: [f32; 3]) -> bool {
        (0..3).all(|idx| (point[idx] - self.center[idx]).abs() <= self.half_size)
    }

    fn intersects_sphere(&self, center: [f32; 3], radius: f32) -> bool {
        (0..3).all(|idx| (self.center[idx] - center[idx]).abs() <= self.half_size + radius)
    }
}

#[derive(Debug, Clone)]
pub struct OctreeNode {
    boundary: BoundingBox,
    capacity: usize,
    units: Vec<(Uuid, [f32; 3])>,
    children: Option<Vec<OctreeNode>>,
}

impl OctreeNode {
    pub fn new(center: [f32; 3], half_size: f32) -> Self {
        Self {
            boundary: BoundingBox { center, half_size },
            capacity: 8,
            units: Vec::new(),
            children: None,
        }
    }

    pub fn insert(&mut self, id: Uuid, pos: [f32; 3]) -> bool {
        if !self.boundary.contains(pos) {
            return false;
        }

        if self.units.len() < self.capacity {
            self.units.push((id, pos));
            return true;
        }

        if self.children.is_none() {
            self.subdivide();
        }

        if let Some(children) = &mut self.children {
            for child in children {
                if child.insert(id, pos) {
                    return true;
                }
            }
        }
        false
    }

    pub fn query(&self, center: [f32; 3], radius: f32, results: &mut Vec<Uuid>) {
        if !self.boundary.intersects_sphere(center, radius) {
            return;
        }

        for (id, pos) in &self.units {
            if distance(*pos, center) <= radius {
                results.push(*id);
            }
        }

        if let Some(children) = &self.children {
            for child in children {
                child.query(center, radius, results);
            }
        }
    }

    fn subdivide(&mut self) {
        let new_half = self.boundary.half_size / 2.0;
        let center = self.boundary.center;
        let mut children = Vec::with_capacity(8);
        for &dx in &[-1.0, 1.0] {
            for &dy in &[-1.0, 1.0] {
                for &dz in &[-1.0, 1.0] {
                    children.push(OctreeNode::new(
                        [
                            center[0] + dx * new_half,
                            center[1] + dy * new_half,
                            center[2] + dz * new_half,
                        ],
                        new_half,
                    ));
                }
            }
        }
        self.children = Some(children);
    }
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    ((dx * dx) + (dy * dy) + (dz * dz)).sqrt()
}
