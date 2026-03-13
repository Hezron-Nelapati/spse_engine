use crate::types::QueueDepths;
use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WorkPriority {
    Inference,
    InteractiveTraining,
    SilentBatch,
    Maintenance,
}

#[derive(Debug, Clone, Default)]
struct SchedulerState {
    queued: BTreeMap<WorkPriority, usize>,
    active: BTreeMap<WorkPriority, usize>,
}

impl SchedulerState {
    fn count(&self, priority: WorkPriority) -> usize {
        self.queued.get(&priority).copied().unwrap_or(0)
            + self.active.get(&priority).copied().unwrap_or(0)
    }

    fn active_higher_than(&self, priority: WorkPriority) -> bool {
        self.active
            .iter()
            .any(|(candidate, count)| *candidate < priority && *count > 0)
    }

    fn queued_higher_than(&self, priority: WorkPriority) -> bool {
        self.queued
            .iter()
            .any(|(candidate, count)| *candidate < priority && *count > 0)
    }
}

#[derive(Default)]
pub struct PriorityScheduler {
    state: Mutex<SchedulerState>,
    wake: Condvar,
}

impl PriorityScheduler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn acquire(self: &Arc<Self>, priority: WorkPriority) -> WorkPermit {
        let mut state = self.state.lock().expect("scheduler mutex poisoned");
        *state.queued.entry(priority).or_insert(0) += 1;
        while state.active_higher_than(priority) || state.queued_higher_than(priority) {
            state = self
                .wake
                .wait(state)
                .expect("scheduler condvar wait poisoned");
        }
        *state.queued.entry(priority).or_insert(1) -= 1;
        *state.active.entry(priority).or_insert(0) += 1;
        drop(state);

        WorkPermit {
            scheduler: self.clone(),
            priority,
        }
    }

    pub fn depths(&self) -> QueueDepths {
        let state = self.state.lock().expect("scheduler mutex poisoned");
        QueueDepths {
            inference: state.count(WorkPriority::Inference),
            interactive_training: state.count(WorkPriority::InteractiveTraining),
            silent_batch: state.count(WorkPriority::SilentBatch),
            maintenance: state.count(WorkPriority::Maintenance),
        }
    }
}

pub struct WorkPermit {
    scheduler: Arc<PriorityScheduler>,
    priority: WorkPriority,
}

impl Drop for WorkPermit {
    fn drop(&mut self) {
        let mut state = self
            .scheduler
            .state
            .lock()
            .expect("scheduler mutex poisoned");
        if let Some(active) = state.active.get_mut(&self.priority) {
            *active = active.saturating_sub(1);
        }
        self.scheduler.wake.notify_all();
    }
}
