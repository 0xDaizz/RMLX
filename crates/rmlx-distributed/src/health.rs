use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct HeartbeatConfig {
    pub interval: Duration,
    pub timeout: Duration,
    pub max_missed: u32,
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(1),
            timeout: Duration::from_secs(5),
            max_missed: 3,
        }
    }
}

#[derive(Debug)]
struct HealthState {
    last_seen: HashMap<u32, Instant>,
    config: HeartbeatConfig,
}

#[derive(Clone, Debug)]
pub struct HealthMonitor {
    inner: Arc<RwLock<HealthState>>,
}

impl HealthMonitor {
    pub fn new(config: HeartbeatConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HealthState {
                last_seen: HashMap::new(),
                config,
            })),
        }
    }

    pub fn record_heartbeat(&self, peer_rank: u32) {
        let mut state = self.inner.write().expect("health monitor lock poisoned");
        state.last_seen.insert(peer_rank, Instant::now());
    }

    pub fn check_health(&self) -> Vec<u32> {
        let state = self.inner.read().expect("health monitor lock poisoned");
        let timeout = state.config.timeout;
        let now = Instant::now();
        let mut unhealthy: Vec<u32> = state
            .last_seen
            .iter()
            .filter_map(|(&peer_rank, &last_seen)| {
                (now.duration_since(last_seen) > timeout).then_some(peer_rank)
            })
            .collect();
        unhealthy.sort_unstable();
        unhealthy
    }

    pub fn is_healthy(&self, peer_rank: u32) -> bool {
        let state = self.inner.read().expect("health monitor lock poisoned");
        state
            .last_seen
            .get(&peer_rank)
            .is_some_and(|last_seen| last_seen.elapsed() <= state.config.timeout)
    }
}

type HeartbeatSendFn = dyn Fn(u32) -> Result<(), String> + Send + Sync;

pub struct HeartbeatSender {
    pub config: HeartbeatConfig,
    pub local_rank: u32,
    pub send_fn: Box<HeartbeatSendFn>,
}

impl HeartbeatSender {
    pub fn new(
        config: HeartbeatConfig,
        local_rank: u32,
        send_fn: impl Fn(u32) -> Result<(), String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            config,
            local_rank,
            send_fn: Box::new(send_fn),
        }
    }

    pub fn send_heartbeat(&self) -> Result<(), String> {
        (self.send_fn)(self.local_rank)
    }
}

#[cfg(test)]
mod tests {
    use super::{HealthMonitor, HeartbeatConfig, HeartbeatSender};
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    fn test_config() -> HeartbeatConfig {
        HeartbeatConfig {
            interval: Duration::from_millis(10),
            timeout: Duration::from_millis(40),
            max_missed: 3,
        }
    }

    #[test]
    fn test_healthy_peer() {
        let monitor = HealthMonitor::new(test_config());

        monitor.record_heartbeat(7);

        assert!(monitor.is_healthy(7));
        assert!(monitor.check_health().is_empty());
    }

    #[test]
    fn test_timeout_detection() {
        let monitor = HealthMonitor::new(test_config());

        monitor.record_heartbeat(11);
        thread::sleep(Duration::from_millis(60));

        assert!(!monitor.is_healthy(11));
        assert_eq!(monitor.check_health(), vec![11]);
    }

    #[test]
    fn test_multiple_peers() {
        let monitor = HealthMonitor::new(test_config());

        monitor.record_heartbeat(1);
        monitor.record_heartbeat(2);
        monitor.record_heartbeat(3);

        thread::sleep(Duration::from_millis(60));

        monitor.record_heartbeat(1);
        monitor.record_heartbeat(3);

        assert_eq!(monitor.check_health(), vec![2]);
    }

    #[test]
    fn test_heartbeat_sender() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = Arc::clone(&counter);
        let sender = HeartbeatSender::new(test_config(), 5, move |rank| {
            if rank != 5 {
                return Err(format!("unexpected rank: {rank}"));
            }

            counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });

        sender
            .send_heartbeat()
            .expect("heartbeat send should succeed");

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_default_config() {
        let config = HeartbeatConfig::default();

        assert_eq!(config.interval, Duration::from_secs(1));
        assert_eq!(config.timeout, Duration::from_secs(5));
        assert_eq!(config.max_missed, 3);
    }

    #[test]
    fn test_unknown_peer_is_unhealthy() {
        let monitor = HealthMonitor::new(test_config());
        // A peer with no recorded heartbeat should not be considered healthy.
        assert!(!monitor.is_healthy(99));
    }

    #[test]
    fn test_check_health_empty_returns_empty() {
        let monitor = HealthMonitor::new(test_config());
        // No peers registered at all.
        assert!(monitor.check_health().is_empty());
    }

    #[test]
    fn test_refresh_heartbeat_resets_timeout() {
        let monitor = HealthMonitor::new(test_config());
        monitor.record_heartbeat(1);

        // Wait almost to timeout.
        thread::sleep(Duration::from_millis(30));
        // Refresh heartbeat before timeout.
        monitor.record_heartbeat(1);

        // Wait another short period.
        thread::sleep(Duration::from_millis(20));
        // Should still be healthy because we refreshed.
        assert!(monitor.is_healthy(1));
    }

    #[test]
    fn test_monitor_is_clone_and_shared() {
        let monitor = HealthMonitor::new(test_config());
        let monitor2 = monitor.clone();

        monitor.record_heartbeat(5);
        // The clone should see the same data (Arc-backed).
        assert!(monitor2.is_healthy(5));
    }

    #[test]
    fn test_heartbeat_sender_error_propagation() {
        let sender =
            HeartbeatSender::new(test_config(), 0, |_rank| Err("network down".to_string()));

        let result = sender.send_heartbeat();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "network down");
    }

    #[test]
    fn test_check_health_sorted_output() {
        let monitor = HealthMonitor::new(test_config());

        // Register peers in non-sorted order.
        monitor.record_heartbeat(10);
        monitor.record_heartbeat(3);
        monitor.record_heartbeat(7);

        // Wait for all to time out.
        thread::sleep(Duration::from_millis(60));

        let unhealthy = monitor.check_health();
        // Should be sorted.
        assert_eq!(unhealthy, vec![3, 7, 10]);
    }
}
