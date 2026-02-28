//! Tests for GPU event and stream manager.

use rmlx_metal::device::GpuDevice;
use rmlx_metal::event::{EventError, GpuEvent};
use rmlx_metal::stream::StreamManager;
use std::time::Duration;

#[test]
fn test_gpu_event_creation() {
    let device = match GpuDevice::system_default() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skipping: no Metal device");
            return;
        }
    };
    let _event = GpuEvent::new(device.raw());
}

#[test]
fn test_gpu_event_counter() {
    let device = match GpuDevice::system_default() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skipping: no Metal device");
            return;
        }
    };
    let event = GpuEvent::new(device.raw());
    assert_eq!(event.current_value(), 0);
    let v1 = event.next_value();
    assert_eq!(v1, 1);
    let v2 = event.next_value();
    assert_eq!(v2, 2);
    assert_eq!(event.current_value(), 2);
}

#[test]
fn test_gpu_event_cancel() {
    let device = match GpuDevice::system_default() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skipping: no Metal device");
            return;
        }
    };
    let event = GpuEvent::new(device.raw());
    event.cancel();
    let result = event.cpu_wait(999, Duration::from_secs(1));
    assert!(matches!(result, Err(EventError::Cancelled)));

    event.reset_cancel();
    // After reset, timeout should work normally
    let result = event.cpu_wait(999, Duration::from_millis(10));
    assert!(matches!(result, Err(EventError::Timeout(_))));
}

#[test]
fn test_stream_manager_creation() {
    let device = match GpuDevice::system_default() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skipping: no Metal device");
            return;
        }
    };
    let _streams = StreamManager::new(device.raw());
}

#[test]
fn test_stream_manager_dual_queue() {
    let device = match GpuDevice::system_default() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skipping: no Metal device");
            return;
        }
    };
    let streams = StreamManager::new(device.raw());
    // Both queues should be usable
    let _cb1 = streams.compute_queue().new_command_buffer();
    let _cb2 = streams.transfer_queue().new_command_buffer();
}
