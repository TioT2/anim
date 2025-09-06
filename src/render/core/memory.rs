//! Memory allocation and implementation file
//!
//! TODO:
//! Create synchronize flush operations (?)

use std::{cell::{Cell, RefCell}, sync::Arc};

use ash::vk;
use vk_mem::Alloc;

use crate::render::core::{device_context::DeviceContext, util::DropGuard};

/// Staging buffer
struct StagingBuffer {
    /// Underlying allocation
    allocation: vk_mem::Allocation,

    /// Buffer
    buffer: vk::Buffer,
}

/// Context of the flush operation, needs to be saved while flush is not finished
/// # Safety
/// Flush context is drop-safe only after flushed operations completed
pub struct FlushContext {
    /// Allocator reference
    allocator: Arc<Allocator>,

    /// Buffer
    staging_buffers: Vec<StagingBuffer>,

    /// Command buffer
    command_buffer: vk::CommandBuffer,
}

impl Drop for FlushContext {
    fn drop(&mut self) {
        // Command buffer and staging buffer array data
        unsafe { self.allocator.free_write_command_buffer(self.command_buffer) };
        self.staging_buffers.drain(..).for_each(|mut b| unsafe {
            self.allocator.destroy_buffer(b.buffer, &mut b.allocation);
        });
    }
}

/// Object that manages memory allocations and transitions
pub struct Allocator {
    /// Ensure that device context is alive
    dc: Arc<DeviceContext>,

    /// Pool of the flush semaphores
    _flush_semaphores: Arc<RefCell<Vec<vk::Semaphore>>>,

    /// Semaphore to await on next flush
    prev_flush_semaphore: Option<vk::Semaphore>,

    /// Set of staging buffers
    staging_buffers: RefCell<Vec<StagingBuffer>>,

    /// Command pool created for write operations
    write_command_pool: vk::CommandPool,

    /// Command buffer used for 'write_*' (host -> device) operation family
    write_command_buffer: Cell<vk::CommandBuffer>,

    /// Underlying allocator
    allocator: vk_mem::Allocator,
}

impl Allocator {
    /// Allocate write command buffer
    fn alloc_write_cb(dc: &DeviceContext, pool: vk::CommandPool) -> Result<vk::CommandBuffer, vk::Result> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(pool);

        let command_buffer = DropGuard::new(
            unsafe { dc.device.allocate_command_buffers(&alloc_info) }?[0],
            |cb| unsafe { dc.device.free_command_buffers(pool, std::array::from_ref(cb)) }
        );

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { dc.device.begin_command_buffer(*command_buffer, &begin_info)? };

        Ok(command_buffer.into_inner())
    }

    /// Create memory allocator
    pub fn new(dc: Arc<DeviceContext>) -> Result<Self, vk::Result> {
        let allocator = {
            let mut create_info = vk_mem::AllocatorCreateInfo::new(&dc.instance, &dc.device, dc.physical_device);
            create_info.vulkan_api_version = dc.api_version;
            unsafe { vk_mem::Allocator::new(create_info) }?
        };

        let write_command_pool = {
            let create_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(dc.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            DropGuard::new(
                unsafe { dc.device.create_command_pool(&create_info, None) }?,
                |pool| unsafe { dc.device.destroy_command_pool(*pool, None) }
            )
        };

        let write_command_buffer = DropGuard::new(
            Self::alloc_write_cb(dc.as_ref(), *write_command_pool)?,
            |cb| unsafe { dc.device.free_command_buffers(*write_command_pool, std::array::from_ref(cb)) }
        );

        Ok(Allocator {
            write_command_buffer: Cell::new(write_command_buffer.into_inner()),
            write_command_pool: write_command_pool.into_inner(),
            staging_buffers: RefCell::new(Vec::new()),
            prev_flush_semaphore: None,
            _flush_semaphores: Arc::new(RefCell::new(Vec::new())),
            allocator,
            dc,
        })
    }

    /// Create raw vulkan buffer
    pub unsafe fn create_buffer(
        &self,
        buffer_create_info: &vk::BufferCreateInfo,
        allocation_create_info: &vk_mem::AllocationCreateInfo
    ) -> Result<(vk::Buffer, vk_mem::Allocation), vk::Result> {
        let (buffer, allocation) = unsafe {
            self.allocator.create_buffer(
                buffer_create_info,
                allocation_create_info
            )?
        };

        Ok((buffer, allocation))
    }

    /// Destroy raw vulkan buffer
    pub unsafe fn destroy_buffer(
        &self,
        buffer: vk::Buffer,
        mut allocation: &mut vk_mem::Allocation
    ) {
        unsafe { self.allocator.destroy_buffer(buffer, &mut allocation) };
    }

    /// Switch current command buffer to the new one
    /// # TODO
    /// Is this function **really** required?
    unsafe fn switch_write_command_buffer(&self) -> Result<vk::CommandBuffer, vk::Result> {
        // Allocate new command buffer
        let new_command_buffer = Self::alloc_write_cb(self.dc.as_ref(), self.write_command_pool)?;
        let old_command_buffer = self.write_command_buffer.replace(new_command_buffer);

        // Segfault on NULL. Where are my validation layers???
        unsafe { self.dc.device.end_command_buffer(old_command_buffer)? };

        Ok(old_command_buffer)
    }

    /// Free flush operation command buffer
    unsafe fn free_write_command_buffer(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.dc.device.free_command_buffers(
                self.write_command_pool,
                std::array::from_ref(&command_buffer)
            );
        }
    }

    /// Create staging buffer and fill it with corresponding data
    unsafe fn write_staging_buffer(&self, data: &[u8]) -> Result<vk::Buffer, vk::Result> {
        let (buffer, mut allocation) = {
            let alloc_info = vk_mem::AllocationCreateInfo {
                flags: vk_mem::AllocationCreateFlags::empty()
                    | vk_mem::AllocationCreateFlags::MAPPED
                    | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                required_flags: vk::MemoryPropertyFlags::empty()
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::HOST_VISIBLE,
                usage: vk_mem::MemoryUsage::AutoPreferHost,
                ..Default::default()
            };
            let create_info = vk::BufferCreateInfo::default()
                .queue_family_indices(std::array::from_ref(&self.dc.queue_family_index))
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .size(data.len() as u64)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC);

            unsafe { self.allocator.create_buffer(&create_info, &alloc_info) }?
        };
        let info = self.allocator.get_allocation_info(&allocation);

        // Write data to the staging buffer
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                info.mapped_data as *mut u8,
                data.len()
            );
        }

        // Unmap staging buffer and write it to corresponding buffer set
        unsafe { self.allocator.unmap_memory(&mut allocation) };
        self.staging_buffers.borrow_mut().push(StagingBuffer { buffer, allocation });

        Ok(buffer)
    }

    /// Write data to device buffer from host
    pub unsafe fn write_buffer(
        &self,
        buffer: vk::Buffer,
        offset: usize,
        data: &[u8]
    ) -> Result<(), vk::Result> {
        let staging_buffer = unsafe { self.write_staging_buffer(data) }?;

        // Copy from staging buffer to target buffer
        unsafe {
            self.dc.device.cmd_copy_buffer(
                self.write_command_buffer.get(),
                staging_buffer,
                buffer,
                std::array::from_ref(&vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: offset as u64,
                    size: data.len() as u64,
                })
            );
        }

        Ok(())
    }

    /// Flush all copy operations
    /// # TODO
    /// Return none if no operations that requires flush performed
    pub fn flush(self: Arc<Self>, transfer_end_semaphore: vk::Semaphore) -> Result<FlushContext, vk::Result> {
        let command_buffer = DropGuard::new(
            unsafe { self.switch_write_command_buffer() }?,
            |cb| unsafe { self.dc.device.free_command_buffers(self.write_command_pool, std::array::from_ref(cb)) }
        );
        let staging_buffers = self.staging_buffers.replace(Vec::new());

        let dst_stage_mask = vk::PipelineStageFlags::TRANSFER;
        let (wait_semaphores, wait_dst_stages) = if let Some(flush_semaphore) = self.prev_flush_semaphore.as_ref() {
            (
                std::array::from_ref(flush_semaphore).as_slice(),
                std::array::from_ref(&dst_stage_mask).as_slice(),
            )
        } else {
            ([].as_slice(), [].as_slice())
        };

        let submit = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_dst_stages)
            .command_buffers(std::array::from_ref(&*command_buffer))
            .signal_semaphores(std::array::from_ref(&transfer_end_semaphore));

        // Submit command buffer
        unsafe {
            self.dc.device.queue_submit(self.dc.queue, std::array::from_ref(&submit), vk::Fence::null())?;
        }

        Ok(FlushContext {
            command_buffer: command_buffer.into_inner(),
            allocator: self.clone(),
            staging_buffers,
        })
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        unsafe {
            _ = self.dc.device.device_wait_idle();
            for mut buffer in self.staging_buffers.replace(Vec::new()).into_iter() {
                self.destroy_buffer(buffer.buffer, &mut buffer.allocation);
            }
            // Do not perform operations from last command buffer
            _ = self.dc.device.end_command_buffer(self.write_command_buffer.get());
            self.dc.device.destroy_command_pool(self.write_command_pool, None);
        }
    }
}
