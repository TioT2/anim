//! Memory allocation implementation file

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
    /// Buffer
    staging_buffers: Vec<StagingBuffer>,

    /// Command buffer
    command_buffer: vk::CommandBuffer,

    /// Allocator reference
    allocator: Arc<Allocator>,
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
/// # Note
/// Field order **does matter** here, Allocator **must be** dropped before DeviceContext (it may be the last reference)
pub struct Allocator {
    /// Underlying allocator
    allocator: vk_mem::Allocator,

    /// Ensure that device context is alive
    dc: Arc<DeviceContext>,

    /// Semaphore to await on next flush
    flush_semaphores: Cell<(vk::Semaphore, vk::Semaphore)>,

    /// Set of staging buffers
    staging_buffers: RefCell<Vec<StagingBuffer>>,

    /// Command pool created for write operations
    write_command_pool: vk::CommandPool,

    /// Command buffer used for 'write_*' (host -> device) operation family
    write_command_buffer: Cell<vk::CommandBuffer>,
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

        let flush_semaphores = {
            let create = || Ok(DropGuard::new(
                unsafe { dc.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?,
                |s| unsafe { dc.device.destroy_semaphore(*s, None) }
            ));

            DropGuard::zip(create()?, create()?)
        };

        // Perform empty submit to trigger signal operation for wait semaphore
        unsafe {
            let submit_info = vk::SubmitInfo::default()
                .signal_semaphores(std::array::from_ref(&flush_semaphores.1));

            dc.device.queue_submit(dc.queue, std::array::from_ref(&submit_info), vk::Fence::null())?;
        }


        Ok(Allocator {
            write_command_buffer: Cell::new(write_command_buffer.into_inner()),
            write_command_pool: write_command_pool.into_inner(),
            staging_buffers: RefCell::new(Vec::new()),
            flush_semaphores: Cell::new(flush_semaphores.into_inner()),
            allocator,
            dc,
        })
    }

    /// Create (infrequent) raw vulkan buffer
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
        allocation: &mut vk_mem::Allocation
    ) {
        unsafe { self.allocator.destroy_buffer(buffer, allocation) };
    }

    /// Create new raw vulkan image
    pub unsafe fn create_image(
        &self,
        image_create_info: &vk::ImageCreateInfo,
        allocation_create_info: &vk_mem::AllocationCreateInfo
    ) -> Result<(vk::Image, vk_mem::Allocation), vk::Result> {
        let (image, allocation) = unsafe {
            self.allocator.create_image(
                image_create_info,
                allocation_create_info
            )?
        };

        Ok((image, allocation))
    }

    /// Destroy raw vulkan image
    pub unsafe fn destroy_image(
        &self,
        image: vk::Image,
        allocation: &mut vk_mem::Allocation
    ) {
        unsafe { self.allocator.destroy_image(image, allocation) };
    }

    /// Write data from host to image
    pub unsafe fn _write_image(
        &self,
        image: vk::Image,
        current_layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
        data: &[u8]
    ) -> Result<(), vk::Result> {
        let staging_buffer = unsafe { self.write_staging_buffer(data) }?;

        unsafe {
            self.dc.device.cmd_copy_buffer_to_image(
                self.write_command_buffer.get(),
                staging_buffer,
                image,
                current_layout,
                regions
            );
        }

        Ok(())
    }

    /// Execute barrier on image memory
    pub unsafe fn _image_memory_barrier(&self, barrier: vk::ImageMemoryBarrier) {
        unsafe {
            self.dc.device.cmd_pipeline_barrier(
                self.write_command_buffer.get(),
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::array::from_ref(&barrier)
            );
        }
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

        let mapped_data = unsafe { self.allocator.map_memory(&mut allocation) }?;

        // Write data to the staging buffer
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                mapped_data as *mut u8,
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

        // Write copy command
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

    /// Access to the host buffer
    pub unsafe fn map_host_allocation<T>(
        &self,
        allocation: &mut vk_mem::Allocation,
        func: impl FnOnce(&mut [u8]) -> T
    ) -> Result<T, vk::Result> {
        _ = unsafe { self.allocator.map_memory(allocation) }?;
        let alloc_info = self.allocator.get_allocation_info(allocation);

        let mapped_slice = unsafe {
            std::slice::from_raw_parts_mut(alloc_info.mapped_data as *mut u8, alloc_info.size as usize)
        };

        // Write data to the staging buffer
        let value = func(mapped_slice);

        // Unmap staging buffer and write it to corresponding buffer set
        unsafe { self.allocator.unmap_memory(allocation) };

        Ok(value)
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

        let (trigger_semaphore, wait_semaphore) = self.flush_semaphores.get();
        // Swap semaphores
        self.flush_semaphores.set((wait_semaphore, trigger_semaphore));

        let dst_stage_mask = vk::PipelineStageFlags::TRANSFER;
        let signal_semaphores = [transfer_end_semaphore, trigger_semaphore];

        // Describe submit operation
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(std::array::from_ref(&wait_semaphore))
            .wait_dst_stage_mask(std::array::from_ref(&dst_stage_mask))
            .command_buffers(std::array::from_ref(&*command_buffer))
            .signal_semaphores(&signal_semaphores);

        // Submit command buffer
        unsafe {
            self.dc.device.queue_submit(
                self.dc.queue,
                std::array::from_ref(&submit_info),
                vk::Fence::null()
            )?;
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

            let (semaphore1, semaphore2) = self.flush_semaphores.get();
            self.dc.device.destroy_semaphore(semaphore1, None);
            self.dc.device.destroy_semaphore(semaphore2, None);

            for mut buffer in self.staging_buffers.replace(Vec::new()).into_iter() {
                self.destroy_buffer(buffer.buffer, &mut buffer.allocation);
            }
            // Do not perform operations from last command buffer
            _ = self.dc.device.end_command_buffer(self.write_command_buffer.get());
            self.dc.device.destroy_command_pool(self.write_command_pool, None);
        }
    }
}
