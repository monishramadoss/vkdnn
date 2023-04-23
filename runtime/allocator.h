// allocator.h : Header file for your target.

#pragma once
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

struct vk_block
{
    size_t size = 0;
    size_t offset = 0;
    uint32_t device_id = 0;
    bool on_device{};
    bool free{};
    void* ptr = nullptr;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkBuffer buf = VK_NULL_HANDLE;
    VkImage img = VK_NULL_HANDLE;

    bool operator==(const vk_block& blk) const
    {
        return offset == blk.offset && size == blk.size &&
            free == blk.free && ptr == blk.ptr &&
            memory == blk.memory && buf == blk.buf &&
            img == blk.img && device_id == blk.device_id;
    }
};

class vk_chunk final
{
    uint32_t device_id_;
    VkDevice device_;
    VkDeviceMemory memory_{};
    void* ptr_=nullptr;

    size_t size_;
    std::vector<std::shared_ptr<vk_block>> blocks_;
    int memory_type_index_;
    bool device_local_;

    void collate_delete();

public:
    vk_chunk(const uint32_t device_id, const VkDevice& dev, const size_t size, const int memory_type_index);
    bool allocate(size_t size, size_t alignment, vk_block** blk);
    bool deallocate(const vk_block* blk) const;
    [[nodiscard]] VkDeviceMemory get_memory() const;
    void cleanup();
    void set_host_visible();
    ~vk_chunk();
};

class vk_allocator final
{
    uint32_t device_id_;
    size_t chunk_size_;
    size_t alignment_{};
    size_t used_mem_{};
    size_t max_mem_{};
    VkDevice device_;
    bool device_local_{};
    
    VkPhysicalDeviceMemoryProperties properties_;
    std::vector<std::shared_ptr<vk_chunk>> chunks_;

public:
    vk_allocator();
    vk_allocator(uint32_t device_id, const VkDevice& dev, const VkPhysicalDeviceMemoryProperties& properties,
        size_t chunk_size = 16384, const size_t max_device_cap = 0, bool make_device_local = true);

    vk_allocator(vk_allocator&&) noexcept ;
    vk_allocator& operator=(vk_allocator&&) noexcept;

    vk_block* allocate_buffer(size_t size, VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    vk_block* allocate_image();
    bool deallocate(const vk_block* blk);

    static VkBuffer& get_buffer(vk_block* blk) { return blk->buf; }
    static VkImage& get_image(vk_block* blk) { return blk->img; }
    void cleanup();
    ~vk_allocator();
};
