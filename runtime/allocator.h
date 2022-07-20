#pragma once
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>


struct vk_block
{
	size_t size{};
	size_t offset{};
	size_t device_id{};
	bool free{};
	void* ptr = nullptr;
	VkDeviceMemory memory{};
	VkBuffer buf = nullptr;
	VkImage img = nullptr;

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
	size_t m_device_id;
	VkDevice m_device{};
	VkDeviceMemory m_memory{};
	void* m_ptr{};

	size_t m_size{};
	std::vector<std::shared_ptr<vk_block>> m_blocks;
	int m_memory_type_index{};
	bool device_local{};

	void collate_delete();
public:
	vk_chunk(size_t device_id, const VkDevice& dev, size_t size, int memoryTypeIndex);
	bool allocate(size_t size, size_t alignment, vk_block** blk);
	bool deallocate(const vk_block* blk) const;
	[[nodiscard]] VkDeviceMemory get_memory() const { return m_memory; }
	void cleanup();
	void set_host_visible();
	~vk_chunk();
};

class vk_allocator final
{
	size_t m_device_id;
	size_t m_size{};
	size_t m_alignment{};
	VkDevice m_device{};
	bool m_device_local{};
	VkPhysicalDeviceMemoryProperties m_properties{};
	std::vector<std::shared_ptr<vk_chunk>> m_chunks;

public:
	vk_allocator();
	vk_allocator(size_t device_id, const VkDevice& dev, const VkPhysicalDeviceMemoryProperties& properties, size_t size = 16384,
	             bool make_device_local = true);
	vk_block* allocate_buffer(size_t size);
	vk_block* allocate_image();
	bool deallocate(const vk_block* blk) const;

	static VkBuffer& get_buffer(vk_block* blk) { return blk->buf; }
	static VkImage& get_image(vk_block* blk) { return blk->img; }
	void cleanup();
	~vk_allocator();
};
