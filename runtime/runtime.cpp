#define _CRT_SECURE_NO_WARNINGS
#include "runtime.h"
#include "allocator.h"
#include "job.h"
#include "device.h"


#include <iostream>
#include <map>
#include <fstream>
#include <cstdio>
#include <vulkan/vulkan.h>



runtime* k_runtime = nullptr;
std::map<std::string, size_t> k_kernel_name_count;
runtime* init()
{
	if (k_runtime == nullptr)
		k_runtime = new runtime();
	return k_runtime;
}


device::device(device&& d) noexcept : device_id_(d.device_id_), logical_device_(d.logical_device_),
                                      physical_device_(d.physical_device_), device_properties_(d.device_properties_),
                                      device_features_(d.device_features_)
                                      , memory_properties_(d.memory_properties_),
                                      subgroup_properites_(d.subgroup_properites_),
                                      cmd_queue_(d.cmd_queue_),
                                      cmd_pool_(d.cmd_pool_), max_device_memory_size_(d.max_device_memory_size_),
                                      max_host_device_memory_size_(d.max_host_device_memory_size_),
                                      max_subgroup_size_(d.max_subgroup_size_)
                                      , host_coherent_allocator_(std::move(d.host_coherent_allocator_)),
                                      device_allocator_(std::move(d.device_allocator_)),
                                      local_thread_(d.local_thread_),
                                      queue_priority_(d.queue_priority_), device_name(d.device_name)
{
	d.wait();
	std::memcpy(max_work_group_size_, d.max_work_group_size_, sizeof(uint32_t) * 3);
	std::memcpy(max_work_group_count_, d.max_work_group_count_, sizeof(uint32_t) * 3);
}

device& device::operator=(device&& d) noexcept
{
	if (this != &d)
	{
		d.wait();
		device_id_ = d.device_id_;
		logical_device_ = d.logical_device_;
		physical_device_ = d.physical_device_;
		device_properties_ = d.device_properties_;
		device_features_ = d.device_features_;
		memory_properties_ = d.memory_properties_;
		subgroup_properites_ = d.subgroup_properites_;
		cmd_queue_ = d.cmd_queue_;
		cmd_pool_ = d.cmd_pool_;
		max_device_memory_size_ = d.max_device_memory_size_;
		max_host_device_memory_size_ = d.max_host_device_memory_size_;
		max_subgroup_size_ = d.max_subgroup_size_;
		std::memcpy(max_work_group_size_, d.max_work_group_size_, sizeof(uint32_t) * 3);
		std::memcpy(max_work_group_count_, d.max_work_group_count_, sizeof(uint32_t) * 3);
		host_coherent_allocator_ = std::move(d.host_coherent_allocator_);
		device_allocator_ = std::move(d.device_allocator_);
		device_name = d.device_name;
		queue_priority_ = d.queue_priority_;
	}
	return *this;
}

void device::create(const std::vector<const char*>& validation_layers, const std::vector<const char*>& extension_layers)
{
	device_properties_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	subgroup_properites_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
	device_properties_.pNext = &subgroup_properites_;

	vkGetPhysicalDeviceProperties(physical_device_, &device_properties_.properties);
	vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties_);
	vkGetPhysicalDeviceFeatures(physical_device_, &device_features_);

	const int device_heap_idx = get_heap_index(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory_properties_);
	const int host_heap_idx = get_heap_index(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	                                         memory_properties_);
	if (device_heap_idx == -1 || host_heap_idx == -1)
		throw std::runtime_error("Cannot find device heap");

	limits_ = device_properties_.properties.limits;
	max_device_memory_size_ = memory_properties_.memoryHeaps[device_heap_idx].size;
	max_host_device_memory_size_ = memory_properties_.memoryHeaps[host_heap_idx].size;
	//size_t buffer_copy_alignment = limits_.optimalBufferCopyOffsetAlignment;
	//size_t buffer_alignment = limits_.minStorageBufferOffsetAlignment;

	max_work_group_count_[0] = limits_.maxComputeWorkGroupCount[0];
	max_work_group_count_[1] = limits_.maxComputeWorkGroupCount[1];
	max_work_group_count_[2] = limits_.maxComputeWorkGroupCount[2];
	max_work_group_size_[0] = limits_.maxComputeWorkGroupSize[0];
	max_work_group_size_[1] = limits_.maxComputeWorkGroupSize[1];
	max_work_group_size_[2] = limits_.maxComputeWorkGroupSize[2];
	max_subgroup_size_ = subgroup_properites_.subgroupSize;
	device_name = device_properties_.properties.deviceName;

	uint32_t queue_family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
	queue_families_.resize(queue_family_count);
	vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families_.data());

	uint32_t combined_queue_index = 0;
	std::vector<uint32_t> transfer_queues_idxes{};
	std::vector<uint32_t> compute_queues_idxes{};

	for (const auto& [queueFlags, queueCount, timestampValidBits, minImageTransferGranularity] : queue_families_)
	{
		if (queueFlags & VK_QUEUE_COMPUTE_BIT && queueFlags & VK_QUEUE_TRANSFER_BIT)
			break;
		combined_queue_index++;
	}

	uint32_t idx = 0;
	for (const auto& [queueFlags, queueCount, timestampValidBits, minImageTransferGranularity] : queue_families_)
	{
		if (queueFlags & VK_QUEUE_COMPUTE_BIT)
			compute_queues_idxes.push_back(idx);
		else if (queueFlags & VK_QUEUE_TRANSFER_BIT && !(queueFlags & VK_QUEUE_COMPUTE_BIT))
			transfer_queues_idxes.push_back(idx);
		idx++;
	}

	const auto compute_queue_count = queue_families_[compute_queues_idxes.front()].queueCount;
	std::vector<float> compute_queue_priorities(compute_queue_count);
	for (uint32_t i = 0; i < compute_queue_count; ++i)
		compute_queue_priorities[i] = 1 - i / static_cast<float>(compute_queue_count);

	VkDeviceQueueCreateInfo device_compute_queue_create_info{};
	device_compute_queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	device_compute_queue_create_info.pNext = nullptr;
	device_compute_queue_create_info.flags = 0;
	device_compute_queue_create_info.queueCount = compute_queue_count;
	device_compute_queue_create_info.queueFamilyIndex = compute_queues_idxes.front();
	device_compute_queue_create_info.pQueuePriorities = compute_queue_priorities.data();

	if (!transfer_queues_idxes.empty())
	{
		const auto transfer_queue_count = queue_families_[transfer_queues_idxes.front()].queueCount;
		std::vector<float> transfer_queue_priortities(transfer_queue_count);
		for (uint32_t i = 0; i < transfer_queue_count; ++i)
			transfer_queue_priortities[i] = 1 - i / static_cast<float>(transfer_queue_count);

		VkDeviceQueueCreateInfo transfer_compute_queue_create_info{};
		transfer_compute_queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		transfer_compute_queue_create_info.pNext = nullptr;
		transfer_compute_queue_create_info.flags = 0;
		transfer_compute_queue_create_info.queueCount = transfer_queue_count;
		transfer_compute_queue_create_info.queueFamilyIndex = transfer_queues_idxes.front();
		transfer_compute_queue_create_info.pQueuePriorities = transfer_queue_priortities.data();
	}

	ci_.queue_info = new VkDeviceQueueCreateInfo[1];
	ci_.queue_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	ci_.queue_info[0].pNext = nullptr;
	ci_.queue_info[0].flags = 0;
	ci_.queue_info[0].queueCount = 1;
	ci_.queue_info[0].queueFamilyIndex = 0;
	ci_.queue_info[0].pQueuePriorities = &queue_priority_;

	ci_.device.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	ci_.device.pNext = nullptr;
	ci_.device.flags = 0;
	ci_.device.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
	ci_.device.ppEnabledLayerNames = validation_layers.data();
	ci_.device.enabledExtensionCount = static_cast<uint32_t>(extension_layers.size());
	ci_.device.ppEnabledExtensionNames = extension_layers.data();
	ci_.device.pEnabledFeatures = &device_features_;
	ci_.device.queueCreateInfoCount = 1;
	ci_.device.pQueueCreateInfos = ci_.queue_info;

	vkCreateDevice(physical_device_, &ci_.device, nullptr, &logical_device_);
	vkGetDeviceQueue(logical_device_, 0, 0, &cmd_queue_);

	VkCommandPoolCreateInfo command_pool_create_info{};
	ci_.cmd_pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	ci_.cmd_pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	ci_.cmd_pool.queueFamilyIndex = combined_queue_index;
	const VkResult result = vkCreateCommandPool(logical_device_, &ci_.cmd_pool, nullptr, &cmd_pool_);

	host_coherent_allocator_ = vk_allocator(device_id_, logical_device_, memory_properties_, 16384,
	                                        max_host_device_memory_size_, false);
	device_allocator_ = vk_allocator(device_id_, logical_device_, memory_properties_, 16384, max_device_memory_size_,
	                                 true);

	if (result != VK_SUCCESS)
		std::cerr << "CANNOT CREATE DEVICE\n";
}




inline bool is_power_of_two(const size_t size)
{
	const bool ret = (size & size - 1) == 0;
	return ret;
}

inline size_t power_floor(size_t x)
{
	size_t power = 1;
	while (x >>= 1) power <<= 1;
	return power;
}

inline size_t power_ceil(size_t x)
{
	if (x <= 1) return 1;
	size_t power = 2;
	x--;
	while (x >>= 1) power <<= 1;
	return power;
}

inline size_t next_power_of_two(const size_t size)
{
	const size_t s = power_ceil(size);
	return s << 1;
}

inline int find_memory_type_index(const uint32_t memory_type_bits,
                               const VkPhysicalDeviceMemoryProperties& properties,
                               const bool should_be_device_local)
{
	auto lambda_get_memory_type = [&](const VkMemoryPropertyFlags property_flags) -> int
	{
		for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
		{
			if (memory_type_bits & 1 << i && (properties.memoryTypes[i].propertyFlags & property_flags) ==
				property_flags)
				return i;
		}
		return -1;
	};


	if (!should_be_device_local)
	{
		constexpr VkMemoryPropertyFlags optimal = VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
		constexpr VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

		const int type = lambda_get_memory_type(optimal);
		if (type == -1)
		{
			const int result = lambda_get_memory_type(required);
			if (result == -1)
				throw std::runtime_error("Memory type does not find");
			return result;
		}
		return type;
	}
	return lambda_get_memory_type(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

bool vk_chunk::allocate(const size_t size, const size_t alignment, vk_block** blk)
{
	if (size_ < size)
		return false;
	for (size_t i = 0; i < blocks_.size(); ++i)
	{
		if (blocks_[i]->free)
		{
			size_t new_size = blocks_[i]->size;
			if (blocks_[i]->offset % alignment != 0)
				new_size -= alignment - blocks_[i]->offset % alignment;
			if (new_size >= size)
			{
				blocks_[i]->size = new_size;
				if (blocks_[i]->offset % alignment != 0)
					blocks_[i]->offset += alignment - blocks_[i]->offset % alignment;
				if (blocks_[i]->size == size)
				{
					blocks_[i]->free = false;
					*blk = blocks_[i].get();
					return true;
				}

				vk_block next_block;
				next_block.free = true;
				next_block.offset = blocks_[i]->offset + size;
				next_block.size = blocks_[i]->size - size;
				next_block.memory = memory_;
				next_block.device_id = blocks_[i]->device_id;
				next_block.on_device = blocks_[i]->on_device;
				if (!device_local_ && ptr_ != nullptr)
					next_block.ptr = next_block.offset + static_cast<char*>(ptr_);
				blocks_.push_back(std::make_shared<vk_block>(next_block));
				blocks_[i]->size = size;
				blocks_[i]->free = false;

				*blk = blocks_[i].get();
				return true;
			}
		}
	}
	return false;
}


void vk_chunk::collate_delete()
{
	bool free_flag = false;
	uint32_t free_idx = 0;
	uint32_t idx = 0;
	while (idx < blocks_.size())
	{
		if (blocks_[idx]->free && !free_flag)
		{
			free_idx = idx;
			free_flag = true;
		}
		else if (blocks_[idx]->free && free_flag)
		{
			if (idx - 1 == free_idx)
			{
				blocks_[free_idx]->size += blocks_[idx]->size;
				blocks_.erase(idx + blocks_.begin());
			}
		}
		++idx;
	}
}

vk_chunk::vk_chunk(const uint32_t device_id, const VkDevice& dev, const size_t size, const int memory_type_index):
	device_id_(device_id), device_(dev), ptr_(nullptr), size_(size), memory_type_index_(memory_type_index),
	device_local_(false)
{
	VkMemoryAllocateInfo alloc_info{};
	alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	alloc_info.pNext = nullptr;
	alloc_info.allocationSize = size;
	alloc_info.memoryTypeIndex = memory_type_index;
	const VkResult result = vkAllocateMemory(device_, &alloc_info, nullptr, &memory_);

	vk_block blk;
	blk.device_id = device_id_;
	blk.size = size;
	blk.offset = 0;
	blk.free = true;
	blk.memory = memory_;
	blk.on_device = true;
	blocks_.push_back(std::make_shared<vk_block>(blk));
	if (result != VK_SUCCESS)
		std::cerr << "CANNOT CREATE MEMORY\n";
}


bool vk_chunk::deallocate(const vk_block* blk) const
{
	return std::ranges::any_of(blocks_, [&](const std::shared_ptr<vk_block>& block){
			if(*blk == *block && !block->free)
			{
				if (block->buf != VK_NULL_HANDLE)
					vkDestroyBuffer(device_, block->buf, nullptr);
				else if (block->img != VK_NULL_HANDLE)
					vkDestroyImage(device_, block->img, nullptr);
				block->free = true;
				return true;
			}
			return false;
		}
	);

}

VkDeviceMemory vk_chunk::get_memory() const { return memory_; }


void vk_chunk::cleanup()
{
	if (ptr_ != nullptr)
		vkUnmapMemory(device_, memory_);
	for (auto& blk : blocks_)
		bool deall = deallocate(blk.get());
	blocks_.clear();
	vkFreeMemory(device_, memory_, nullptr);
}

void vk_chunk::set_host_visible()
{
	ptr_ = nullptr;
	vkMapMemory(device_, memory_, 0, size_, 0, &ptr_);
	for (const auto& blk : blocks_)
	{
		blk->ptr = blk->offset + static_cast<char*>(ptr_);
		blk->on_device = false;
	}
}

vk_chunk::~vk_chunk() = default;

vk_allocator::vk_allocator() : device_id_(0), chunk_size_(0), alignment_(0), used_mem_(0), max_mem_(0),
                               device_(nullptr), device_local_(false), properties_()

{
}

vk_allocator::vk_allocator(const uint32_t device_id, const VkDevice& dev,
                           const VkPhysicalDeviceMemoryProperties& properties,
                           const size_t chunk_size, const size_t max_device_cap, const bool make_device_local) :
	device_id_(device_id),
	chunk_size_(chunk_size),
	max_mem_(max_device_cap),
	device_(dev),
	device_local_(make_device_local),
	properties_(properties)
{
	if (!is_power_of_two(chunk_size))
		throw std::runtime_error("Size must be in allocation of power 2");
}

vk_allocator::vk_allocator(vk_allocator&& vka) noexcept: device_id_(vka.device_id_), chunk_size_(vka.chunk_size_),
                                                         used_mem_(vka.used_mem_), max_mem_(vka.max_mem_),
                                                         device_(vka.device_), device_local_(vka.device_local_),
                                                         properties_(vka.properties_), chunks_(std::move(vka.chunks_))
{
}

vk_allocator& vk_allocator::operator=(vk_allocator&& vka) noexcept
{
	if (this != &vka)
	{
		device_id_ = vka.device_id_;
		chunk_size_ = vka.chunk_size_;
		used_mem_ = vka.used_mem_;
		max_mem_ = vka.max_mem_;
		device_ = vka.device_;
		device_local_ = vka.device_local_;
		properties_ = vka.properties_;
		chunks_ = std::move(vka.chunks_);
	}
	return *this;
}

vk_block* vk_allocator::allocate_buffer(const size_t size, const VkBufferUsageFlags usage)
{
	if (size + used_mem_ > max_mem_)
		return nullptr;

	VkBuffer buffer{};
	VkBufferCreateInfo buffer_create_info{};
	buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_create_info.pNext = nullptr;
	buffer_create_info.flags = 0;
	buffer_create_info.size = size;
	buffer_create_info.usage = usage; 
	buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // VK_SHARING_MODE_CONCURRENT;
	buffer_create_info.queueFamilyIndexCount = 0;
	buffer_create_info.pQueueFamilyIndices = nullptr;

	const VkResult result = vkCreateBuffer(device_, &buffer_create_info, nullptr, &buffer);
	VkMemoryRequirements buffer_memory_requirements;
	vkGetBufferMemoryRequirements(device_, buffer, &buffer_memory_requirements);
	const size_t alignment = buffer_memory_requirements.alignment;
	uint32_t memory_type_index = find_memory_type_index(buffer_memory_requirements.memoryTypeBits, properties_,
	                                                 device_local_);

	vk_block* blk = nullptr;
	for (const auto& chunk : chunks_)
	{
		if (chunk->allocate(size, alignment, &blk))
		{
			blk->buf = buffer;
			vkBindBufferMemory(device_, blk->buf, blk->memory, blk->offset);
			used_mem_ += size;
			return blk;
		}
	}

	chunk_size_ = size > chunk_size_ ? next_power_of_two(size) : chunk_size_;
	chunks_.push_back(std::make_shared<vk_chunk>(device_id_, device_, chunk_size_, memory_type_index));
	if (!chunks_.back()->allocate(size, alignment, &blk))
		return nullptr;

	if (!device_local_)
		chunks_.back()->set_host_visible();

	blk->buf = buffer;
	vkBindBufferMemory(device_, blk->buf, blk->memory, blk->offset);

	used_mem_ += size;
	return blk;
}

vk_block* vk_allocator::allocate_image()
{
	constexpr uint32_t size = 16;
	if (size + used_mem_ > max_mem_)
		return nullptr;

	VkImage image{};

	VkImageCreateInfo image_create_info{};
	image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image_create_info.pNext = nullptr;
	image_create_info.flags = 0;
	image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	image_create_info.queueFamilyIndexCount = 0;
	image_create_info.pQueueFamilyIndices = nullptr;
	const VkResult result = vkCreateImage(device_, &image_create_info, nullptr, &image);
	VkMemoryRequirements image_memory_requirements;
	vkGetImageMemoryRequirements(device_, image, &image_memory_requirements);
	const size_t alignment = image_memory_requirements.alignment;
	uint32_t memory_type_index = find_memory_type_index(image_memory_requirements.memoryTypeBits, properties_,
	                                                 device_local_);

	vk_block* blk = nullptr;
	for (const auto& chunk : chunks_)
	{
		if (chunk->allocate(size, alignment, &blk))
		{
			blk->img = image;
			used_mem_ += size;
			return blk;
		}
	}

	chunk_size_ = size > chunk_size_ ? next_power_of_two(size) : chunk_size_;
	chunks_.push_back(std::make_shared<vk_chunk>(device_id_, device_, chunk_size_, memory_type_index));
	if (!chunks_.back()->allocate(size, alignment, &blk))
		throw std::bad_alloc();

	blk->img = image;
	used_mem_ += size;
	return blk;
}

bool vk_allocator::deallocate(const vk_block* blk)
{
	if (std::ranges::any_of(chunks_, [&](const std::shared_ptr<vk_chunk>& chunk) { return chunk->deallocate(blk); }))
	{
		used_mem_ -= blk->size;
		return true;
	}

	return false;
}

void vk_allocator::cleanup()
{
	for (const auto& chunk : chunks_)
		chunk->cleanup();
	chunks_.clear();
}

vk_allocator::~vk_allocator() = default;
