// device.h : Header file for your target.

#pragma once
#include "threads.h"
#include "allocator.h"
#include "job.h"

#include <vector>
#include <map>
#include <mutex>
#include <vulkan/vulkan.h>
#include <chrono>


inline uint32_t align_size(const uint32_t sz, const int n) { return sz + n - 1 & -n; }


inline int get_heap_index(const VkMemoryPropertyFlags& flags, const VkPhysicalDeviceMemoryProperties& properties)
{
	for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
	{
		if ((flags & properties.memoryTypes[i].propertyFlags) == flags)
			return properties.memoryTypes[i].heapIndex;
	}
	return -1;
}

struct device_create_info
{
	VkDeviceQueueCreateInfo* queue_info;
	VkDeviceCreateInfo device;
	VkCommandPoolCreateInfo cmd_pool;
};

class device
{
	device_create_info ci_;

	uint32_t device_id_{};
	uint32_t device_count_{};
	VkDevice logical_device_{};
	VkPhysicalDevice physical_device_{};
	VkPhysicalDeviceProperties2 device_properties_{};
	VkPhysicalDeviceFeatures device_features_{};
	VkPhysicalDeviceMemoryProperties memory_properties_{};
	VkPhysicalDeviceSubgroupProperties subgroup_properites_{};
	VkPhysicalDeviceLimits limits_{};
	std::vector<VkQueueFamilyProperties> queue_families_{};

	VkQueue cmd_queue_{};


	size_t max_device_memory_size_{};
	size_t max_host_device_memory_size_{};
	uint32_t max_work_group_count_[3]{0, 0, 0};
	uint32_t max_work_group_size_[3]{0, 0, 0};
	uint32_t max_subgroup_size_{};

	vk_allocator host_coherent_allocator_;
	vk_allocator device_allocator_;

	device_submission_thread* local_thread_;
	//cmd_thread_pool* thread_pool_;
	VkCommandPool main_cmd_pool_;
	uint32_t combined_queue_index_;

	float queue_priority_ = 0.8f;

public:
	device() = default;
	device(const uint32_t device_id, const uint32_t device_count, const VkInstance& instance, const VkPhysicalDevice& p_dev) : device(
		device_id, device_count, instance, p_dev, {}, {})	{ }

	device(const uint32_t device_id, const uint32_t device_count, const VkInstance& instance, const VkPhysicalDevice& p_dev,
	       const std::vector<const char*>& validation_layers,
	       const std::vector<const char*>& extension_layers) : device_id_(device_id), device_count_(device_count),physical_device_(p_dev),
	                                                           local_thread_(new device_submission_thread),  combined_queue_index_(0)
	{
		create(validation_layers, extension_layers);
		local_thread_->start_on(logical_device_, cmd_queue_, main_cmd_pool_);
		//thread_pool_->start_on(std::thread::hardware_concurrency() - device_count_, logical_device_, combined_queue_index_);
	}

	~device() { cleanup(); }

	void create(const std::vector<const char*>& validation_layers, const std::vector<const char*>& extension_layers){
		device_properties_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		subgroup_properites_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
		device_properties_.pNext = &subgroup_properites_;

		vkGetPhysicalDeviceProperties2(physical_device_, &device_properties_);
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
		//VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
		max_work_group_count_[0] = limits_.maxComputeWorkGroupCount[0];
		max_work_group_count_[1] = limits_.maxComputeWorkGroupCount[1];
		max_work_group_count_[2] = limits_.maxComputeWorkGroupCount[2];
		max_work_group_size_[0] = limits_.maxComputeWorkGroupSize[0];
		max_work_group_size_[1] = limits_.maxComputeWorkGroupSize[1];
		max_work_group_size_[2] = limits_.maxComputeWorkGroupSize[2];
		max_subgroup_size_ = subgroup_properites_.subgroupSize;
		device_name = device_properties_.properties.deviceName;
		std::cout << "device subgroup_size " << max_subgroup_size_ << "\n";
		uint32_t queue_family_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
		queue_families_.resize(queue_family_count);
		vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families_.data());

		
		std::vector<uint32_t> transfer_queues_idxes{};
		std::vector<uint32_t> compute_queues_idxes{};

		for (const auto& [queueFlags, queueCount, timestampValidBits, minImageTransferGranularity] : queue_families_)
		{
			if (queueFlags & VK_QUEUE_COMPUTE_BIT && queueFlags & VK_QUEUE_TRANSFER_BIT)
				break;
			combined_queue_index_++;
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
		command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		command_pool_create_info.pNext = nullptr;
		command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		command_pool_create_info.queueFamilyIndex = combined_queue_index_;

		VkResult result = vkCreateCommandPool(logical_device_, &command_pool_create_info, nullptr, &main_cmd_pool_);

		host_coherent_allocator_ = vk_allocator(device_id_, logical_device_, memory_properties_, 16384,
			max_host_device_memory_size_, false);
		device_allocator_ = vk_allocator(device_id_, logical_device_, memory_properties_, 16384, max_device_memory_size_,
			true);

	}

	void cleanup()
	{
		wait();
		host_coherent_allocator_.cleanup();
		device_allocator_.cleanup();

		vkDeviceWaitIdle(logical_device_);
		if (logical_device_ != nullptr)
			vkDestroyDevice(logical_device_, nullptr);
	}

	device(const device&) = delete;
	device& operator=(const device&) = delete;
	device(device&& d) noexcept;
	device& operator=(device&& d) noexcept;

	vk_block* malloc(const size_t size, const bool host = false){
		vk_block* block = nullptr;
		if (!host)
			block = device_allocator_.allocate_buffer(size);
		else if (block == nullptr || host)
			block = host_coherent_allocator_.allocate_buffer(size);
		return block;
	}

	void free(const vk_block* block)
	{
		if (host_coherent_allocator_.deallocate(block) || device_allocator_.deallocate(block))
			block = nullptr;
	}

	void trigger(job* j) const
	{
		if(j != nullptr)
		{
			auto* gd = j->generation_data(logical_device_, main_cmd_pool_);
			local_thread_->add_job(gd);
		}
	}

	void wait() const
	{
		//thread_pool_->wait();
		local_thread_->wait();
	}

	std::string device_name;
};



class device_group: public device
{
	
};