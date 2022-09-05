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
	VkDevice logical_device_{};
	VkPhysicalDevice physical_device_{};
	VkPhysicalDeviceProperties2 device_properties_{};
	VkPhysicalDeviceFeatures device_features_{};
	VkPhysicalDeviceMemoryProperties memory_properties_{};
	VkPhysicalDeviceSubgroupProperties subgroup_properites_{};
	VkPhysicalDeviceLimits limits_{};
	std::vector<VkQueueFamilyProperties> queue_families_{};

	VkQueue cmd_queue_{};
	VkCommandPool cmd_pool_{};

	size_t max_device_memory_size_{};
	size_t max_host_device_memory_size_{};
	uint32_t max_work_group_count_[3]{0, 0, 0};
	uint32_t max_work_group_size_[3]{0, 0, 0};
	uint32_t max_subgroup_size_{};

	std::mutex device_lock_;
	vk_allocator host_coherent_allocator_;
	vk_allocator device_allocator_;

	device_submission_thread* local_thread_;
	std::vector<VkSubmitInfo> submission_queue_;
	std::vector<job*> job_queue_;
	std::vector<long long unsigned> latencys_;
	float queue_priority_ = 0.8f;

public:
	device() = default;
	device(const uint32_t device_id, const VkInstance& instance, const VkPhysicalDevice& p_dev) : device(
		device_id, instance, p_dev, {}, {})	{ }

	device(const uint32_t device_id, const VkInstance& instance, const VkPhysicalDevice& p_dev,
	       const std::vector<const char*>& validation_layers,
	       const std::vector<const char*>& extension_layers) : device_id_(device_id), physical_device_(p_dev),
	                                                           local_thread_(new device_submission_thread)
	{
		create(validation_layers, extension_layers);
		local_thread_->start_on(logical_device_, cmd_queue_);
	}

	~device() { cleanup(); }

	void create(const std::vector<const char*>& validation_layers, const std::vector<const char*>& extension_layers);

	void cleanup()
	{
		wait();
		host_coherent_allocator_.cleanup();
		device_allocator_.cleanup();

		if (cmd_pool_ != nullptr)
			vkDestroyCommandPool(logical_device_, cmd_pool_, nullptr);

		vkDeviceWaitIdle(logical_device_);
		if (logical_device_ != nullptr)
			vkDestroyDevice(logical_device_, nullptr);
	}

	template <class P>
	job* make_job(const std::string& kernel_name, const std::string& kernel, const std::vector<vk_block*>& blks, P p, uint32_t group_size_x=1, uint32_t group_size_y=1, uint32_t group_size_z=1);

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

	void memcpy(const vk_block* src, const vk_block* dst, const uint32_t src_offset= 0, const uint32_t dst_offset=0)
	{
		auto* j = new copy(logical_device_, cmd_pool_, src, dst, src_offset, dst_offset);
		local_thread_->add_job({ j->get_submit_info() });
		job_queue_.push_back(j);
	}

	void wait() const { local_thread_->wait(); }
	std::string device_name;
};



template <class P>
job* device::make_job(const std::string& kernel_name, const std::string& kernel, const std::vector<vk_block*>& blks, P p, const uint32_t group_size_x,
                      const uint32_t group_size_y, const uint32_t group_size_z)
{
	auto* j = new job(static_cast<uint32_t>(blks.size()), kernel_name);

	//m_param.total = x.count();
	//m_group_x = static_cast<int>(alignSize(m_param.total, local_sz)) / local_sz;
	//if (m_group_x > max_compute_work_group_count)
	//m_group_x = max_compute_work_group_count - 1;

	j->set_group_size(group_size_x, group_size_y, group_size_z);
	j->set_shader(kernel);
	j->set_push_constants(&p, sizeof(P));

	for (uint32_t i = 0; i < blks.size(); ++i)
		j->bind_buffer(blks[i], i, 0);

	j->trigger(logical_device_, cmd_pool_);
	local_thread_->add_job({ j->get_submit_info() });
	job_queue_.push_back(j);
	return j;
}


class device_group: public device
{
	
};