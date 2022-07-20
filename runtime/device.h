#pragma once
#include "allocator.h"
#include "job.h"

#include <vector>
#include <map>
#include <mutex>
#include <vulkan/vulkan.h>



class device final
{
	size_t m_device_id{};
	VkDevice m_logical_device{};
	VkPhysicalDevice m_physical_device{};
	VkPhysicalDeviceProperties2 m_device_properties{};
	VkPhysicalDeviceFeatures m_device_features{};
	VkPhysicalDeviceMemoryProperties m_memory_properties{};
	VkPhysicalDeviceSubgroupProperties m_subgroup_properites{};
	VkQueue m_cmd_queue{};
	VkCommandPool m_cmd_pool{};

	size_t m_max_device_memory_size{};
	uint32_t m_max_work_group_count[3]{};
	uint32_t m_max_work_group_size[3]{};
	uint32_t m_max_subgroup_size{};

	std::mutex device_lock;
	vk_allocator m_staging_allocator{};
	vk_allocator m_device_allocator{};

	std::vector<job*> jobs;
	
public:
	device();
	device(size_t device_id, const VkInstance& instance, const VkPhysicalDevice& pDev);
	device(size_t device_id, const VkInstance& instance, const VkPhysicalDevice& pDev, const std::vector<const char*>& validation_layers,
	       const std::vector<const char*>& extension_layers);
	~device();

	void cleanup();
	job* make_job(const std::string& kernel_name, const std::vector<vk_block*>& blks, void* p, size_t param_size);

	// The copy operations are implicitly deleted, but you can
	// spell that out explicitly if you want:
	device(const device&);
	device& operator=(const device&);

	vk_block* malloc(size_t size);
	void free(vk_block** block) const;

	void run() const;
	std::map<std::string, std::string> kernel_mapping{};
};
