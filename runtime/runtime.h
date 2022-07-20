// runtime.h : Header file for your target.
#pragma once
#include "device.h"

#include <vector>
#include <vulkan/vulkan.h>

class runtime
{
	VkInstance m_instance = nullptr;
	std::vector<VkPhysicalDevice> m_physical_devices{};
	std::vector<device> m_devices{};
	uint32_t m_device_count = 0;
	VkDebugUtilsMessengerEXT m_debug_messenger{};

	void cleanup() const;
public:
	runtime& create();
	runtime();
	~runtime();

	[[nodiscard]] device& get_device(size_t idx = 0);

	vk_block** malloc(size_t size);
	void free(vk_block** blk);

};

static runtime kRuntime = runtime();

//static runtime kRuntime; 
