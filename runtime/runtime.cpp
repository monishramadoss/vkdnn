#include "runtime.h"
#include "allocator.h"
#include "job.h"
#include "device.h"

#include <iostream>
#include <fstream>
#include <cstdio>
#include <vulkan/vulkan.h>

#ifdef NDEBUG
bool enableValidationLayers = false;
#else
//bool enableValidationLayers = true;
bool enableValidationLayers = true;
#endif

// Define a callback to capture the messages
VKAPI_ATTR VkBool32 VKAPI_CALL debug_messenger_callback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
	void* userData)
{
	std::cerr << "validation layer: " << callbackData->pMessage << std::endl;

	return false;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllcator, VkDebugUtilsMessengerEXT* pDebugMessenger)
{
	const auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(
		instance, "vkCreateDebugUtilsMessengerEXT"));
	if (func != nullptr)
		return func(instance, pCreateInfo, pAllcator, pDebugMessenger);
	return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks* pAllocator)
{
	const auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
		vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
	if (func != nullptr)
		func(instance, debugMessenger, pAllocator);
}

runtime& runtime::create()
{
	std::vector<const char*> enabledLayers;
	std::vector<const char*> enabledExtensions;

	if (enableValidationLayers)
	{
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		std::vector<VkLayerProperties> layerProperties(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data());


		bool foundLayer = false;
		//for (const VkLayerProperties& prop : layerProperties)
	    //	enabledLayers.push_back(prop.layerName);

		enabledLayers.push_back("VK_LAYER_KHRONOS_validation");

		uint32_t extensionCount;
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties> extensionProperties(extensionCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensionProperties.data());

		std::cout << "available layers:\n";
		for (const auto& layer : layerProperties)
			std::cout << '\t' << layer.layerName << '\n';

		std::cout << "available extensions:\n";
		for (const auto& extension : extensionProperties)
			std::cout << '\t' << extension.extensionName << '\n';
		
		
		enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
		enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);		
		enabledExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
		//enabledExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

		// for (const VkExtensionProperties& prop : extensionProperties)
		//     enabledExtensions.push_back(prop.extensionName);
	}

	VkApplicationInfo applicationInfo;
	applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	applicationInfo.pNext = nullptr;
	applicationInfo.pApplicationName = "Vulkan GPU Training BABY";
	applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	applicationInfo.pEngineName = "vulkan_dnn";
	applicationInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
	applicationInfo.apiVersion = VK_API_VERSION_1_2;

	VkInstanceCreateInfo instanceCreateInfo;
	instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instanceCreateInfo.pNext = nullptr;
	instanceCreateInfo.flags = 0;
//	instanceCreateInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
	instanceCreateInfo.pApplicationInfo = &applicationInfo;
	instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayers.size());
	instanceCreateInfo.ppEnabledLayerNames = enabledLayers.data();
	instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
	instanceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();

	VkResult result;
	VkDebugUtilsMessengerCreateInfoEXT messengerInfo = {};
	if (enableValidationLayers)
	{
		messengerInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		messengerInfo.pNext = nullptr;
		messengerInfo.messageSeverity =
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		messengerInfo.messageType =
			VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		messengerInfo.pfnUserCallback = debug_messenger_callback;
		messengerInfo.pUserData = nullptr;

		instanceCreateInfo.pNext = &messengerInfo;
		result = vkCreateInstance(&instanceCreateInfo, nullptr, &m_instance);
		if(result != VK_SUCCESS)
			throw std::runtime_error("failed to setup instance!");
		result = CreateDebugUtilsMessengerEXT(m_instance, &messengerInfo, nullptr, &m_debug_messenger);
		if (result != VK_SUCCESS)
			throw std::runtime_error("failed to set up debug messenger!");
	}
	else
	{
		instanceCreateInfo.pNext = nullptr;
		result = vkCreateInstance(&instanceCreateInfo, nullptr, &m_instance);
		if (result != VK_SUCCESS)
			throw std::runtime_error("failed to set up instance!");
	}

	result = vkEnumeratePhysicalDevices(m_instance, &m_device_count, nullptr);
	m_physical_devices.resize(m_device_count);
	vkEnumeratePhysicalDevices(m_instance, &m_device_count, m_physical_devices.data());


	size_t i = 0;
	for (const auto& pDev : m_physical_devices)
		m_devices.emplace_back(i++, m_instance, pDev); // , {}, { "VK_KHR_portability_subset" }));

	if (result != VK_SUCCESS)
		throw std::runtime_error("CANNOT CREATE LAYER");
	return *this;
}

runtime::runtime()
{
	create();
}

runtime::~runtime()
{	
	cleanup();
}

void runtime::cleanup() const
{
	for (auto device : m_devices)
		device.cleanup();
	if (enableValidationLayers)
		DestroyDebugUtilsMessengerEXT(m_instance, m_debug_messenger, nullptr);
	vkDestroyInstance(m_instance, nullptr);
}

device& runtime::get_device(size_t idx)
{
	return m_devices[idx];
}

vk_block** runtime::malloc(size_t size)
{
	vk_block* ptr = get_device().malloc(size);
	return &ptr;
}

void runtime::free(vk_block** blk)
{
	get_device().free(blk);
}


static std::vector<uint32_t> compile(const std::string shader_entry, const std::string& source, char* filename = nullptr);
constexpr float queuePriority = 1.0f;

inline uint32_t get_heap_index(const VkMemoryPropertyFlags& flags, const VkPhysicalDeviceMemoryProperties& properties)
{
	for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
	{
		if ((flags & properties.memoryTypes[i].propertyFlags) == flags)
			return properties.memoryTypes[i].heapIndex;
	}
	return -1;
}

device::device(size_t device_id, const VkInstance& instance, const VkPhysicalDevice& pDev,
	const std::vector<const char*>& validation_layers,
	const std::vector<const char*>& extension_layers) : m_device_id(device_id), m_physical_device(pDev)
{
	m_device_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	m_subgroup_properites.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
	m_device_properties.pNext = &m_subgroup_properites;

	vkGetPhysicalDeviceProperties(m_physical_device, &m_device_properties.properties);
	vkGetPhysicalDeviceMemoryProperties(m_physical_device, &m_memory_properties);
	vkGetPhysicalDeviceFeatures(m_physical_device, &m_device_features);

	const int heap_idx = get_heap_index(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_memory_properties);
	if (heap_idx == -1)
		throw std::runtime_error("Cannot find device heap");

	const VkPhysicalDeviceLimits limits = m_device_properties.properties.limits;
	m_max_device_memory_size = m_memory_properties.memoryHeaps[heap_idx].size;
	size_t buffer_copy_offset = limits.optimalBufferCopyOffsetAlignment;
	m_max_work_group_count[0] = limits.maxComputeWorkGroupCount[0];
	m_max_work_group_count[1] = limits.maxComputeWorkGroupCount[1];
	m_max_work_group_count[2] = limits.maxComputeWorkGroupCount[2];
	m_max_work_group_size[0] = limits.maxComputeWorkGroupSize[0];
	m_max_work_group_size[1] = limits.maxComputeWorkGroupSize[1];
	m_max_work_group_size[2] = limits.maxComputeWorkGroupSize[2];
	//m_max_subgroup_size = m_subgroup_properites.subgroupSize;

	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &queueFamilyCount, queueFamilies.data());

	uint32_t combinedQueueIndex = 0;
	for (const auto& queueFamily : queueFamilies)
	{
		if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT && queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT)
			break;
		combinedQueueIndex++;
	}

	VkDeviceQueueCreateInfo deviceQueueCreateInfo;
	deviceQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	deviceQueueCreateInfo.pNext = nullptr;
	deviceQueueCreateInfo.flags = 0;
	deviceQueueCreateInfo.queueCount = queueFamilies[combinedQueueIndex].queueCount;
	deviceQueueCreateInfo.queueFamilyIndex = combinedQueueIndex;
	deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

	VkDeviceCreateInfo deviceCreateInfo = {};
	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.pNext = nullptr;
	deviceCreateInfo.flags = 0;
	deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
	deviceCreateInfo.ppEnabledLayerNames = validation_layers.data();
	deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(extension_layers.size());
	deviceCreateInfo.ppEnabledExtensionNames = extension_layers.data();
	deviceCreateInfo.pEnabledFeatures = &m_device_features;
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
	VkResult result = vkCreateDevice(m_physical_device, &deviceCreateInfo, nullptr, &m_logical_device);

	vkGetDeviceQueue(m_logical_device, combinedQueueIndex, 0, &m_cmd_queue);

	VkCommandPoolCreateInfo commandPoolCreateInfo = {};
	commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	commandPoolCreateInfo.queueFamilyIndex = combinedQueueIndex;
	result = vkCreateCommandPool(m_logical_device, &commandPoolCreateInfo, nullptr, &m_cmd_pool);

	m_staging_allocator = vk_allocator(m_device_id, m_logical_device, m_memory_properties, 16384, false);
	m_device_allocator = vk_allocator(m_device_id, m_logical_device, m_memory_properties, 16384, true);

	if (result != VK_SUCCESS)
		std::cerr << "CANNOT CREATE DEVICE\n";
}

device::device(const device& other) : m_device_id(other.m_device_id), m_logical_device(other.m_logical_device),
m_physical_device(other.m_physical_device),
m_device_properties(other.m_device_properties),
m_device_features(other.m_device_features),
m_memory_properties(other.m_memory_properties),
m_subgroup_properites(other.m_subgroup_properites),
m_cmd_queue(other.m_cmd_queue),
m_cmd_pool(other.m_cmd_pool),
m_max_device_memory_size(other.m_max_device_memory_size),
m_max_subgroup_size(other.m_max_subgroup_size),
m_staging_allocator(other.m_staging_allocator),
m_device_allocator(other.m_device_allocator)
{
	std::memcpy(m_max_work_group_size, other.m_max_work_group_size, sizeof(uint32_t) * 3);
	std::memcpy(m_max_work_group_count, other.m_max_work_group_count, sizeof(uint32_t) * 3);
}

device& device::operator=(const device& other)
{
	m_device_id = other.m_device_id;
	m_logical_device = other.m_logical_device;
	m_physical_device = other.m_physical_device;
	m_device_properties = other.m_device_properties;
	m_device_features = other.m_device_features;
	m_memory_properties = other.m_memory_properties;
	m_subgroup_properites = other.m_subgroup_properites;
	m_cmd_queue = other.m_cmd_queue;
	m_cmd_pool = other.m_cmd_pool;
	m_max_device_memory_size = other.m_max_device_memory_size;
	m_max_subgroup_size = other.m_max_subgroup_size;
	std::memcpy(m_max_work_group_size, other.m_max_work_group_size, sizeof(uint32_t) * 3);
	std::memcpy(m_max_work_group_count, other.m_max_work_group_count, sizeof(uint32_t) * 3);
	m_staging_allocator = other.m_staging_allocator;
	m_device_allocator = other.m_device_allocator;
	return *this;
}

void device::cleanup()
{
	for (const auto& j : jobs)
		j->cleanup();
	m_staging_allocator.cleanup();
	m_device_allocator.cleanup();

	if(m_cmd_pool != nullptr)
		vkDestroyCommandPool(m_logical_device, m_cmd_pool, nullptr);

	//	vkDeviceWaitIdle(m_logical_device);
	if (m_logical_device !=  nullptr)
		vkDestroyDevice(m_logical_device, nullptr);
}

device::device() = default;

device::device(size_t device_id, const VkInstance& instance, const VkPhysicalDevice& pDev) : device(device_id, instance, pDev, {}, {})
{
}

device::~device() = default;


vk_block* device::malloc(size_t size)
{
	vk_block* host_block = m_staging_allocator.allocate_buffer(size);
	vk_block* device_block = m_device_allocator.allocate_buffer(size);
	return host_block;
}

void device::free(vk_block** block) const
{
	if (m_staging_allocator.deallocate(*block) || m_device_allocator.deallocate(*block))
		*block = nullptr;
}

void device::run() const
{
	VkFence fence;
	VkFenceCreateInfo fence_create_info_ = {};
	fence_create_info_.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fence_create_info_.flags = 0;

	for (const auto j : jobs)
	{
		auto submit_info = j->get_submit_info();
		vkCreateFence(m_logical_device, &fence_create_info_, nullptr, &fence);
		vkQueueSubmit(m_cmd_queue, 1, &submit_info, fence);
		vkWaitForFences(m_logical_device, 1, &fence, VK_TRUE, 100000);
		vkDestroyFence(m_logical_device, fence, nullptr);
	}

}


inline size_t alignSize(size_t sz, int n) { return (sz + n - 1) & -n; }

void job::bind_buffer(const vk_block* blk, uint32_t i)
{
	if (m_pipeline == nullptr)
		create_pipeline();

	m_set_buffers++;

	VkDescriptorBufferInfo descBufferInfo;
	descBufferInfo.buffer = blk->buf;
	descBufferInfo.offset = 0;
	descBufferInfo.range = blk->size;

	VkWriteDescriptorSet writeDescSet;
	writeDescSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescSet.pNext = nullptr;
	writeDescSet.dstArrayElement = 0;
	writeDescSet.dstSet = m_descriptor_set;
	writeDescSet.dstBinding = i;
	writeDescSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writeDescSet.descriptorCount = 1;
	writeDescSet.pBufferInfo = &descBufferInfo;
	writeDescSet.pTexelBufferView = nullptr;
	writeDescSet.pImageInfo = nullptr;
	vkUpdateDescriptorSets(m_device, 1, &writeDescSet, 0, nullptr);

	if (m_set_buffers == m_num_buffers)
	{
		record_pipeline();
		m_set_buffers = 0;
	}

}

VkSubmitInfo job::get_submit_info() const
{
	return m_submit_info;
}


void job::cleanup() const
{
	if (m_shader_module != nullptr)
		vkDestroyShaderModule(m_device, m_shader_module, nullptr);
	if (m_descriptor_pool != nullptr)
		vkDestroyDescriptorPool(m_device, m_descriptor_pool, nullptr);
	if (m_descriptor_set_layout != nullptr)
		vkDestroyDescriptorSetLayout(m_device, m_descriptor_set_layout, nullptr);
	if (m_pipeline != nullptr)
		vkDestroyPipeline(m_device, m_pipeline, nullptr);
	if (m_pipeline_layout != nullptr)
		vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
}

void job::create_pipeline()
{
	if (m_compiled_shader_code.empty())
		std::cerr << "ERROR CODE NOT FOUND\n";

	VkShaderModuleCreateInfo shaderModuleCreateInfo;
	shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shaderModuleCreateInfo.pNext = nullptr;
	shaderModuleCreateInfo.flags = 0;
	shaderModuleCreateInfo.pCode = m_compiled_shader_code.data();
	shaderModuleCreateInfo.codeSize = sizeof(uint32_t) * m_compiled_shader_code.size();
	VkResult result = vkCreateShaderModule(m_device, &shaderModuleCreateInfo, nullptr, &m_shader_module);

	VkPipelineShaderStageCreateInfo pipelineStageCreateInfo;
	pipelineStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	pipelineStageCreateInfo.pNext = nullptr;
	pipelineStageCreateInfo.flags = 0;
	pipelineStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	pipelineStageCreateInfo.module = m_shader_module;
	pipelineStageCreateInfo.pName = m_kernel_entry.c_str();
	pipelineStageCreateInfo.pSpecializationInfo = m_specialization_info;

	VkPushConstantRange pushConstantRanges;
	pushConstantRanges.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	pushConstantRanges.offset = 0;
	pushConstantRanges.size = m_push_constants_size;

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;
	pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutCreateInfo.pNext = nullptr;
	pipelineLayoutCreateInfo.flags = 0;
	pipelineLayoutCreateInfo.pushConstantRangeCount = m_push_constants_size ? 1 : 0;
	pipelineLayoutCreateInfo.pPushConstantRanges = m_push_constants_size ? &pushConstantRanges : nullptr;
	pipelineLayoutCreateInfo.setLayoutCount = 1;
	pipelineLayoutCreateInfo.pSetLayouts = &m_descriptor_set_layout;
	result = vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_pipeline_layout);

	VkComputePipelineCreateInfo computePipelineCreateInfo;
	computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	computePipelineCreateInfo.pNext = nullptr;
	computePipelineCreateInfo.flags = 0;
	computePipelineCreateInfo.stage = pipelineStageCreateInfo;
	computePipelineCreateInfo.layout = m_pipeline_layout;
	computePipelineCreateInfo.basePipelineIndex = -1;
	computePipelineCreateInfo.basePipelineHandle = nullptr;
	result = vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &m_pipeline);

	if (result != VK_SUCCESS)
		std::cerr << "FAILED TO CREATE PIPELINE\n";
}

void job::record_pipeline()
{
	VkCommandBufferBeginInfo cmdBufferBeginInfo;
	cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmdBufferBeginInfo.pNext = nullptr;
	cmdBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
	//beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
	cmdBufferBeginInfo.pInheritanceInfo = nullptr;
	VkResult result = vkBeginCommandBuffer(m_cmd_buffer, &cmdBufferBeginInfo);

	if (m_push_constants_size)
		vkCmdPushConstants(m_cmd_buffer, m_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, m_push_constants_size, m_push_constants);

	vkCmdBindPipeline(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
	vkCmdBindDescriptorSets(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 0, 1, &m_descriptor_set, 0, nullptr);

	vkCmdDispatch(m_cmd_buffer, m_groups[0], m_groups[1], m_groups[2]);

	result = vkEndCommandBuffer(m_cmd_buffer);

	if (result != VK_SUCCESS)
		std::cerr << "FAILED TO RECORD CMD BUFFER\n";

	m_submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	m_submit_info.commandBufferCount = 1;
	m_submit_info.pCommandBuffers = &m_cmd_buffer;
}

job::job() = default;

job::job(const VkDevice& dev, const VkCommandPool& cmd_pool, uint32_t num_buffers, std::string kernel_name) :
	m_device(dev), m_cmd_pool(cmd_pool), m_num_buffers(num_buffers), m_kernel_type(kernel_name), m_kernel_entry(kernel_name)
{
	create();
}

job::~job()
{
	cleanup();
}

void job::create()
{
	std::vector<VkDescriptorSetLayoutBinding> bindings(m_num_buffers);
	for (unsigned i = 0; i < m_num_buffers; i++)
	{
		bindings[i].binding = i;
		bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}

	VkDescriptorSetLayoutCreateInfo descriptor_set_info = {};
	descriptor_set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	descriptor_set_info.pNext = nullptr;
	descriptor_set_info.bindingCount = m_num_buffers;
	descriptor_set_info.pBindings = &bindings[0];
	VkResult result = vkCreateDescriptorSetLayout(m_device, &descriptor_set_info, nullptr, &m_descriptor_set_layout);

	VkDescriptorPoolSize pool_size;
	pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	pool_size.descriptorCount = m_num_buffers;

	VkDescriptorPoolCreateInfo descriptor_pool_info = {};
	descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptor_pool_info.pNext = nullptr;
	descriptor_pool_info.maxSets = 1;
	descriptor_pool_info.poolSizeCount = 1;
	descriptor_pool_info.pPoolSizes = &pool_size;
	result = vkCreateDescriptorPool(m_device, &descriptor_pool_info, nullptr, &m_descriptor_pool);

	VkDescriptorSetAllocateInfo descriptor_allocate_info;
	descriptor_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	descriptor_allocate_info.pNext = nullptr;
	descriptor_allocate_info.descriptorPool = m_descriptor_pool;
	descriptor_allocate_info.descriptorSetCount = 1;
	descriptor_allocate_info.pSetLayouts = &m_descriptor_set_layout;
	result = vkAllocateDescriptorSets(m_device, &descriptor_allocate_info, &m_descriptor_set);

	VkCommandBufferAllocateInfo command_buffer_alloc_info;
	command_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	command_buffer_alloc_info.pNext = nullptr;
	command_buffer_alloc_info.commandPool = m_cmd_pool;
	command_buffer_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	command_buffer_alloc_info.commandBufferCount = 1;
	result = vkAllocateCommandBuffers(m_device, &command_buffer_alloc_info, &m_cmd_buffer);

	if (result != VK_SUCCESS)
		std::cerr << "CANNOT CREATE LAYER\n";
}

void job::set_push_constants(void* params, uint32_t params_size)
{
	m_push_constants = params; m_push_constants_size = params_size;
}

void job::set_shader(const std::string& shader, VkSpecializationInfo* specialization_info)
{
	m_compiled_shader_code = compile(m_kernel_entry, shader);
	m_specialization_info = specialization_info;
}

void job::set_group_size(uint32_t x, uint32_t y, uint32_t z)
{
	m_groups[0] = x; m_groups[1] = y; m_groups[2] = z;
}



std::vector<uint32_t> compile(const std::string shader_entry, const std::string& source, char* filename)
{
	char tmp_filename_in[L_tmpnam];
	char tmp_filename_out[L_tmpnam];

	tmpnam(tmp_filename_in);
	tmpnam(tmp_filename_out);

	FILE* tmp_file;
	tmp_file = fopen(tmp_filename_in, "wb+");
	fputs(source.c_str(), tmp_file);
	fclose(tmp_file);

	tmp_file = fopen(tmp_filename_out, "wb+");
	fclose(tmp_file);

	const std::string cmd_str = std::string("glslangValidator -V " + std::string(tmp_filename_in) + " --entry-point " + shader_entry + " --source-entrypoint main -S comp -o " + tmp_filename_out);

	if (system(cmd_str.c_str()))
		throw std::runtime_error("Error running glslangValidator command");
	std::ifstream fileStream(tmp_filename_out, std::ios::binary);
	std::vector<char> buffer;
	buffer.insert(buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
	return { reinterpret_cast<uint32_t*>(buffer.data()), reinterpret_cast<uint32_t*>(buffer.data() + buffer.size()) };
}



inline bool isPowerOfTwo(size_t size)
{
	const bool ret = (size & (size - 1)) == 0;
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

inline size_t nextPowerOfTwo(size_t size)
{
	const size_t _size = power_ceil(size);
	return _size << 1;
}

inline int findMemoryTypeIndex(uint32_t memoryTypeBits,
	const VkPhysicalDeviceMemoryProperties& properties,
	bool shouldBeDeviceLocal)
{
	auto lambdaGetMemoryType = [&](VkMemoryPropertyFlags propertyFlags) -> int
	{
		for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
		{
			if ((memoryTypeBits & (1 << i)) && ((properties.memoryTypes[i].propertyFlags & propertyFlags) ==
				propertyFlags))
				return i;
		}
		return -1;
	};


	if (!shouldBeDeviceLocal)
	{
		VkMemoryPropertyFlags optimal = VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
		VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

		const int type = lambdaGetMemoryType(optimal);
		if (type == -1)
		{
			const int result = lambdaGetMemoryType(required);
			if (result == -1)
				throw std::runtime_error("Memory type does not find");
			return result;
		}
		return type;
	}
	return lambdaGetMemoryType(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

vk_chunk::vk_chunk(size_t device_id, const VkDevice& dev, size_t size, int memoryTypeIndex) : m_device_id(device_id),
	m_device(dev), m_ptr(nullptr), m_size(size), m_memory_type_index(memoryTypeIndex)
{
	VkMemoryAllocateInfo alloc_info{};
	alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	alloc_info.pNext = nullptr;
	alloc_info.allocationSize = size;
	alloc_info.memoryTypeIndex = memoryTypeIndex;
	const VkResult result = vkAllocateMemory(m_device, &alloc_info, nullptr, &m_memory);

	vk_block blk;
	blk.device_id = m_device_id;
	blk.size = size;
	blk.offset = 0;
	blk.free = true;
	blk.memory = m_memory;
	m_blocks.push_back(std::make_shared<vk_block>(blk));
	if (result != VK_SUCCESS)
		std::cerr << "CANNOT CREATE MEMORY\n";
}

bool vk_chunk::allocate(size_t size, size_t alignment, vk_block** blk)
{
	if (size > m_size)
		return false;
	for (size_t i = 0; i < m_blocks.size(); ++i)
	{
		if (m_blocks[i]->free)
		{
			size_t new_size = m_blocks[i]->size;
			if (m_blocks[i]->offset % alignment != 0)
				new_size -= alignment - m_blocks[i]->offset % alignment;
			if (new_size >= size)
			{
				m_blocks[i]->size = new_size;
				if (m_blocks[i]->offset % alignment != 0)
					m_blocks[i]->offset += alignment - m_blocks[i]->offset % alignment;
				if (m_blocks[i]->ptr != nullptr)
					m_blocks[i]->ptr = m_blocks[i]->ptr;
				if (m_blocks[i]->size == size)
				{
					m_blocks[i]->free = false;
					*blk = m_blocks[i].get();
					return true;
				}

				vk_block nextBlock;
				nextBlock.free = true;
				nextBlock.offset = m_blocks[i]->offset + size;
				nextBlock.size = m_blocks[i]->size - size;
				nextBlock.memory = m_memory;
				nextBlock.device_id = m_blocks[i]->device_id;
				m_blocks.push_back(std::make_shared<vk_block>(nextBlock));
				m_blocks[i]->size = size;
				m_blocks[i]->free = false;

				*blk = m_blocks[i].get();
				return true;
			}
		}
	}
	return false;
}


void vk_chunk::collate_delete()
{
	bool free_flag = false;
	size_t free_idx = 0;
	size_t idx = 0;
	while (idx < m_blocks.size())
	{
		if (m_blocks[idx]->free && !free_flag)
		{
			free_idx = idx;
			free_flag = true;
		}
		else if (m_blocks[idx]->free && free_flag)
		{
			if (idx - 1 == free_idx)
			{
				m_blocks[free_idx]->size += m_blocks[idx]->size;
				m_blocks.erase(m_blocks.begin() + idx);
			}
		}
		++idx;
	}
}


bool vk_chunk::deallocate(const vk_block* blk) const
{
	for (size_t i = 0; i < m_blocks.size(); ++i)
	{
		if (*blk == *m_blocks[i])
		{
			if (m_blocks[i]->buf != nullptr)
				vkDestroyBuffer(m_device, m_blocks[i]->buf, nullptr);
			else if (m_blocks[i]->img != nullptr)
				vkDestroyImage(m_device, m_blocks[i]->img, nullptr);
			m_blocks[i]->free = true;
			return true;
		}
	}
	return false;
}


void vk_chunk::cleanup()
{
	for (auto& blk : m_blocks)
	{
		bool deall = deallocate(blk.get());
	}
	m_blocks.clear();
	if (m_ptr != nullptr)
		vkUnmapMemory(m_device, m_memory);
	vkFreeMemory(m_device, m_memory, nullptr);
}

void vk_chunk::set_host_visible()
{
	m_ptr = nullptr;
	vkMapMemory(m_device, m_memory, 0, m_size, 0, &m_ptr);
	for (const auto& blk : m_blocks)
		blk->ptr = m_ptr;
}

vk_chunk::~vk_chunk()
{
	//cleanup();
}

vk_allocator::vk_allocator() : m_size(0), m_alignment(0), m_device(nullptr), m_properties()
{
}

vk_allocator::vk_allocator(size_t device_id, const VkDevice& dev, const VkPhysicalDeviceMemoryProperties& properties, size_t size,
	bool make_device_local) : m_device_id(device_id), m_size(size), m_device(dev), m_device_local(make_device_local), m_properties(properties)
{
	if (!isPowerOfTwo(size))
		throw std::runtime_error("Size must be in allocation of power 2");
}

vk_block* vk_allocator::allocate_buffer(size_t size)
{
	uint32_t memoryTypeIndex = 0;
	size_t alignment = 0;
	VkBuffer buffer{};
	VkBufferCreateInfo bufferCreateInfo{};
	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferCreateInfo.pNext = nullptr;
	bufferCreateInfo.flags = 0;
	bufferCreateInfo.size = size;
	bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	//bufferCreateInfo.queueFamilyIndexCount = 1;
	//bufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;
	/*
	 *requiredMemorySize += bufferMemoryRequirements.size;
	 *if (bufferMemoryRequirements.size % bufferMemoryRequirements.alignment != 0)
	 *	requiredMemorySize += bufferMemoryRequirements.alignment - bufferMemoryRequirements.size % bufferMemoryRequirements.alignment;
	 */
	const VkResult result = vkCreateBuffer(m_device, &bufferCreateInfo, nullptr, &buffer);
	VkMemoryRequirements bufferMemoryRequirements;
	vkGetBufferMemoryRequirements(m_device, buffer, &bufferMemoryRequirements);
	alignment = bufferMemoryRequirements.alignment;
	memoryTypeIndex = findMemoryTypeIndex(bufferMemoryRequirements.memoryTypeBits, m_properties, m_device_local);

	vk_block* blk = nullptr;
	for (const auto& chunk : m_chunks)
	{
		if (chunk->allocate(size, alignment, &blk))
		{
			blk->buf = buffer;
			vkBindBufferMemory(m_device, blk->buf, blk->memory, blk->offset);
			return blk;
		}
	}

	m_size = size > m_size ? nextPowerOfTwo(size) : m_size;
	m_chunks.push_back(std::make_shared<vk_chunk>(m_device_id, m_device, m_size, memoryTypeIndex));
	if (!m_chunks.back()->allocate(size, alignment, &blk))
		throw std::bad_alloc();

	if (!m_device_local)
		m_chunks.back()->set_host_visible();

	blk->buf = buffer;
	vkBindBufferMemory(m_device, blk->buf, blk->memory, blk->offset);
	return blk;
}

vk_block* vk_allocator::allocate_image()
{
	uint32_t memoryTypeIndex = 0;
	size_t alignment = 0;
	size_t size = 16;

	VkImage image{};
	VkImageCreateInfo imageCreateInfo{};
	imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageCreateInfo.pNext = nullptr;
	imageCreateInfo.flags = 0;
	imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	const VkResult result = vkCreateImage(m_device, &imageCreateInfo, nullptr, &image);
	VkMemoryRequirements imageMemoryRequirements;
	vkGetImageMemoryRequirements(m_device, image, &imageMemoryRequirements);
	alignment = imageMemoryRequirements.alignment;
	memoryTypeIndex = findMemoryTypeIndex(imageMemoryRequirements.memoryTypeBits, m_properties, m_device_local);

	vk_block* blk = nullptr;
	for (const auto& chunk : m_chunks)
	{
		if (chunk->allocate(size, alignment, &blk))
		{
			blk->img = image;
			return blk;
		}
	}

	m_size = size > m_size ? nextPowerOfTwo(size) : m_size;
	m_chunks.push_back(std::make_shared<vk_chunk>(m_device_id, m_device, m_size, memoryTypeIndex));
	if (!m_chunks.back()->allocate(size, alignment, &blk))
		throw std::bad_alloc();

	blk->img = image;
	if (result != VK_SUCCESS)
		std::cerr << "CANNOT CREATE IMAGE\n";
	return blk;
}

bool vk_allocator::deallocate(const vk_block* blk) const
{
	for (const auto& chunk : m_chunks)
	{
		if (chunk->deallocate(blk))
			return true;
	}
	return false;
}

void vk_allocator::cleanup()
{
	for (const auto& chunk : m_chunks)
		chunk->cleanup();
	m_chunks.clear();
}

vk_allocator::~vk_allocator() = default;
