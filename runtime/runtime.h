// runtime.h : Header file for your target.

#pragma once
#include "device.h"
#include "threads.h"
#include <vulkan/vulkan.h>

#ifdef NDEBUG
inline bool enable_validation_layers = false;
#else
inline bool enable_validation_layers = true;
#endif


// Define a callback to capture the messages
VKAPI_ATTR inline VkBool32 VKAPI_CALL debug_messenger_callback(
	VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
	VkDebugUtilsMessageTypeFlagsEXT message_type,
	const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
	void* user_data)
{
	//std::cerr << "validation layer: " << callbackData->pMessage << std::endl;
	return false;
}

inline VkResult create_debug_utils_messenger_ext(const VkInstance instance,
                                                 const VkDebugUtilsMessengerCreateInfoEXT* p_create_info,
                                                 const VkAllocationCallbacks* p_allcator,
                                                 VkDebugUtilsMessengerEXT* p_debug_messenger)
{
	if (const auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(
		instance, "vkCreateDebugUtilsMessengerEXT")); func != nullptr)
		return func(instance, p_create_info, p_allcator, p_debug_messenger);
	return VK_ERROR_EXTENSION_NOT_PRESENT;
}

inline void destroy_debug_utils_messenger_ext(const VkInstance instance, const VkDebugUtilsMessengerEXT debug_messenger,
                                              const VkAllocationCallbacks* p_allocator)
{
	if (const auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
		vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT")); func != nullptr)
		func(instance, debug_messenger, p_allocator);
}

struct runtime_create_info
{
	VkApplicationInfo application;
	VkInstanceCreateInfo instance;
	VkDebugUtilsMessengerCreateInfoEXT messenger;
};

class runtime final
{
	runtime_create_info ci_;
	VkInstance instance_{};
	VkPhysicalDevice* physical_devices_{};
	device** devices_{};
	uint32_t device_count_{};
	VkDebugUtilsMessengerEXT debug_messenger_{};
	//cmd_thread_pool thread_pool_;

	void cleanup() const
	{
		for (uint32_t i = 0; i < device_count_; ++i)
			devices_[i]->cleanup();
		if (enable_validation_layers)
			destroy_debug_utils_messenger_ext(instance_, debug_messenger_, nullptr);
		delete[] physical_devices_;

		vkDestroyInstance(instance_, nullptr);
	}

public:
	runtime& create() {
		std::vector<const char*> enabled_layers;
		std::vector<const char*> enabled_extensions;

		if (enable_validation_layers)
		{
			uint32_t layer_count;
			vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
			std::vector<VkLayerProperties> layer_properties(layer_count);
			vkEnumerateInstanceLayerProperties(&layer_count, layer_properties.data());


			bool foundLayer = false;
			//for (const VkLayerProperties& prop : layerProperties)
			//	enabledLayers.push_back(prop.layerName);

			enabled_layers.push_back("VK_LAYER_KHRONOS_validation");

			uint32_t extension_count;
			vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
			std::vector<VkExtensionProperties> extension_properties(extension_count);
			vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extension_properties.data());
			enabled_extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
			enabled_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
			enabled_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
			//enabledExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

			// for (const VkExtensionProperties& prop : extensionProperties)
			//     enabledExtensions.push_back(prop.extensionName);
		}

		ci_.application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		ci_.application.pNext = nullptr;
		ci_.application.pApplicationName = "Vulkan GPU Training BABY";
		ci_.application.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		ci_.application.pEngineName = "vulkan_dnn";
		ci_.application.engineVersion = VK_MAKE_VERSION(0, 1, 0);
		ci_.application.apiVersion = VK_API_VERSION_1_2;

		VkInstanceCreateInfo instance_create_info{};
		ci_.instance.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		ci_.instance.pNext = nullptr;
		ci_.instance.flags = 0;
		ci_.instance.pApplicationInfo = &ci_.application;
		ci_.instance.enabledLayerCount = static_cast<uint32_t>(enabled_layers.size());
		ci_.instance.ppEnabledLayerNames = enabled_layers.data();
		ci_.instance.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
		ci_.instance.ppEnabledExtensionNames = enabled_extensions.data();

		VkResult result;
		;
		if (enable_validation_layers)
		{
			ci_.messenger.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			ci_.messenger.pNext = nullptr;
			ci_.messenger.messageSeverity =
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			ci_.messenger.messageType =
				VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			ci_.messenger.pfnUserCallback = debug_messenger_callback;
			ci_.messenger.pUserData = nullptr;

			ci_.instance.pNext = &ci_.messenger;
			result = vkCreateInstance(&ci_.instance, nullptr, &instance_);
			if (result != VK_SUCCESS)
				throw std::runtime_error("failed to setup instance!");
			result = create_debug_utils_messenger_ext(instance_, &ci_.messenger, nullptr, &debug_messenger_);
			if (result != VK_SUCCESS)
				throw std::runtime_error("failed to set up debug messenger!");
		}
		else
		{
			ci_.instance.pNext = nullptr;
			result = vkCreateInstance(&ci_.instance, nullptr, &instance_);
			if (result != VK_SUCCESS)
				throw std::runtime_error("failed to set up instance!");
		}

		result = vkEnumeratePhysicalDevices(instance_, &device_count_, nullptr);
		if (device_count_ > 0)
		{
			physical_devices_ = new VkPhysicalDevice[device_count_];
			devices_ = new device * [device_count_];
			result = vkEnumeratePhysicalDevices(instance_, &device_count_, physical_devices_);
			for (uint32_t i = 0; i < device_count_; ++i)
				devices_[i] = new device(i, instance_, physical_devices_[i]);
		}
		return *this;
	}
	runtime() {  create(); }
	~runtime() { cleanup(); }

	[[nodiscard]] device& get_device(const uint32_t idx = 0) const { return *devices_[idx]; }

	[[nodiscard]] vk_block* malloc(const size_t size, const bool host) const {
		for (auto i = 0; i < device_count_; ++i)
		{
			if (auto* blk = devices_[i]->malloc(size, host))
				return blk;
		}
		throw std::runtime_error("CANNOT ALLOCATE DATA");
	}

	void free(const vk_block* blk) const
	{
		if (blk == nullptr || blk->size == 0)
			return;
		get_device(blk->device_id).free(blk);
	}

	void wait() const{
		for (auto i = 0; i < device_count_; ++i)
			devices_[i]->wait();
	}
};

extern runtime* k_runtime;
runtime* init();
