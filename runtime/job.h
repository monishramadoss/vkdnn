#pragma once
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

class job final
{
	VkDevice m_device{};
	VkCommandPool m_cmd_pool{};
	VkShaderModule m_shader_module{};
	VkPipeline m_pipeline{};
	VkPipelineLayout m_pipeline_layout{};
	VkCommandBuffer m_cmd_buffer{};
	VkDescriptorPool m_descriptor_pool{};
	VkDescriptorSet m_descriptor_set{};
	VkDescriptorSetLayout m_descriptor_set_layout{};
	VkSubmitInfo m_submit_info{};

	uint32_t m_num_buffers{};
	uint32_t m_set_buffers{};

	std::string m_kernel_type;
	std::string m_kernel_entry;
	std::vector<uint32_t> m_compiled_shader_code;

	uint32_t m_groups[3]{1, 1, 1};
	uint32_t m_push_constants_size = 0;
	void* m_push_constants{};
	VkSpecializationInfo* m_specialization_info{};

	void create_pipeline();
	void record_pipeline();

public:
	job();
	job(const VkDevice& dev, const VkCommandPool& cmd_pool, uint32_t num_buffers, std::string kernel_name = "nop");
	~job();

	void create();

	void set_push_constants(void* params, uint32_t params_size);
	void set_shader(const std::string& shader, VkSpecializationInfo* specifialization_info = nullptr);
	void set_group_size(uint32_t x = 1, uint32_t y = 1, uint32_t z = 1);

	void cleanup() const;
	void bind_buffer(const vk_block* blk, uint32_t i);

	VkSubmitInfo get_submit_info() const;
};
