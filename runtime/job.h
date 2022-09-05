// job.h : Header file for your target.

#pragma once
#include <string>
#include <map>
#include <vector>
#include <cstdio>
#include <fstream>
#include <vulkan/vulkan.h>

extern std::map<std::string, size_t> k_kernel_name_count;

inline std::vector<uint32_t> compile(const std::string& shader_entry, const std::string& source, char* filename)
{
	char tmp_filename_in[L_tmpnam];
	char tmp_filename_out[L_tmpnam];

	tmpnam(tmp_filename_in);
	tmpnam(tmp_filename_out);

	FILE* tmp_file = fopen(tmp_filename_in, "wb+");
	fputs(source.c_str(), tmp_file);
	fclose(tmp_file);

	tmp_file = fopen(tmp_filename_out, "wb+");
	fclose(tmp_file);

	const auto cmd_str = std::string(
		"glslangValidator -V --quiet " + std::string(tmp_filename_in) + " --entry-point " + shader_entry +
		" --source-entrypoint main -S comp -o " + tmp_filename_out);

	if (system(cmd_str.c_str()))
		throw std::runtime_error("Error running glslangValidator command");
	std::ifstream fileStream(tmp_filename_out, std::ios::binary);
	std::vector<char> buffer;
	buffer.insert(buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
	return { reinterpret_cast<uint32_t*>(buffer.data()), reinterpret_cast<uint32_t*>(buffer.data() + buffer.size()) };
}

struct job_create_info
{
	//init
	size_t num_buffers;
	VkDescriptorSetLayoutBinding* bindings;
	VkDescriptorSetLayoutCreateInfo descriptor_set_layout;
	VkDescriptorPoolSize descriptor_pool_size;
	VkDescriptorPoolCreateInfo descriptor_pool;
	VkDescriptorSetAllocateInfo descriptor_set_alloc;

	//create_pipeline 
	VkShaderModuleCreateInfo shader_module;
	VkPipelineShaderStageCreateInfo pipeline_stage;
	VkPushConstantRange push_constant_range;
	VkPipelineLayoutCreateInfo pipeline_layout;
	VkComputePipelineCreateInfo compute_pipeline;

	//bind_tensor
	VkDescriptorBufferInfo* descriptor_buffer;
	VkWriteDescriptorSet* write_desciptor_sets;

	//record_pipeline
	VkCommandBufferAllocateInfo command_buffer_alloc;
	VkCommandBufferBeginInfo command_buffer_begin;

};

inline void create_job_(job_create_info& ci, const VkDevice& device, VkDescriptorSetLayout* descriptor_set_layout,
                       VkDescriptorPool* descriptor_pool, VkDescriptorSet* descriptor_set)
{


	vkCreateDescriptorSetLayout(device, &ci.descriptor_set_layout, nullptr, descriptor_set_layout);
	VkResult result = vkCreateDescriptorPool(device, &ci.descriptor_pool, nullptr, descriptor_pool);
	if (result != VK_SUCCESS)
		std::cerr << "CANNOT CREATE DESCRIPTOR POOL\n";


	ci.descriptor_set_alloc.descriptorPool = *descriptor_pool;
	ci.descriptor_set_alloc.descriptorSetCount = 1;
	ci.descriptor_set_alloc.pSetLayouts = descriptor_set_layout;
	ci.pipeline_layout.pSetLayouts = descriptor_set_layout;

	result = vkAllocateDescriptorSets(device, &ci.descriptor_set_alloc, descriptor_set);
	if (result != VK_SUCCESS)
		std::cerr << "CANNOT CREATE DESCRIPTOR SET\n";
}


inline void create_pipeline_(job_create_info& ci, const VkDevice& device, VkShaderModule* shader_module,
                            VkPipelineLayout* pipeline_layout, VkPipeline* pipeline)
{
	vkCreateShaderModule(device, &ci.shader_module, nullptr, shader_module);
	ci.pipeline_stage.module = *shader_module;
	VkResult result = vkCreatePipelineLayout(device, &ci.pipeline_layout, nullptr, pipeline_layout);
	
	if (result != VK_SUCCESS)
		std::cerr << "FAILED TO CREATE PIPELINE LAYOUT\n";
	ci.compute_pipeline.stage = ci.pipeline_stage;
	ci.compute_pipeline.layout = *pipeline_layout;
	result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci.compute_pipeline, nullptr, pipeline);
	if (result != VK_SUCCESS)
		std::cerr << "FAILED TO CREATE PIPELINE\n";
}

inline void record_pipeline_(const job_create_info& ci, const VkDevice device, VkCommandBuffer* cmd_buffer, const VkPipelineLayout* pipeline_layout,
                            const uint32_t push_constants_size, const void* push_constants,
                            const VkPipeline* pipeline, const VkDescriptorSet* descriptor_set, uint32_t groups[3]
	)
{
	VkResult result = vkAllocateCommandBuffers(device, &ci.command_buffer_alloc, cmd_buffer);
	if (result != VK_SUCCESS)
		std::cerr << "CANNOT CREATE COMMAND BUFFER\n";
	vkBeginCommandBuffer(*cmd_buffer, &ci.command_buffer_begin);
	if (push_constants_size)
		vkCmdPushConstants(*cmd_buffer, *pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constants_size, push_constants);
	vkCmdBindPipeline(*cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
	vkCmdBindDescriptorSets(*cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout, 0, 1, descriptor_set, 0, nullptr);
	vkCmdPipelineBarrier(*cmd_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 0, nullptr);
	vkCmdDispatch(*cmd_buffer, groups[0], groups[1], groups[2]);
	result = vkEndCommandBuffer(*cmd_buffer);
	if (result != VK_SUCCESS)
		std::cerr << "FAILED TO RECORD CMD BUFFER\n";
}

class job
{
protected:
	//VkDevice device_{};
	//VkCommandPool cmd_pool_{};
	job_create_info ci_;

	VkDescriptorSetLayout descriptor_set_layout_{};
	VkDescriptorPool descriptor_pool_{};
	VkDescriptorSet descriptor_set_{};

	VkShaderModule shader_module_{};
	VkPipeline pipeline_{};
	VkPipelineLayout pipeline_layout_{};
	VkCommandBuffer cmd_buffer_{};
	VkBufferMemoryBarrier* memory_barriers_{};
	VkSubmitInfo submit_info_{};
	uint32_t num_buffers_{};

	std::string kernel_type_;
	std::string kernel_entry_;
	std::vector<uint32_t> compiled_shader_code_;

	uint32_t groups_[3]{1, 1, 1};
	uint32_t push_constants_size_{};
	void* push_constants_{};
	VkSpecializationInfo* specialization_info_{};

	std::string name_;

	void create_pipeline(){

		if (compiled_shader_code_.empty())
			std::cerr << "ERROR CODE NOT FOUND\n";

		ci_.shader_module.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		ci_.shader_module.pNext = nullptr;
		ci_.shader_module.flags = 0;
		ci_.shader_module.pCode = compiled_shader_code_.data();
		ci_.shader_module.codeSize = sizeof(uint32_t) * compiled_shader_code_.size();

		ci_.pipeline_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		ci_.pipeline_stage.pNext = nullptr;
		ci_.pipeline_stage.flags = 0;
		ci_.pipeline_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		ci_.pipeline_stage.pName = kernel_entry_.c_str();
		ci_.pipeline_stage.pSpecializationInfo = specialization_info_;

		ci_.push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		ci_.push_constant_range.offset = 0;
		ci_.push_constant_range.size = push_constants_size_;

		ci_.pipeline_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		ci_.pipeline_layout.pNext = nullptr;
		ci_.pipeline_layout.flags = 0;
		ci_.pipeline_layout.pushConstantRangeCount = push_constants_size_ ? 1 : 0;
		ci_.pipeline_layout.pPushConstantRanges = push_constants_size_ ? &ci_.push_constant_range : nullptr;
		ci_.pipeline_layout.setLayoutCount = 1;
		
		ci_.compute_pipeline.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		ci_.compute_pipeline.pNext = nullptr;
		ci_.compute_pipeline.flags = 0;
		ci_.compute_pipeline.stage = ci_.pipeline_stage;
		//	compute_pipeline_create_info.basePipelineIndex = 0;
		//	compute_pipeline_create_info.basePipelineHandle = nullptr;
	}

	void record_pipeline(const VkCommandPool cmd_pool){
		ci_.command_buffer_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		ci_.command_buffer_alloc.pNext = nullptr;
		ci_.command_buffer_alloc.commandPool = cmd_pool;
		ci_.command_buffer_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		ci_.command_buffer_alloc.commandBufferCount = 1;

		ci_.command_buffer_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		ci_.command_buffer_begin.pNext = nullptr;
		ci_.command_buffer_begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		ci_.command_buffer_begin.pInheritanceInfo = nullptr;

	}

public:
	job() = default;
	job(uint32_t num_buffers, const std::string& kernel_name = "nop") : ci_(),
	                                                                   num_buffers_(num_buffers),
	                                                                   kernel_type_(kernel_name),
	                                                                   kernel_entry_(kernel_name)
	{
		name_ = kernel_name + '_' + std::to_string(k_kernel_name_count[kernel_type_]++);
		create();
	}

	job(std::string kernel_name) : kernel_type_(std::move(kernel_name)){};
	

	void create() {
		ci_.num_buffers = num_buffers_;
		ci_.bindings = new VkDescriptorSetLayoutBinding[num_buffers_];
		memory_barriers_ = new VkBufferMemoryBarrier[num_buffers_];
		ci_.descriptor_buffer = new VkDescriptorBufferInfo[num_buffers_];
		ci_.write_desciptor_sets = new VkWriteDescriptorSet[num_buffers_];

		for (unsigned i = 0; i < num_buffers_; i++)
		{
			ci_.bindings[i].binding = i;
			ci_.bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			ci_.bindings[i].descriptorCount = 1;
			ci_.bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}

		ci_.descriptor_set_layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		ci_.descriptor_set_layout.pNext = nullptr;
		ci_.descriptor_set_layout.bindingCount = num_buffers_;
		ci_.descriptor_set_layout.pBindings = ci_.bindings;

		ci_.descriptor_set_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		ci_.descriptor_set_alloc.pNext = nullptr;

		ci_.descriptor_pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		ci_.descriptor_pool.pNext = nullptr;
		ci_.descriptor_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		ci_.descriptor_pool_size.descriptorCount = num_buffers_;
		ci_.descriptor_pool.maxSets = 1;
		ci_.descriptor_pool.poolSizeCount = 1;
		ci_.descriptor_pool.pPoolSizes = &ci_.descriptor_pool_size;

	}

	void cleanup(const VkDevice device) const
	{
		delete[] memory_barriers_;
		if (shader_module_ != nullptr)
			vkDestroyShaderModule(device, shader_module_, nullptr);
		if (descriptor_pool_ != nullptr)
			vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
		if (descriptor_set_layout_ != nullptr)
			vkDestroyDescriptorSetLayout(device, descriptor_set_layout_, nullptr);
		if (pipeline_ != nullptr)
			vkDestroyPipeline(device, pipeline_, nullptr);
		if (pipeline_layout_ != nullptr)
			vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
	}

	~job() = default;

	void set_push_constants(void* params, uint32_t params_size)
	{
		push_constants_ = params;
		push_constants_size_ = params_size;
	}

	void set_shader(const std::string& shader, VkSpecializationInfo* specifialization_info = nullptr) {
		compiled_shader_code_ = compile(kernel_entry_, shader, nullptr);
		specialization_info_ = specifialization_info;
	}

	void set_group_size(const uint32_t x = 1, const uint32_t y = 1, const uint32_t z = 1) {
		groups_[0] = x;
		groups_[1] = y;
		groups_[2] = z;
	}

	[[nodiscard]] VkSubmitInfo get_submit_info() const { return submit_info_; }
	[[nodiscard]] VkCommandBuffer get_command_buffer() const { return cmd_buffer_; }

	void bind_buffer(const vk_block* blk, uint32_t i, uint32_t offset) {
		if (i == 0)
			create_pipeline();
		/*memory_barriers_[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		memory_barriers_[i].pNext = nullptr;
		memory_barriers_[i].buffer = blk->buf;
		memory_barriers_[i].offset = offset;
		memory_barriers_[i].size = blk->size;
		memory_barriers_[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
		memory_barriers_[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;*/

		ci_.descriptor_buffer[i].buffer = blk->buf;
		ci_.descriptor_buffer[i].offset = offset;
		ci_.descriptor_buffer[i].range = blk->size;

		ci_.write_desciptor_sets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		ci_.write_desciptor_sets[i].pNext = nullptr;
		ci_.write_desciptor_sets[i].dstArrayElement = 0;
		ci_.write_desciptor_sets[i].dstBinding = i;
		ci_.write_desciptor_sets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		ci_.write_desciptor_sets[i].descriptorCount = 1;
		ci_.write_desciptor_sets[i].pBufferInfo = &ci_.descriptor_buffer[i];
		ci_.write_desciptor_sets[i].pTexelBufferView = nullptr;
		ci_.write_desciptor_sets[i].pImageInfo = nullptr;
	}
		

	void trigger(const VkDevice device, VkCommandPool cmd_pool) {
		record_pipeline(cmd_pool);

		create_job_(ci_, device, &descriptor_set_layout_, &descriptor_pool_, &descriptor_set_);
		create_pipeline_(ci_, device, &shader_module_, &pipeline_layout_, &pipeline_);
		for (auto i = 0; i < num_buffers_; ++i)
		{
			ci_.write_desciptor_sets[i].dstSet = descriptor_set_;
			vkUpdateDescriptorSets(device, 1, &ci_.write_desciptor_sets[i], 0, nullptr);
		}
		record_pipeline_(ci_, device, &cmd_buffer_, &pipeline_layout_, push_constants_size_, push_constants_, &pipeline_, &descriptor_set_, groups_);

		submit_info_.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info_.pNext = nullptr;
		submit_info_.commandBufferCount = 1;
		submit_info_.pCommandBuffers = &cmd_buffer_;
	}


};


class copy : public job
{
public:
	copy(const VkDevice device, const VkCommandPool& cmd_pool, const vk_block* src, const vk_block* dst,
	     const size_t dst_offset, const size_t src_offset) : job("MEMCOPY")
	{
		VkBufferCopy copy_region{};
		copy_region.srcOffset = src_offset;
		copy_region.dstOffset = dst_offset;
		copy_region.size = std::min<size_t>(src->size, dst->size);

		ci_.command_buffer_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		ci_.command_buffer_alloc.pNext = nullptr;
		ci_.command_buffer_alloc.commandPool = cmd_pool;
		ci_.command_buffer_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		ci_.command_buffer_alloc.commandBufferCount = 1;

		ci_.command_buffer_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		ci_.command_buffer_begin.pNext = nullptr;
		ci_.command_buffer_begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		VkMemoryBarrier memory_barrier{};
		memory_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		memory_barrier.pNext = nullptr;
		memory_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
		memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;



		VkResult result = vkAllocateCommandBuffers(device, &ci_.command_buffer_alloc, &cmd_buffer_);
		vkBeginCommandBuffer(cmd_buffer_, &ci_.command_buffer_begin);
		vkCmdPipelineBarrier(cmd_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &memory_barrier,
			0, nullptr, 0, nullptr);
		vkCmdCopyBuffer(cmd_buffer_, src->buf, dst->buf, 1, &copy_region);
		vkEndCommandBuffer(cmd_buffer_);
		submit_info_.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info_.pNext = nullptr;
		submit_info_.commandBufferCount = 1;
		submit_info_.pCommandBuffers = &cmd_buffer_;
	}
};
