// job.h : Header file for your target.
#pragma once
#define _CRT_SECURE_NO_WARNINGS

#include <string.h>
#include <map>
#include <vector>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vulkan/vulkan.h>

#include "allocator.h"
#ifdef __linux__
#include <stdlib.h>
#endif

extern std::map<std::string, size_t> k_kernel_name_count;

inline std::vector<uint32_t> compile(const std::string& shader_entry, const std::string& source, char* filename)
{
	char tmp_filename_in[L_tmpnam];
	char tmp_filename_out[L_tmpnam];
	std::vector<char> buffer;
	FILE* tmp_file;
#ifdef WIN32
	auto err1 = tmpnam_s(tmp_filename_in, L_tmpnam);
	auto err2 = tmpnam_s(tmp_filename_out, L_tmpnam);

	auto err3 = fopen_s(&tmp_file, tmp_filename_in, "wb+");
	int i = fputs(source.c_str(), tmp_file);
	i = fclose(tmp_file);
	auto err4 = fopen_s(&tmp_file, tmp_filename_out, "wb+");
	i = fclose(tmp_file);
#else
	
	
	tmpnam(tmp_filename_in);
	tmpnam(tmp_filename_out);

	tmp_file = fopen(tmp_filename_in, "wb+");
	int i = fputs(source.c_str(), tmp_file);
	i = fclose(tmp_file);
	tmp_file = fopen(tmp_filename_out, "wb+");
	i = fclose(tmp_file);

#endif
	
	if(i != 0)
		throw std::runtime_error("Error loading or creating temp file");
	
	std::cout << source << "\n";
	const auto cmd_str = std::string(
		"glslangValidator -V --quiet --target-env vulkan1.2 " + std::string(tmp_filename_in) + " --entry-point " + shader_entry +
		" --source-entrypoint main -S comp -o " + tmp_filename_out
	);

	if (system(cmd_str.c_str()))
		throw std::runtime_error("Error running glslangValidator command");

	std::ifstream file_stream(tmp_filename_out, std::ios::binary);
	buffer.insert(buffer.begin(), std::istreambuf_iterator<char>(file_stream), {});
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

	VkPipelineShaderStageRequiredSubgroupSizeCreateInfo shader_subgroup_size;
};


enum op_type
{
	COMPUTE_OP = 1,
	DEVICE_HOST_OP = 1,
	HOST_DEVICE_OP = 2
};


class job_create_info_data
{
public:
	VkDevice device = nullptr;
	VkCommandPool cmd_pool = nullptr;

	VkBufferMemoryBarrier* memory_buffer_barriers = nullptr;
	VkCommandBuffer secondary_cmd_buffer = nullptr;
	VkPipelineLayout pipeline_layout = nullptr;
	uint32_t push_constants_size = 0;
	void* push_constants = nullptr;
	VkPipeline pipeline = nullptr;
	VkDescriptorSet descriptor_set = nullptr;
	uint32_t groups[3]{};
	VkSubmitInfo submit_info;
	op_type op = COMPUTE_OP;

	virtual void generate_pipeline()
	{
		VkCommandBufferBeginInfo cmd_buffer_begin_info{};
		cmd_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		cmd_buffer_begin_info.pNext = nullptr;
		cmd_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		cmd_buffer_begin_info.pInheritanceInfo = nullptr;


		VkResult result = vkBeginCommandBuffer(secondary_cmd_buffer, &cmd_buffer_begin_info);
		if (result != VK_SUCCESS)
			std::cerr << "CANNOT BEGIN COMMAND BUFFER\n";

		if (push_constants_size)
			vkCmdPushConstants(secondary_cmd_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constants_size,
				push_constants);
		vkCmdBindPipeline(secondary_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
		vkCmdBindDescriptorSets(secondary_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
		vkCmdPipelineBarrier(secondary_cmd_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 0, nullptr);
		vkCmdDispatch(secondary_cmd_buffer, groups[0], groups[1], groups[2]);

		result = vkEndCommandBuffer(secondary_cmd_buffer);
		if (result != VK_SUCCESS)
			std::cerr << "FAILED TO RECORD CMD BUFFER\n";

		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.pNext = nullptr;
		submit_info.signalSemaphoreCount = 0;
		submit_info.pSignalSemaphores = nullptr;
		submit_info.waitSemaphoreCount = 0;
		submit_info.pWaitSemaphores = nullptr;
	}

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
                            const VkPipeline* pipeline, const VkDescriptorSet* descriptor_set, uint32_t groups[3])
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
	job_create_info ci_;
	job_create_info_data gd_;

	VkDescriptorSetLayout descriptor_set_layout_{};
	VkDescriptorPool descriptor_pool_{};
	
	VkShaderModule shader_module_{};
	uint32_t num_buffers_{};

	std::string kernel_type_;
	std::string kernel_entry_;
	std::vector<uint32_t> compiled_shader_code_;
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

		VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_create_info;
		subgroup_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
		subgroup_create_info.pNext = nullptr;
		subgroup_create_info.requiredSubgroupSize = 32;

		ci_.pipeline_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		ci_.pipeline_stage.pNext = nullptr;
		ci_.pipeline_stage.flags = 0;
		ci_.pipeline_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		ci_.pipeline_stage.pName = kernel_entry_.c_str();
		ci_.pipeline_stage.pSpecializationInfo = specialization_info_;

		ci_.push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		ci_.push_constant_range.offset = 0;
		ci_.push_constant_range.size = gd_.push_constants_size;

		ci_.pipeline_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		ci_.pipeline_layout.pNext = nullptr;
		ci_.pipeline_layout.flags = 0;
		ci_.pipeline_layout.pushConstantRangeCount = gd_.push_constants_size ? 1 : 0;
		ci_.pipeline_layout.pPushConstantRanges = gd_.push_constants_size ? &ci_.push_constant_range : nullptr;
		ci_.pipeline_layout.setLayoutCount = 1;
		
		ci_.compute_pipeline.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		ci_.compute_pipeline.pNext = nullptr;
		ci_.compute_pipeline.flags = 0;
		ci_.compute_pipeline.stage = ci_.pipeline_stage;
		//	compute_pipeline_create_info.basePipelineIndex = 0;
		//	compute_pipeline_create_info.basePipelineHandle = nullptr;

		ci_.command_buffer_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		ci_.command_buffer_alloc.pNext = nullptr;
		ci_.command_buffer_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		ci_.command_buffer_alloc.commandBufferCount = 1;

		ci_.command_buffer_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		ci_.command_buffer_begin.pNext = nullptr;
		ci_.command_buffer_begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		ci_.command_buffer_begin.pInheritanceInfo = nullptr;
	}


public:
	job() {}

	explicit job(const uint32_t num_buffers, const std::string& kernel_name = "nop") : ci_(),
	                                                                                   num_buffers_(num_buffers),
	                                                                                   kernel_type_(kernel_name),
	                                                                                   kernel_entry_(kernel_name)
	{
		name_ = kernel_name + '_' + std::to_string(k_kernel_name_count[kernel_type_]++);
		create();
	}

	job(std::string kernel_name) : kernel_type_(std::move(kernel_name))
	{
		
	}

	~job() {}

	void create() {
		ci_.num_buffers = num_buffers_;
		ci_.bindings = new VkDescriptorSetLayoutBinding[num_buffers_];
		gd_.memory_buffer_barriers = new VkBufferMemoryBarrier[num_buffers_];
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
		if (shader_module_ != nullptr)
			vkDestroyShaderModule(device, shader_module_, nullptr);
		if (descriptor_pool_ != nullptr)
			vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
		if (descriptor_set_layout_ != nullptr)
			vkDestroyDescriptorSetLayout(device, descriptor_set_layout_, nullptr);
		if (gd_.pipeline != nullptr)
			vkDestroyPipeline(device, gd_.pipeline, nullptr);
		if (gd_.pipeline_layout != nullptr)
			vkDestroyPipelineLayout(device, gd_.pipeline_layout, nullptr);
	}




	void set_push_constants(void* params, uint32_t params_size)
	{
		gd_.push_constants = params;
		gd_.push_constants_size = params_size;
	}

	void set_shader(const std::string& shader, VkSpecializationInfo* specifialization_info = nullptr) {
		compiled_shader_code_ = compile(kernel_entry_, shader, nullptr);
		specialization_info_ = specifialization_info;
	}

	void set_group_size(const uint32_t x = 1, const uint32_t y = 1, const uint32_t z = 1) {
		gd_.groups[0] = x;
		gd_.groups[1] = y;
		gd_.groups[2] = z;
	}

	[[nodiscard]] VkSubmitInfo* get_submit_info() 
	{
		return &gd_.submit_info;
	}

	void bind_buffer(const vk_block* blk, const uint32_t i, const uint32_t offset) {
		if (i == 0)
			create_pipeline();

		gd_.memory_buffer_barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		gd_.memory_buffer_barriers[i].pNext = nullptr;
		gd_.memory_buffer_barriers[i].buffer = blk->buf;
		gd_.memory_buffer_barriers[i].offset = offset;
		gd_.memory_buffer_barriers[i].size = blk->size;
		gd_.memory_buffer_barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
		gd_.memory_buffer_barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;

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
		

	virtual job_create_info_data* generation_data(const VkDevice device, const VkCommandPool cmd_pool) 
	{
		ci_.command_buffer_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		ci_.command_buffer_alloc.pNext = nullptr;
		ci_.command_buffer_alloc.commandPool = cmd_pool;
		ci_.command_buffer_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		ci_.command_buffer_alloc.commandBufferCount = 1;
		if (vkAllocateCommandBuffers(device, &ci_.command_buffer_alloc, &gd_.secondary_cmd_buffer) != VK_SUCCESS)
			std::cerr << "CANNOT CONSTRUCT VULKAN BUFFER\n";

		gd_.device = device;
		gd_.cmd_pool = cmd_pool;
		create_job_(ci_, device, &descriptor_set_layout_, &descriptor_pool_, &gd_.descriptor_set);
		create_pipeline_(ci_, device, &shader_module_, &gd_.pipeline_layout, &gd_.pipeline);
		for (uint32_t i = 0; i < num_buffers_; ++i)
		{
			ci_.write_desciptor_sets[i].dstSet = gd_.descriptor_set;
			vkUpdateDescriptorSets(device, 1, &ci_.write_desciptor_sets[i], 0, nullptr);
		}

		gd_.generate_pipeline();
		return &gd_;
	}
};


class copy_generation_data final : public job_create_info_data
{
public:
	VkBufferCopy copy_region{};
	VkMemoryBarrier memory_barrier{};
	vk_block* src = nullptr;
	vk_block* dst = nullptr;
	void generate_pipeline() override
	{
		VkCommandBufferBeginInfo cmd_buffer_begin_info{};
		cmd_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		cmd_buffer_begin_info.pNext = nullptr;
		cmd_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		cmd_buffer_begin_info.pInheritanceInfo = nullptr;
		if (vkBeginCommandBuffer(secondary_cmd_buffer, &cmd_buffer_begin_info) != VK_SUCCESS)
			std::cerr << "CANNOT BEGIN COMMAND BUFFER\n";
		vkCmdPipelineBarrier(secondary_cmd_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &memory_barrier,
			0, nullptr, 0, nullptr);
		vkCmdCopyBuffer(secondary_cmd_buffer, src->buf, dst->buf, 1, &copy_region);
		if (vkEndCommandBuffer(secondary_cmd_buffer) != VK_SUCCESS)
			std::cerr << "CANNOT END COMMAND BUFFER\n";

		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.pNext = nullptr;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &secondary_cmd_buffer;
		submit_info.signalSemaphoreCount = 0;
		submit_info.waitSemaphoreCount = 0;
	}
};

class copy final : public job
{	
	copy_generation_data gd_;
	
public:
	copy(vk_block* src, vk_block* dst, const size_t dst_offset, const size_t src_offset) : job("MEMCOPY")
	{
		num_buffers_ = 1;
		gd_.memory_buffer_barriers = new VkBufferMemoryBarrier[1];

		gd_.src = src;
		gd_.dst = dst;

		gd_.copy_region.srcOffset = src_offset;
		gd_.copy_region.dstOffset = dst_offset;
		gd_.copy_region.size = std::min<size_t>(src->size, dst->size);

		ci_.command_buffer_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		ci_.command_buffer_begin.pNext = nullptr;
		ci_.command_buffer_begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		ci_.command_buffer_begin.pInheritanceInfo = nullptr;


		gd_.memory_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		gd_.memory_barrier.pNext = nullptr;
		gd_.memory_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
		gd_.memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;

		gd_.memory_buffer_barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		gd_.memory_buffer_barriers[0].pNext = nullptr;
		gd_.memory_buffer_barriers[0].buffer = src->buf;
		gd_.memory_buffer_barriers[0].offset = src_offset;
		gd_.memory_buffer_barriers[0].size = src->size;
		gd_.memory_buffer_barriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
		gd_.memory_buffer_barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
		gd_.op = DEVICE_HOST_OP; // HOST_DEVICE_OP;
	}

	copy_generation_data* generation_data(const VkDevice device, const VkCommandPool cmd_pool) override
	{	
		gd_.device = device;
		ci_.command_buffer_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		ci_.command_buffer_alloc.pNext = nullptr;
		ci_.command_buffer_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		ci_.command_buffer_alloc.commandBufferCount = 1;
		ci_.command_buffer_alloc.commandPool = cmd_pool;
		if (vkAllocateCommandBuffers(device, &ci_.command_buffer_alloc, &gd_.secondary_cmd_buffer) != VK_SUCCESS)
			std::cerr << "CANNOT ALLOCATE COMMAND BUFFER\n";
		gd_.generate_pipeline();

		return &gd_;
	}

};
