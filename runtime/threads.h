// threads.h : Header file for your target.

#pragma once
#include <mutex>
#include <queue>
#include <thread>
#include <functional>

#include <vulkan/vulkan.h>

struct thread_data
{
	VkSubmitInfo submit_info;
	VkFence fence;
};

class device_submission_thread final
{
private:
	bool destroying_ = false;
	std::thread worker_ {};
	std::queue<thread_data> job_queue_ {};
	std::mutex queue_mutex_ {};
	std::condition_variable condition_ {};

	VkDevice device_{};
	VkQueue cmd_queue_{};
	VkFence fence_{};

	void queue_loop()
	{
		while (true)
		{
			thread_data job{};
			{
				std::unique_lock<std::mutex> lock(queue_mutex_);
				condition_.wait(lock, [this] { return !job_queue_.empty() || destroying_; });
				if (destroying_)
					break;
				job = job_queue_.front();
			}
			vkQueueSubmit(cmd_queue_, 1, &job.submit_info, fence_);
			vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINTMAX_MAX);
			{
				std::lock_guard<std::mutex> lock(queue_mutex_);
				vkResetFences(device_, 1, &fence_);
				job_queue_.pop();
				condition_.notify_one();
			}
		}
	}

public:

	device_submission_thread(device_submission_thread&&t) noexcept: destroying_(t.destroying_), job_queue_(std::move(t.job_queue_)), device_(t.device_), cmd_queue_(t.cmd_queue_), fence_(t.fence_)
	{
		worker_ = std::thread(&device_submission_thread::queue_loop, this);
	}

	device_submission_thread& operator=(device_submission_thread&& t) noexcept
	{
		if(this != &t)
		{
			destroying_ = t.destroying_;
			worker_ = std::move(t.worker_);
			job_queue_ = std::move(t.job_queue_);
			device_ = t.device_;
			cmd_queue_ = t.cmd_queue_;
			fence_ = t.fence_;
		}
		return *this;
	}

	device_submission_thread() = default;

	void start_on(const VkDevice device, const VkQueue cmd_queue)
	{
		device_ = device;
		cmd_queue_ = cmd_queue;
		VkFenceCreateInfo fence_create_info{};
		fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fence_create_info.pNext = nullptr;
		fence_create_info.flags = 0;

		VkResult result = vkCreateFence(device_, &fence_create_info, nullptr, &fence_);
		worker_ = std::thread(&device_submission_thread::queue_loop, this);
	}

	~device_submission_thread()
	{
		if (worker_.joinable())
		{
			wait();
			queue_mutex_.lock();
			destroying_ = true;
			condition_.notify_one();
			queue_mutex_.unlock();
			worker_.join();
			vkDestroyFence(device_, fence_, nullptr);
		}
	}

	void add_job(const thread_data& function)
	{
		std::lock_guard<std::mutex> lock(queue_mutex_);
		job_queue_.push(function);
		condition_.notify_one();
	}

	void wait()
	{
		std::unique_lock<std::mutex> lock(queue_mutex_);
		condition_.wait(lock, [this]() { return job_queue_.empty(); });
	}

	[[nodiscard]] std::thread::id pid() const { return worker_.get_id(); }
};

struct generation_data
{
	VkDevice device;
	VkCommandPool cmd_pool;
	VkCommandBuffer cmd_buffer;
	VkPipelineLayout pipeline_layout;
	uint32_t push_constants_size;
	void* push_constants;
	VkPipeline pipeline;
	VkDescriptorSet descriptor_set;
	uint32_t groups[3];
	VkSubmitInfo submit_info;
};



#include <iostream>


class generation_thread final
{
private:
	bool destroying_ = false;
	std::thread worker_{};
	std::queue<generation_data> job_queue_{};
	std::mutex queue_mutex_{};
	std::condition_variable condition_{};

	VkDevice device_{};
	VkQueue cmd_queue_{};
	VkFence fence_{};

	static void record_pipeline(generation_data& data)
	{
		VkCommandBufferAllocateInfo command_buffer_alloc_info{};
		command_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		command_buffer_alloc_info.pNext = nullptr;
		command_buffer_alloc_info.commandPool = data.cmd_pool;
		command_buffer_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		command_buffer_alloc_info.commandBufferCount = 1;
		VkResult result = vkAllocateCommandBuffers(data.device, &command_buffer_alloc_info, &data.cmd_buffer);

		VkCommandBufferBeginInfo cmd_buffer_begin_info{};
		cmd_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		cmd_buffer_begin_info.pNext = nullptr;
		cmd_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		cmd_buffer_begin_info.pInheritanceInfo = nullptr;

		vkBeginCommandBuffer(data.cmd_buffer, &cmd_buffer_begin_info);

		if (data.push_constants_size)
			vkCmdPushConstants(data.cmd_buffer, data.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, data.push_constants_size,
				data.push_constants);

		vkCmdBindPipeline(data.cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, data.pipeline);
		vkCmdBindDescriptorSets(data.cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, data.pipeline_layout, 0, 1, &data.descriptor_set, 0,
			nullptr);
		vkCmdPipelineBarrier(data.cmd_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 0, nullptr);
		vkCmdDispatch(data.cmd_buffer, data.groups[0], data.groups[1], data.groups[2]);

		result = vkEndCommandBuffer(data.cmd_buffer);

		if (result != VK_SUCCESS)
			std::cerr << "FAILED TO RECORD CMD BUFFER\n";

		data.submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		data.submit_info.commandBufferCount = 1;
		data.submit_info.pCommandBuffers = &data.cmd_buffer;
	}

	void queue_loop()
	{
		while (true)
		{
			generation_data job{};
			{
				std::unique_lock<std::mutex> lock(queue_mutex_);
				condition_.wait(lock, [this] { return !job_queue_.empty() || destroying_; });
				if (destroying_)
					break;
				job = job_queue_.front();
			}
			record_pipeline(job);
			{
				std::lock_guard<std::mutex> lock(queue_mutex_);
				vkResetFences(device_, 1, &fence_);
				job_queue_.pop();
				condition_.notify_one();
			}
		}
	}

public:



	generation_thread() = default;

	void start_on()
	{
		worker_ = std::thread(&generation_thread::queue_loop, this);
	}

	~generation_thread()
	{
		if (worker_.joinable())
		{
			wait();
			queue_mutex_.lock();
			destroying_ = true;
			condition_.notify_one();
			queue_mutex_.unlock();
			worker_.join();
			vkDestroyFence(device_, fence_, nullptr);
		}
	}

	void add_job(const generation_data& data)
	{
		std::lock_guard<std::mutex> lock(queue_mutex_);
		job_queue_.push(data);
		condition_.notify_one();
	}

	void wait()
	{
		std::unique_lock<std::mutex> lock(queue_mutex_);
		condition_.wait(lock, [this]() { return job_queue_.empty(); });
	}

	[[nodiscard]] std::thread::id pid() const { return worker_.get_id(); }
};

class cmd_thread_pool final
{
	std::vector<generation_thread> generation_threads_;
	std::vector<device_submission_thread> device_threads_;

	size_t thread_id_ = 0;
	size_t num_threads_;
public:
	explicit cmd_thread_pool(const size_t device_threads) : generation_threads_(std::thread::hardware_concurrency() - device_threads), num_threads_(device_threads) {
		
		for (auto i = 0; i < device_threads; ++i)
			generation_threads_[i].start_on();
	}

	void add_job(const generation_data data)
	{
		const auto idx = (thread_id_ + 1) % num_threads_;
		generation_threads_[idx].add_job(data);
	}
};
