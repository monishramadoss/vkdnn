// threads.h : Header file for your target.

#pragma once
#include "job.h"
#include <mutex>
#include <queue>
#include <thread>
#include <functional>
#include <iostream>
#include <condition_variable>
#include <vulkan/vulkan.h>


class device_submission_thread final
{
private:
	bool destroying_ = false;
	std::thread worker_ {};
	std::queue<job_create_info_data*> job_queue_ {};
	std::mutex queue_mutex_ {};
	std::condition_variable condition_ {};

	VkDevice device_{};
	VkQueue cmd_queue_{};
	VkCommandPool cmd_pool_{};
	VkCommandBuffer cmd_buffer_{};
	VkFence fence_{};

	void queue_loop()
	{
		VkCommandBufferBeginInfo cmd_buffer_begin_info{};
		cmd_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		cmd_buffer_begin_info.pNext = nullptr;
		cmd_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		cmd_buffer_begin_info.pInheritanceInfo = nullptr;


		while (true)
		{
			
			job_create_info_data* job{};
			{
				std::unique_lock<std::mutex> lock(queue_mutex_);
				condition_.wait(lock, [this] { return !job_queue_.empty() || destroying_; });
				if (destroying_)
					break;
				job = job_queue_.front();
				job_queue_.pop();

				job->submit_info.commandBufferCount = 1;
				job->submit_info.pCommandBuffers = &job->secondary_cmd_buffer;
				vkQueueSubmit(cmd_queue_, 1, &job->submit_info, fence_);
				vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINTMAX_MAX);
				vkResetFences(device_, 1, &fence_);
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

	void start_on(const VkDevice device, const VkQueue cmd_queue, const VkCommandPool cmd_pool)
	{
		device_ = device;
		cmd_queue_ = cmd_queue;
		cmd_pool_ = cmd_pool;

		VkCommandBufferAllocateInfo command_buffer_alloc_info{};
		command_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		command_buffer_alloc_info.pNext = nullptr;
		command_buffer_alloc_info.commandPool = cmd_pool_;
		command_buffer_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		command_buffer_alloc_info.commandBufferCount = 1;
		if (const VkResult result = vkAllocateCommandBuffers(device, &command_buffer_alloc_info, &cmd_buffer_); result != VK_SUCCESS)
			std::cerr << "CANNOT ALLOCATE COMMAND BUFFER\n";



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
			queue_mutex_.lock();
			destroying_ = true;
			condition_.notify_one();
			queue_mutex_.unlock();
			worker_.join();
			vkDestroyFence(device_, fence_, nullptr);
		}
	}

	void add_job(job_create_info_data* function)
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



class generation_thread final
{
private:
	bool destroying_ = false;
	std::thread worker_{};
	std::queue<job_create_info_data*> job_queue_{};
	std::mutex queue_mutex_{};
	std::condition_variable condition_{};
	
	VkDevice device_{};
	VkCommandPool cmd_pool_{};
	VkCommandBuffer cmd_buffer_{};
	void queue_loop()
	{
		while (true)
		{
			job_create_info_data* job{};
			{
				std::unique_lock<std::mutex> lock(queue_mutex_);
				condition_.wait(lock, [this] { return !job_queue_.empty() || destroying_; });
				if (destroying_)
					break;
				job = job_queue_.front();			
				job_queue_.pop();
			}
			
			{
				condition_.notify_one();				
				job->generate_pipeline();
				//job->device_thread->add_job(&job->submit_info);
			}
		}
	}

public:

	generation_thread() = default;

	void start_on(const VkDevice device, const VkCommandPool cmd_pool, const VkCommandBuffer cmd_buffer)
	{
		device_ = device;
		cmd_pool_ = cmd_pool;
		cmd_buffer_ = cmd_buffer;
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
		}
	}

	void add_job(job_create_info_data* data)
	{
		data->device = device_;
		data->cmd_pool = cmd_pool_;
		data->secondary_cmd_buffer = cmd_buffer_;
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

/*
 * MULTI THREAD COMMAND BUFFER GENERATION DOES NOT WORK WITH COMPUTE
 * COMPUTE IS NOT INCLUDED IN BUFFER INHERITENCE INFO
 */

class cmd_thread_pool final
{
	size_t thread_id_ = 0;
	size_t num_threads_;
	VkCommandBuffer* cmd_buffers_{};
	VkCommandPool* cmd_pools_{};
	std::vector<generation_thread> generation_threads_;

public:
	cmd_thread_pool(): num_threads_(0) {}

	void start_on(const size_t num_threads, const VkDevice device, uint32_t queue_index) 
	{
		num_threads_ = num_threads;
		generation_threads_ = std::vector<generation_thread>(num_threads_);
		cmd_buffers_ = new VkCommandBuffer[num_threads_];
		cmd_pools_ = new VkCommandPool[num_threads_];

		for (size_t i = 0; i < num_threads_; ++i)
		{
			VkCommandPoolCreateInfo command_pool_create_info{};
			command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
			command_pool_create_info.pNext = nullptr;
			command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
			command_pool_create_info.queueFamilyIndex = queue_index;
			VkResult result = vkCreateCommandPool(device, &command_pool_create_info, nullptr, &cmd_pools_[i]);
			if (result != VK_SUCCESS)
				std::cerr << "CANNOT CREATE COMMAND POOL\n";

			VkCommandBufferAllocateInfo command_buffer_alloc_info{};
			command_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			command_buffer_alloc_info.pNext = nullptr;
			command_buffer_alloc_info.commandPool = cmd_pools_[i];
			command_buffer_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			command_buffer_alloc_info.commandBufferCount = 1;
			result = vkAllocateCommandBuffers(device, &command_buffer_alloc_info, &cmd_buffers_[i]);
			VkCommandBufferInheritanceInfo command_buffer_inheritence_info{};
			command_buffer_inheritence_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
			command_buffer_inheritence_info.pNext = nullptr;



			if (result != VK_SUCCESS)
				std::cerr << "CANNOT ALLOCATE COMMAND BUFFER\n";

			generation_threads_[i].start_on(device, cmd_pools_[i], cmd_buffers_[i]);
		}
	}

	void add_job(job_create_info_data* data)
	{
		const auto idx = (thread_id_++) % num_threads_;
		generation_threads_[idx].add_job(data);
	}

	void wait()
	{
		for (auto& th : generation_threads_)
			th.wait();
	}
};
