#include <vulkan/vulkan.h>
#include "utils.h"
#include "tensor.h"


void box (tensor&t1, tensor&t2, )
{
	auto* blk1 = t1.get_data();
	auto* blk2 = t2.get_data();
	if (blk1->device_id == blk2->device_id && blk1->on_device == blk2->on_device)
	{
		t2 = t1;
	}


}


