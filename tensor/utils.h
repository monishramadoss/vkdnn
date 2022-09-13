#pragma once
#include <format>
#define SHADER_VERSION "#version 460\n"
#define SUBGROUP_ENABLE "#extension GL_KHR_shader_subgroup_arithmetic : enable\n"


template<typename... Args>
std::string Format(const std::string_view message, Args... formatItems)
{
	return std::vformat(message, std::make_format_args(std::forward<Args>(formatItems)...));
}

inline std::string singlton_shader_code(const std::string& kernel_shader_code, const std::string_view fn_pass, const tensor& t1)
{
	std::string local_shader = kernel_shader_code;
	std::string shader_tensor;
	const int ext_int_0 = tensor_injection(local_shader, shader_tensor, 0, t1);
	local_shader = shader_extensions[ext_int_0] + local_shader;

	local_shader = SHADER_VERSION + local_shader;

	local_shader += "\nvoid main() {\n\tfor (uint i = gl_GlobalInvocationID.x; i < total; i += gl_NumWorkGroups.x * gl_WorkGroupSize.x){\n\t\t";
	local_shader += Format(fn_pass, shader_tensor.c_str()) + "\n\t}\n}";
	return local_shader;
}

inline std::string unary_shader_code(const std::string& kernel_shader_code, const std::string_view fn_pass, const tensor& t1, const tensor& t2)
{
	std::string local_shader = kernel_shader_code;
	std::string shader_tensor[2];
	const int ext_int_0 = tensor_injection(local_shader, shader_tensor[0], 0, t1);
	const int ext_int_1 = tensor_injection(local_shader, shader_tensor[1], 1, t2);
	if (ext_int_0 == ext_int_1)
		local_shader = shader_extensions[ext_int_0] + local_shader;
	else
	{
		local_shader = shader_extensions[ext_int_0] + local_shader;
		local_shader = shader_extensions[ext_int_1] + local_shader;
	}
	local_shader = SHADER_VERSION + local_shader;

	local_shader += "\nvoid main() {\n\tfor (uint i = gl_GlobalInvocationID.x; i < total; i += gl_NumWorkGroups.x * gl_WorkGroupSize.x){\n\t\t";
	local_shader += Format(fn_pass, shader_tensor[0].c_str(), shader_tensor[1].c_str()) + "\n\t}\n}";
	return local_shader;
}

inline std::string binary_shader_code(const std::string& kernel_shader_code, const std::string_view& fn_pass, const tensor& t1, const tensor& t2, const tensor& t3)
{
	std::string local_shader = kernel_shader_code;
	std::string shader_tensor[3];

	const int ext_int_0 = tensor_injection(local_shader, shader_tensor[0], 0, t1);
	const int ext_int_1 = tensor_injection(local_shader, shader_tensor[1], 1, t2);
	const int ext_int_2 = tensor_injection(local_shader, shader_tensor[2], 2, t3);
	if (ext_int_0 == ext_int_1 && ext_int_0 == ext_int_2)
		local_shader = shader_extensions[ext_int_0] + local_shader;
	else if (ext_int_0 == ext_int_1 && ext_int_0 != ext_int_2)
	{
		local_shader = shader_extensions[ext_int_0] + local_shader;
		local_shader = shader_extensions[ext_int_2] + local_shader;
	}
	else if (ext_int_0 != ext_int_1 && ext_int_0 == ext_int_2)
	{
		local_shader = shader_extensions[ext_int_0] + local_shader;
		local_shader = shader_extensions[ext_int_1] + local_shader;
	}
	else if (ext_int_1 == ext_int_2 && ext_int_0 != ext_int_1)
	{
		local_shader = shader_extensions[ext_int_1] + local_shader;
		local_shader = shader_extensions[ext_int_2] + local_shader;
	}
	else
	{
		local_shader = shader_extensions[ext_int_0] + local_shader;
		local_shader = shader_extensions[ext_int_1] + local_shader;
		local_shader = shader_extensions[ext_int_2] + local_shader;
	}
	local_shader = SHADER_VERSION + local_shader;

	local_shader += "\nvoid main() {\n\tfor (uint i = gl_GlobalInvocationID.x; i < total; i += gl_NumWorkGroups.x * gl_WorkGroupSize.x){\n\t\t";
	local_shader += Format(fn_pass, shader_tensor[0].c_str(), shader_tensor[1].c_str(), shader_tensor[2].c_str()) + "\n\t}\n}";
	return local_shader;
}

inline std::string reduction_shader_code_math(const std::string& kernel_shader_code, const std::string& fn_pass, const tensor& t1, const tensor& t2)
{
	std::string local_shader = kernel_shader_code;
	std::string shader_tensor[2];
	const int ext_int_0 = tensor_injection(local_shader, shader_tensor[0], 0, t1);
	const int ext_int_1 = tensor_injection(local_shader, shader_tensor[1], 1, t2);
	if (ext_int_0 == ext_int_1)
		local_shader = shader_extensions[ext_int_0] + local_shader;
	else
	{
		local_shader = shader_extensions[ext_int_0] + local_shader;
		local_shader = shader_extensions[ext_int_1] + local_shader;
	}
	local_shader = SUBGROUP_ENABLE + local_shader;
	local_shader = SHADER_VERSION + local_shader;

	std::string output_type;
	gen_type(t2.get_type(), output_type);

	local_shader = local_shader + "shared " +  output_type + " sdata[32]; " + "\nvoid main()\n{\t\n\t" + output_type + " acc = 0; \n" +
		"\tif(gl_GlobalInvocationID.x < total){\n\t\tacc = " + shader_tensor[0] +
		"[gl_GlobalInvocationID.x];\n\t}\n\tacc = " + fn_pass +
		";\n\n\tif(gl_SubgroupInvocationID == 0)\n\t{\n\t\tsdata[gl_SubgroupID] = acc;\n\t}\n\n\tmemoryBarrierShared();\n\tbarrier();\n\n\tif(gl_SubgroupID==0)\n\t{\n\t\tacc = gl_SubgroupInvocationID < gl_NumSubgroups ? sdata[gl_SubgroupInvocationID] : 0;\n\t\tacc = "
		+ fn_pass + ";\n\t}\n\n\tif(gl_LocalInvocationID.x == 0)\n\t{\n\t\t" + shader_tensor[1] +
		"[gl_WorkGroupID.x] = gl_SubgroupInvocationID;\n\t}\n}";



	return local_shader;
}