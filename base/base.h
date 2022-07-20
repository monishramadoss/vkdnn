// base.h : Header file for your target.

#pragma once
#include "../runtime/runtime.h"
#include "arithmetic.h"

#include <string>
#include <sstream>

inline std::string& tensor_injection(std::string& body, int i, tensor& t1)
{
	DTYPE type = t1.getType();
	std::string var_name = "tensor_" + std::to_string(i);
	char tensor_injection[128];
	std::string type_name;

	switch(type)
	{
	case DTYPE::FLOAT:
		type_name = "float";
	case DTYPE::INT:
		type_name = "int";
	case DTYPE::UINT:
		type_name = "uint";
	case DTYPE::BOOL:
		type_name = "bool";
	case DTYPE::DOUBLE:
		type_name = "double";
	case DTYPE::HFLOAT:
		type_name = "float16_t";
		body = "#extension GL_AMD_gpu_shader_half_float : enable" + body;
	case DTYPE::HINT:
		type_name = "int16_t";
		body = "#extension GL_AMD_gpu_shader_half_float : enable" + body;
	case DTYPE::HUINT:
		type_name = "uint16_t";
		body = "#extension GL_AMD_gpu_shader_half_float : enable" + body;

	}

	const auto n = sprintf(tensor_injection, "layout(binding = %d) readonly buffer buf1 { %s %s[]; };\n", i,
	                       type_name.c_str(), var_name.c_str());
	tensor_injection[n+1] = '\0';
	
	body += tensor_injection;
	return var_name;
}
