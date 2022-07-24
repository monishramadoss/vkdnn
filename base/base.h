// base.h : Header file for your target.

#pragma once
#include "../runtime/runtime.h"
#include "../tensor/tensor.h"

#include <string>
#include <format>
#include <sstream>
/*
*
* GL_EXT_shader_explicit_arithmetic_types_int8
* GL_EXT_shader_explicit_arithmetic_types_int16
* GL_EXT_shader_explicit_arithmetic_types_int32
* GL_EXT_shader_explicit_arithmetic_types_int64
* GL_EXT_shader_explicit_arithmetic_types_float16
* GL_EXT_shader_explicit_arithmetic_types_float32
* GL_EXT_shader_explicit_arithmetic_types_float64
*/

inline const char* shader_extensions[]{
	"",																		// 0
	"#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable",		// 1
	"#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable",	// 2 
	"#extension GL_EXT_shader_explicit_arithmetic_types_int32: enable",		// 3
	"#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable",	// 4
	"#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable",	// 5
	"#extension GL_EXT_shader_explicit_arithmetic_types_float32 : enable",	// 6
	"#extension GL_EXT_shader_explicit_arithmetic_types_float64 : enable"	// 7
};


inline int gen_type(DTYPE type, std::string& type_name);




inline int tensor_injection(std::string& body, std::string& var_name, int i, const tensor& t1);