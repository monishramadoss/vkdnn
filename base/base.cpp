// base.cpp : Source file for your target.
//

#include "base.h"


inline int gen_type(DTYPE type, std::string& type_name)
{
	if (type == BOOL)
	{
		type_name = "bool";
		return 0;
	}
	if (type == HFLOAT)
	{
		type_name = "float16_t";
		return 5;
	}
	if (type == FLOAT)
	{
		type_name = "float";
		return 6;
	}
	if (type == DOUBLE)
	{
		type_name = "float64_t";
		return 7;
	}
	if (type == INT8)
	{
		type_name = "int8_t";
		return 1;
	}
	if (type == HINT)
	{
		type_name = "int16_t";
		return 2;
	}
	if (type == INT)
	{
		type_name = "int32_t";
		return 3;
	}
	if (type == LINT)
	{
		type_name = "int64_t";
		return 4;
	}
	if (type == UINT8)
	{
		type_name = "uint8_t";
		return 1;
	}
	if (type == HUINT)
	{
		type_name = "uint16_t";
		return 2;
	}
	if (type == UINT)
	{
		type_name = "uint32_t";
		return 3;
	}
	if (type == LUINT)
	{
		type_name = "uint64_t";
		return 4;
	}
	if (type == NONE)
	{
		type_name = "";
		return -1;
	}

	return -1;
}

inline int tensor_injection(std::string& body, std::string& var_name, int i, const tensor& t1)
{
	const DTYPE type = t1.getType();
	var_name = "tensor_" + std::to_string(i);

	std::string type_name;
	const int ext_id = gen_type(t1.getType(), type_name);

	std::format_to(std::back_inserter(body), "layout(binding={0}) buffer tensor_{0} { {1} {2}[]; }; ",
	               std::to_string(i), type_name, var_name);

	return ext_id;
}
