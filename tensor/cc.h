// cc.h : Header file for your target.
/*
	COLLECTIVE CALLS
*/

#pragma once
#include "utils.h"
#include "tensor.h"

void all_reduce(){ // each rank receives the reduction of input values across ranks.
	//out[i] = sum(inX[i])
}


void broadcast() { // all ranks receive data from a “root” rank.
	//out[i] = in[i]
}

void reduce(){ //one rank receives the reduction of input values across ranks.
	//out[i] = sum(inX[i])
}

void allgather() { // each rank receives the aggregation of data from all ranks in the order of the ranks.
	//out[rank * count + i] = inY{i]
	
}

void reduceScatter() { //  input values are reduced across ranks, with each rank receiving a subpart of the result.
	//outY[i] = sum(inX[rank * count + i])
}