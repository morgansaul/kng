// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <cuda_runtime.h>
#include <sm_20_atomic_functions.h>

#include "defs.h"
#include "RCGpuUtils.h"

//imp2 table points for KernelA
__device__ __constant__ u64 jmp2_table[8 * JMP_CNT];


#define BLOCK_CNT	gridDim.x
#define BLOCK_X		blockIdx.x
#define THREAD_X	threadIdx.x

//coalescing
#define LOAD_VAL_256(dst, ptr, group) { *((int4*)&(dst)[0]) = *((int4*)&(ptr)[BLOCK_SIZE * 4 * BLOCK_CNT * (group)]); *((int4*)&(dst)[2]) = *((int4*)&(ptr)[2 * BLOCK_SIZE + BLOCK_SIZE * 4 * BLOCK_CNT * (group)]); }
#define SAVE_VAL_256(ptr, src, group) { *((int4*)&(ptr)[BLOCK_SIZE * 4 * BLOCK_CNT * (group)]) = *((int4*)&(src)[0]); *((int4*)&(ptr)[2 * BLOCK_SIZE + BLOCK_SIZE * 4 * BLOCK_CNT * (group)]) = *((int4*)&(src)[2]); }


extern __shared__ u64 LDS[]; 

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef OLD_GPU

//this kernel performs main jumps
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelA(const TKparams Kparams)
{
	u64* L2x = Kparams.L2 + 2 * THREAD_X + 4 * BLOCK_SIZE * BLOCK_X;
	u64* L2y = L2x + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;
	u64* L2s = L2y + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;
	//list of distances of performed jumps for KernelB
	int4* jlist = (int4*)(Kparams.JumpsList + (u64)BLOCK_X * STEP_CNT * PNT_GROUP_CNT * BLOCK_SIZE / 4);
	jlist += (THREAD_X / 32) * 32 * PNT_GROUP_CNT / 8;
	//list of last visited points for KernelC
	u64* x_last0 = Kparams.LastPnts + 2 * THREAD_X + 4 * BLOCK_SIZE * BLOCK_X;
	u64* y_last0 = x_last0 + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;
      
	u64* jmp1_table = LDS; //32KB
	u16* lds_jlist = (u16*)&LDS[8 * JMP_CNT]; //4KB, must be aligned 16bytes

	int i = THREAD_X;
	while (i < JMP_CNT)
    {	
		*(int4*)&jmp1_table[8 * i + 0] = *(int4*)&Kparams.Jumps1[12 * i + 0];
		*(int4*)&jmp1_table[8 * i + 2] = *(int4*)&Kparams.Jumps1[12 * i + 2];
		*(int4*)&jmp1_table[8 * i + 4] = *(int4*)&Kparams.Jumps1[12 * i + 4];
		*(int4*)&jmp1_table[8 * i + 6] = *(int4*)&Kparams.Jumps1[12 * i + 6];
		i += BLOCK_SIZE;
    }

    __syncthreads(); 

	__align__(16) u64 x[4], y[4], tmp[4], tmp2[4];
	u64 dp_mask64 = ~((1ull << (64 - Kparams.DP)) - 1);
	u16 jmp_ind;

	//copy kangs from global to L2
	u32 kang_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE);
	for (u32 group = 0; group < PNT_GROUP_CNT; group++)
	{	
		tmp[0] = Kparams.Kangs[(kang_ind + group) * 12 + 0];
		tmp[1] = Kparams.Kangs[(kang_ind + group) * 12 + 1];
		tmp[2] = Kparams.Kangs[(kang_ind + group) * 12 + 2];
		tmp[3] = Kparams.Kangs[(kang_ind + group) * 12 + 3];
		SAVE_VAL_256(L2x, tmp, group);
		tmp[0] = Kparams.Kangs[(kang_ind + group) * 12 + 4];
		tmp[1] = Kparams.Kangs[(kang_ind + group) * 12 + 5];
		tmp[2] = Kparams.Kangs[(kang_ind + group) * 12 + 6];
		tmp[3] = Kparams.Kangs[(kang_ind + group) * 12 + 7];
		SAVE_VAL_256(L2y, tmp, group);
	}

	u32 L1S2 = Kparams.L1S2[BLOCK_X * BLOCK_SIZE + THREAD_X];
	u64 current_distance[PNT_GROUP_CNT]; // New array to hold distances
	for (u32 group = 0; group < PNT_GROUP_CNT; group++) {
        current_distance[group] = Kparams.Kangs[(kang_ind + group) * 12 + 8];
    }
    
    for (int step_ind = 0; step_ind < STEP_CNT; step_ind++)
    {
        __align__(16) u64 inverse[5];
		u64* jmp_table;
		__align__(16) u64 jmp_x[4];
		__align__(16) u64 jmp_y[4];
		
		//first group
		LOAD_VAL_256(x, L2x, 0);
		jmp_ind = x[0] % JMP_CNT;
		jmp_table = ((L1S2 >> 0) & 1) ? jmp2_table : jmp1_table;
		Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
		SubModP(inverse, x, jmp_x);
		SAVE_VAL_256(L2s, inverse, 0);
		//the rest
		for (int group = 1; group < PNT_GROUP_CNT; group++)
		{
			LOAD_VAL_256(x, L2x, group);
			jmp_ind = x[0] % JMP_CNT;
			jmp_table = ((L1S2 >> group) & 1) ? jmp2_table : jmp1_table;
			Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
			SubModP(tmp, x, jmp_x);
			MulModP(inverse, inverse, tmp);
			SAVE_VAL_256(L2s, inverse, group);
		}

		InvModP((u32*)inverse);

        for (int group = PNT_GROUP_CNT - 1; group >= 0; group--)
        {
            __align__(16) u64 x0[4];
            __align__(16) u64 y0[4];
            __align__(16) u64 dxs[4];

			LOAD_VAL_256(x0, L2x, group);
            LOAD_VAL_256(y0, L2y, group);
			jmp_ind = x0[0] % JMP_CNT;
			jmp_table = ((L1S2 >> group) & 1) ? jmp2_table : jmp1_table;
			Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
			Copy_int4_x2(jmp_y, jmp_table + 8 * jmp_ind + 4);
			u32 inv_flag = (u32)y0[0] & 1;
			if (inv_flag)
			{
				jmp_ind |= INV_FLAG;
				NegModP(jmp_y);
			}
            if (group)
            {
				LOAD_VAL_256(tmp, L2s, group - 1);
				SubModP(tmp2, x0, jmp_x);
				MulModP(dxs, tmp, inverse);
				MulModP(inverse, inverse, tmp2);
            }
			else
				Copy_u64_x4(dxs, inverse);

			SubModP(tmp2, y0, jmp_y);
			MulModP(tmp, tmp2, dxs);
			SqrModP(tmp2, tmp);

			SubModP(x, tmp2, jmp_x);
			SubModP(x, x, x0);
			SAVE_VAL_256(L2x, x, group);

			SubModP(y, x0, x);
			MulModP(y, y, tmp);
			SubModP(y, y, y0);
			SAVE_VAL_256(L2y, y, group);
			
            // Update distance
            u64 jmp_dist = (u64)(jmp_ind & JMP_MASK) + 1;
            current_distance[group] += jmp_dist;

			// NEW: Collision detection loop
			for (u32 pubkey_idx = 0; pubkey_idx < Kparams.PubkeyCount; pubkey_idx++) 
			{
				u64 pubkey_x[4], pubkey_y[4];
				Copy_u64_x4(pubkey_x, &Kparams.Pubkeys[pubkey_idx * POINT_DATA_SIZE]);
				Copy_u64_x4(pubkey_y, &Kparams.Pubkeys[pubkey_idx * POINT_DATA_SIZE + POINT_SIZE_U64S]);

				if (is_equal(x, y, pubkey_x, pubkey_y))
				{
					u32 collision_idx = atomicAdd(Kparams.CollisionResults, 1);
					
					// Store the public key index and the private key part
					if (collision_idx < (Kparams.KangCnt - 1)) {
						((CollisionResult*)Kparams.CollisionResults)[collision_idx + 1].pubkeyIndex = pubkey_idx;
						((CollisionResult*)Kparams.CollisionResults)[collision_idx + 1].privateKeyPart = current_distance[group];
					}
				}
			}

			if (((L1S2 >> group) & 1) == 0) //normal mode, check L1S2 loop
			{
				u32 jmp_next = x[0] % JMP_CNT;
				jmp_next |= ((u32)y[0] & 1) ? 0 : INV_FLAG; //inverted
				L1S2 |= (jmp_ind == jmp_next) ? (1u << group) : 0; //loop L1S2 detected
			}
			else
			{
				L1S2 &= ~(1u << group);
				jmp_ind |= JMP2_FLAG;
			}
			
			if ((x[3] & dp_mask64) == 0)
			{
				u32 kang_ind_dp = (THREAD_X + BLOCK_X * BLOCK_SIZE) * PNT_GROUP_CNT + group;
				u32 ind = atomicAdd(Kparams.DPTable + kang_ind_dp, 1);
				ind = min(ind, DPTABLE_MAX_CNT - 1);
				int4* dst = (int4*)(Kparams.DPTable + Kparams.KangCnt + (kang_ind_dp * DPTABLE_MAX_CNT + ind) * 4);
				dst[0] = ((int4*)x)[0];
				jmp_ind |= DP_FLAG;
			}

			lds_jlist[8 * THREAD_X + (group % 8)] = jmp_ind;
			if ((group % 8) == 0)
				st_cs_v4_b32(&jlist[(group / 8) * 32 + (THREAD_X % 32)], *(int4*)&lds_jlist[8 * THREAD_X]); //skip L2 cache

			if (step_ind + Kparams.MD_LEN >= STEP_CNT) //store last kangs to be able to find loop exit point
			{
				int n = step_ind + Kparams.MD_LEN - STEP_CNT;
				u64* x_last = x_last0 + n * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
				u64* y_last = y_last0 + n * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
				SAVE_VAL_256(x_last, x, group);
				SAVE_VAL_256(y_last, y, group);
			}
        }
		jlist += PNT_GROUP_CNT * BLOCK_SIZE / 8;
    } 

	Kparams.L1S2[BLOCK_X * BLOCK_SIZE + THREAD_X] = L1S2;
	//copy kangs from L2 to global
	kang_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE);
	for (u32 group = 0; group < PNT_GROUP_CNT; group++)
	{
		LOAD_VAL_256(tmp, L2x, group);
		Kparams.Kangs[(kang_ind + group) * 12 + 0] = tmp[0];
		Kparams.Kangs[(kang_ind + group) * 12 + 1] = tmp[1];
		Kparams.Kangs[(kang_ind + group) * 12 + 2] = tmp[2];
		Kparams.Kangs[(kang_ind + group) * 12 + 3] = tmp[3];
		LOAD_VAL_256(tmp, L2y, group);
		Kparams.Kangs[(kang_ind + group) * 12 + 4] = tmp[0];
		Kparams.Kangs[(kang_ind + group) * 12 + 5] = tmp[1];
		Kparams.Kangs[(kang_ind + group) * 12 + 6] = tmp[2];
		Kparams.Kangs[(kang_ind + group) * 12 + 7] = tmp[3];
		Kparams.Kangs[(kang_ind + group) * 12 + 8] = current_distance[group];
	}
} 

#else

#define LOAD_VAL_256_m(dst,p,i) { *((int4*)&(dst)[0]) = *((int4*)&(p)[4 * (i)]); *((int4*)&(dst)[2]) = *((int4*)&(p)[4 * (i) + 2]); }
#define SAVE_VAL_256_m(p,src,i) { *((int4*)&(p)[4 * (i)]) = *((int4*)&(src)[0]); *((int4*)&(p)[4 * (i) + 2]) = *((int4*)&(src)[2]); }


//this kernel performs main jumps for old cards
//not good but works
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelA(const TKparams Kparams)
{
	__align__(16) u64 Lx[4 * PNT_GROUP_CNT];
	__align__(16) u64 Ly[4 * PNT_GROUP_CNT];
	__align__(16) u64 Ls[4 * PNT_GROUP_CNT / 2]; //we store only half so need only half mem

	//list of distances of performed jumps for KernelB
	int4* jlist = (int4*)(Kparams.JumpsList + (u64)BLOCK_X * STEP_CNT * PNT_GROUP_CNT * BLOCK_SIZE / 4);
	jlist += (THREAD_X / 32) * 32 * PNT_GROUP_CNT / 8;
	//list of last visited points for KernelC
	u64* x_last0 = Kparams.LastPnts + 2 * THREAD_X + 4 * BLOCK_SIZE * BLOCK_X;
	u64* y_last0 = x_last0 + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;

	u64* jmp1_table = LDS; //32KB
	u16* lds_jlist = (u16*)&LDS[8 * JMP_CNT]; //8KB, must be aligned 16bytes

	int i = THREAD_X;
	while (i < JMP_CNT)
	{
		*(int4*)&jmp1_table[8 * i + 0] = *(int4*)&Kparams.Jumps1[12 * i + 0];
		*(int4*)&jmp1_table[8 * i + 2] = *(int4*)&Kparams.Jumps1[12 * i + 2];
		*(int4*)&jmp1_table[8 * i + 4] = *(int4*)&Kparams.Jumps1[12 * i + 4];
		*(int4*)&jmp1_table[8 * i + 6] = *(int4*)&Kparams.Jumps1[12 * i + 6];
		i += BLOCK_SIZE;
	}

	__syncthreads();

	__align__(16) u64 inverse[5];
	__align__(16) u64 x[4], y[4], tmp[4], tmp2[4];
	u64 dp_mask64 = ~((1ull << (64 - Kparams.DP)) - 1);
	u16 jmp_ind;

	//copy kangs from global to local
	u32 kang_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE);
	for (u32 group = 0; group < PNT_GROUP_CNT; group++)
	{
		tmp[0] = Kparams.Kangs[(kang_ind + group) * 12 + 0];
		tmp[1] = Kparams.Kangs[(kang_ind + group) * 12 + 1];
		tmp[2] = Kparams.Kangs[(kang_ind + group) * 12 + 2];
		tmp[3] = Kparams.Kangs[(kang_ind + group) * 12 + 3];
		SAVE_VAL_256_m(Lx, tmp, group);
		tmp[0] = Kparams.Kangs[(kang_ind + group) * 12 + 4];
		tmp[1] = Kparams.Kangs[(kang_ind + group) * 12 + 5];
		tmp[2] = Kparams.Kangs[(kang_ind + group) * 12 + 6];
		tmp[3] = Kparams.Kangs[(kang_ind + group) * 12 + 7];
		SAVE_VAL_256_m(Ly, tmp, group);
	}
	
	u64 L1S2 = ((u64*)Kparams.L1S2)[BLOCK_X * BLOCK_SIZE + THREAD_X];
	u64* jmp_table;
	__align__(16) u64 jmp_x[4];
	__align__(16) u64 jmp_y[4];
	
	u64 current_distance[PNT_GROUP_CNT]; // New array to hold distances
	for (u32 group = 0; group < PNT_GROUP_CNT; group++) {
        current_distance[group] = Kparams.Kangs[(kang_ind + group) * 12 + 8];
    }

	//preparations (first calc for inv)
	for (int group = 0; group < PNT_GROUP_CNT; group++)
	{
		LOAD_VAL_256_m(x, Lx, group);
		jmp_ind = x[0] % JMP_CNT;
		jmp_table = ((L1S2 >> group) & 1) ? jmp2_table : jmp1_table;
		Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
		SubModP(tmp, x, jmp_x);
		if (group == 0)
		{
			Copy_u64_x4(inverse, tmp);
			SAVE_VAL_256_m(Ls, tmp, 0);
		}
		else
		{
			MulModP(inverse, inverse, tmp);
			if ((group & 1) == 0)
				SAVE_VAL_256_m(Ls, inverse, group / 2);
		}
	}

	//main loop
	int g_beg = PNT_GROUP_CNT - 1; //start val
	int g_end = -1; //first val after range
	int g_inc = -1;
	int s_mask = 1;
	int jlast_add = 0;
	__align__(16) u64 t_cache[4], x0_cache[4], jmpx_cached[4];
	t_cache[0] = t_cache[1] = t_cache[2] = t_cache[3] = 0;
	x0_cache[0] = x0_cache[1] = x0_cache[2] = x0_cache[3] = 0;
	for (int step_ind = 0; step_ind < STEP_CNT; step_ind++)
	{
		__align__(16) u64 next_inv[4];
		InvModP((u32*)inverse);
		int group = g_beg;
		bool cached = false;
		while (group != g_end)
		{
			__align__(16) u64 dx[4], x0[4], y0[4], dx0[4];
			if (cached)
			{
				Copy_u64_x4(x0, x0_cache);
			}
			else
			{
				LOAD_VAL_256_m(x0, Lx, group);
			}
			LOAD_VAL_256_m(y0, Ly, group);
			jmp_ind = x0[0] % JMP_CNT;
			jmp_table = ((L1S2 >> group) & 1) ? jmp2_table : jmp1_table;
			if (cached)
			{
				Copy_u64_x4(jmp_x, jmpx_cached);
			}
			else
			{
				Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
			}
			Copy_int4_x2(jmp_y, jmp_table + 8 * jmp_ind + 4);
			u32 inv_flag = (u32)y0[0] & 1;
			if (inv_flag)
			{
				jmp_ind |= INV_FLAG;
				NegModP(jmp_y);
			}
			if (group == g_end - g_inc)
			{
				Copy_u64_x4(dx, inverse);
			}
			else
			{
				if (((group - g_inc) & 1) == 0)
				{
					if (cached)
					{
						MulModP(dx, t_cache, inverse);
					}
					else
					{
						LOAD_VAL_256_m(tmp, Ls, (group - g_inc) / 2);
						MulModP(dx, tmp, inverse);
					}
				}
				else
				{
					if (cached)
					{
						MulModP(tmp, t_cache, inverse);
					}
					else
					{
						LOAD_VAL_256_m(tmp, Ls, (group - g_inc) / 2);
						MulModP(tmp, tmp, inverse);
					}
					SubModP(tmp2, x0, jmp_x);
					MulModP(dx, tmp, tmp2);
				}
				
			}
			if ((group == g_beg) || ((group & 1) != 0))
			{
				SubModP(tmp, x0, jmp_x);
				if ((group == g_beg) || (group == g_end + g_inc))
					Copy_u64_x4(inverse, tmp);
				else if (s_mask)
				{
					LOAD_VAL_256_m(tmp2, Ls, (group - g_inc) / 2);
					MulModP(inverse, tmp2, tmp);
					s_mask = 0;
				}
				else
				{
					MulModP(inverse, inverse, tmp);
					if (((group + g_inc) & 1) == 0)
					{
						SAVE_VAL_256_m(Ls, inverse, (group + g_inc) / 2);
						s_mask = 1;
					}
				}
			}

			SubModP(tmp2, y0, jmp_y);
			MulModP(tmp, tmp2, dx);
			SqrModP(tmp2, tmp);

			SubModP(x, tmp2, jmp_x);
			SubModP(x, x, x0);
			SAVE_VAL_256_m(Lx, x, group);

			SubModP(y, x0, x);
			MulModP(y, y, tmp);
			SubModP(y, y, y0);
			SAVE_VAL_256_m(Ly, y, group);

            // Update distance
            u64 jmp_dist = (u64)(jmp_ind & JMP_MASK) + 1;
            current_distance[group] += jmp_dist;

			// NEW: Collision detection loop
			for (u32 pubkey_idx = 0; pubkey_idx < Kparams.PubkeyCount; pubkey_idx++) 
			{
				u64 pubkey_x[4], pubkey_y[4];
				Copy_u64_x4(pubkey_x, &Kparams.Pubkeys[pubkey_idx * POINT_DATA_SIZE]);
				Copy_u64_x4(pubkey_y, &Kparams.Pubkeys[pubkey_idx * POINT_DATA_SIZE + POINT_SIZE_U64S]);

				if (is_equal(x, y, pubkey_x, pubkey_y))
				{
					u32 collision_idx = atomicAdd(Kparams.CollisionResults, 1);
					
					// Store the public key index and the private key part
					if (collision_idx < (Kparams.KangCnt - 1)) {
						((CollisionResult*)Kparams.CollisionResults)[collision_idx + 1].pubkeyIndex = pubkey_idx;
						((CollisionResult*)Kparams.CollisionResults)[collision_idx + 1].privateKeyPart = current_distance[group];
					}
				}
			}

			if (((L1S2 >> group) & 1) == 0) //normal mode, check L1S2 loop
			{
				u32 jmp_next = x[0] % JMP_CNT;
				jmp_next |= ((u32)y[0] & 1) ? 0 : INV_FLAG; //inverted
				L1S2 |= (jmp_ind == jmp_next) ? (1u << group) : 0; //loop L1S2 detected
			}
			else
			{
				L1S2 &= ~(1u << group);
				jmp_ind |= JMP2_FLAG;
			}
			
			if ((x[3] & dp_mask64) == 0)
			{
				u32 kang_ind_dp = (THREAD_X + BLOCK_X * BLOCK_SIZE) * PNT_GROUP_CNT + group;
				u32 ind = atomicAdd(Kparams.DPTable + kang_ind_dp, 1);
				ind = min(ind, DPTABLE_MAX_CNT - 1);
				int4* dst = (int4*)(Kparams.DPTable + Kparams.KangCnt + (kang_ind_dp * DPTABLE_MAX_CNT + ind) * 4);
				dst[0] = ((int4*)x)[0];
				jmp_ind |= DP_FLAG;
			}

			lds_jlist[8 * THREAD_X + (group % 8)] = jmp_ind;
			if ((group % 8) == 0)
				st_cs_v4_b32(&jlist[(group / 8) * 32 + (THREAD_X % 32)], *(int4*)&lds_jlist[8 * THREAD_X]); //skip L2 cache

			if (step_ind + Kparams.MD_LEN >= STEP_CNT) //store last kangs to be able to find loop exit point
			{
				int n = step_ind + Kparams.MD_LEN - STEP_CNT;
				u64* x_last = x_last0 + n * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
				u64* y_last = y_last0 + n * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
				SAVE_VAL_256(x_last, x, group);
				SAVE_VAL_256(y_last, y, group);
			}

			group += g_inc;
		}
		jlist += PNT_GROUP_CNT * BLOCK_SIZE / 8;

		//next loop prep
		for (int group = PNT_GROUP_CNT - 1; group >= 0; group--)
		{
			LOAD_VAL_256_m(x, Lx, group);
			jmp_ind = x[0] % JMP_CNT;
			jmp_table = ((L1S2 >> group) & 1) ? jmp2_table : jmp1_table;
			Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
			SubModP(tmp, x, jmp_x);
			if (group == PNT_GROUP_CNT - 1)
			{
				Copy_u64_x4(inverse, tmp);
			}
			else
			{
				MulModP(inverse, inverse, tmp);
			}
		}
	}
	
	((u64*)Kparams.L1S2)[BLOCK_X * BLOCK_SIZE + THREAD_X] = L1S2;
	//copy kangs from local to global
	kang_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE);
	for (u32 group = 0; group < PNT_GROUP_CNT; group++)
	{
		LOAD_VAL_256_m(tmp, Lx, group);
		Kparams.Kangs[(kang_ind + group) * 12 + 0] = tmp[0];
		Kparams.Kangs[(kang_ind + group) * 12 + 1] = tmp[1];
		Kparams.Kangs[(kang_ind + group) * 12 + 2] = tmp[2];
		Kparams.Kangs[(kang_ind + group) * 12 + 3] = tmp[3];
		LOAD_VAL_256_m(tmp, Ly, group);
		Kparams.Kangs[(kang_ind + group) * 12 + 4] = tmp[0];
		Kparams.Kangs[(kang_ind + group) * 12 + 5] = tmp[1];
		Kparams.Kangs[(kang_ind + group) * 12 + 6] = tmp[2];
		Kparams.Kangs[(kang_ind + group) * 12 + 7] = tmp[3];
		Kparams.Kangs[(kang_ind + group) * 12 + 8] = current_distance[group];
	}
}

#endif //OLD_GPU

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelB(const TKparams Kparams)
{
	// Existing KernelB logic not shown here
}

extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelC(const TKparams Kparams)
{
	// Existing KernelC logic not shown here
}

extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelGen(const TKparams Kparams)
{
	// Existing KernelGen logic not shown here
}

void CallGpuKernelABC(TKparams Kparams)
{
	cudaMemsetAsync(Kparams.CollisionResults, 0, sizeof(CollisionResult) * Kparams.KangCnt, 0);

	KernelA <<< Kparams.BlockCnt, Kparams.BlockSize >>> (Kparams);
	KernelB <<< Kparams.BlockCnt, Kparams.BlockSize >>> (Kparams);
	KernelC <<< Kparams.BlockCnt, Kparams.BlockSize >>> (Kparams);
}

void CallGpuKernelGen(TKparams Kparams)
{
	KernelGen <<< Kparams.BlockCnt, Kparams.BlockSize >>> (Kparams);
}

cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table)
{
	cudaError_t err = cudaSuccess;
	err = cudaFuncSetAttribute(KernelA, cudaFuncAttributeMaxDynamicSharedMemorySize, Kparams.KernelA_LDS_Size);
	if (err != cudaSuccess) return err;
	err = cudaFuncSetAttribute(KernelB, cudaFuncAttributeMaxDynamicSharedMemorySize, Kparams.KernelB_LDS_Size);
	if (err != cudaSuccess) return err;
	err = cudaFuncSetAttribute(KernelC, cudaFuncAttributeMaxDynamicSharedMemorySize, Kparams.KernelC_LDS_Size);
	if (err != cudaSuccess) return err;
	err = cudaMemcpyToSymbol(jmp2_table, _jmp2_table, 8 * JMP_CNT * sizeof(u64), 0, cudaMemcpyHostToDevice);
	return err;
}
