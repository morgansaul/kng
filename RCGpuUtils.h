// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC

#pragma once

#include <stdint.h>
#include <device_functions.h>
#include <sm_20_atomic_functions.h>
#include "defs.h"

//PTX asm
//"volatile" is important
#define add_64(res, a, b)				asm volatile ("add.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b)  );
#define add_cc_64(res, a, b)			asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b)  );
#define addc_64(res, a, b)				asm volatile ("addc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b));
#define addc_cc_64(res, a, b)			asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b)  );

#define add_32(res, a, b)				asm volatile ("add.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b)  );
#define add_cc_32(res, a, b)			asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b)  );
#define addc_32(res, a, b)				asm volatile ("addc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
#define addc_cc_32(res, a, b)			asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b)  );

#define sub_64(res, a, b)				asm volatile ("sub.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b));
#define sub_cc_64(res, a, b)			asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b) );
#define subc_cc_64(res, a, b)			asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b)  );
#define subc_64(res, a, b)				asm volatile ("subc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b));

#define sub_32(res, a, b)				asm volatile ("sub.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b) );
#define sub_cc_32(res, a, b)			asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b) );
#define subc_cc_32(res, a, b)			asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b)  );
#define subc_32(res, a, b)				asm volatile ("subc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));

#define mul_lo_64(res, a, b)			asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b));
#define mul_hi_64(res, a, b)			asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b));
#define mad_lo_64(res, a, b, c)			asm volatile ("mad.lo.u64 %0, %1, %2, %3;" : "=l"(res) : "l"(a), "l"(b), "l"(c) );
#define mad_hi_64(res, a, b, c)			asm volatile ("mad.hi.u64 %0, %1, %2, %3;" : "=l"(res) : "l"(a), "l"(b), "l"(c) );
#define mad_lo_cc_64(res, a, b, c)		asm volatile ("mad.lo.cc.u64 %0, %1, %2, %3;" : "=l"(res) : "l"(a), "l"(b), "l"(c) );
#define mad_hi_cc_64(res, a, b, c)		asm volatile ("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(res) : "l"(a), "l"(b), "l"(c) );
#define madc_lo_64(res, a, b, c)		asm volatile ("madc.lo.u64 %0, %1, %2, %3;" : "=l"(res) : "l"(a), "l"(b), "l"(c));
#define madc_hi_64(res, a, b, c)		asm volatile ("madc.hi.u64 %0, %1, %2, %3;" : "=l"(res) : "l"(a), "l"(b), "l"(c));
#define madc_lo_cc_64(res, a, b, c)		asm volatile ("madc.lo.cc.u64 %0, %1, %2, %3;" : "=l"(res) : "l"(a), "l"(b), "l"(c) );
#define madc_hi_cc_64(res, a, b, c)		asm volatile ("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(res) : "l"(a), "l"(b), "l"(c) );

#define mul_lo_32(res, a, b)			asm volatile ("mul.lo.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
#define mul_hi_32(res, a, b)			asm volatile ("mul.hi.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
#define mad_lo_32(res, a, b, c)			asm volatile ("mad.lo.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
#define mad_hi_32(res, a, b, c)			asm volatile ("mad.hi.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
#define mad_lo_cc_32(res, a, b, c)		asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
#define mad_hi_cc_32(res, a, b, c)		asm volatile ("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
#define madc_lo_32(res, a, b, c)		asm volatile ("madc.lo.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
#define madc_hi_32(res, a, b, c)		asm volatile ("madc.hi.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
#define madc_lo_cc_32(res, a, b, c)		asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
#define madc_hi_cc_32(res, a, b, c)		asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));

#define mul_wide_32(res, a, b)			asm volatile ("mul.wide.u32 %0, %1, %2;" : "=l"(res) : "r"(a), "r"(b));
#define mad_wide_32(res,a,b,c)			asm volatile ("mad.wide.u32 %0, %1, %2, %3;" : "=l"(res) : "r"(a), "r"(b), "l"(c) );

#define st_cs_v4_b32(addr,val)			asm volatile("st.cs.global.v4.b32 [%0], {%1, %2, %3, %4};\n":: "l"(addr), "r"((val).x), "r"((val).y), "r"((val).z), "r"((val).w));

#define PTX_SHFL_XOR_I(var, laneMask) asm volatile("shfl.sync.b32 %0, %0, %1, 0x1f; " : "=r"(var) : "r"(laneMask));


//P-related constants
#define P_0			0xFFFFFFFEFFFFFC2Full
#define P_123		0xFFFFFFFFFFFFFFFFull
#define P_INV32		0x000003D1

#define Add192to192(res, val) { \
  add_cc_64((res)[0], (res)[0], (val)[0]); \
  addc_cc_64((res)[1], (res)[1], (val)[1]); \
  addc_64((res)[2], (res)[2], (val)[2]); }

#define Sub192from192(res, val) { \
  sub_cc_64((res)[0], (res)[0], (val)[0]); \
  subc_cc_64((res)[1], (res)[1], (val)[1]); \
  subc_64((res)[2], (res)[2], (val)[2]); }

#define Copy_int4_x2(dst, src) {\
  ((int4*)(dst))[0] = ((int4*)(src))[0]; \
  ((int4*)(dst))[1] = ((int4*)(src))[1]; }

#define Copy_u64_x4(dst, src) {\
  ((u64*)(dst))[0] = ((u64*)(src))[0]; \
  ((u64*)(dst))[1] = ((u64*)(src))[1]; \
  ((u64*)(dst))[2] = ((u64*)(src))[2]; \
  ((u64*)(dst))[3] = ((u64*)(src))[3]; }

__device__ __forceinline__ void NegModP(u64* res)
{
	sub_cc_64(res[0], P_0, res[0]);
	subc_cc_64(res[1], P_123, res[1]);
	subc_cc_64(res[2], P_123, res[2]);
	subc_64(res[3], P_123, res[3]);
}

__device__ __forceinline__ void SubModP(u64* res, u64* val1, u64* val2)
{
	sub_cc_64(res[0], val1[0], val2[0]);
    subc_cc_64(res[1], val1[1], val2[1]);
    subc_cc_64(res[2], val1[2], val2[2]);
    subc_cc_64(res[3], val1[3], val2[3]);
    u32 carry;
    subc_32(carry, 0, 0);
    if (carry)
    { 
		add_cc_64(res[0], res[0], P_0);
		addc_cc_64(res[1], res[1], P_123);
		addc_cc_64(res[2], res[2], P_123);
		addc_64(res[3], res[3], P_123);
    }
}

__device__ __forceinline__ void AddModP(u64* res, u64* val1, u64* val2)
{
	u64 tmp[4];
	u32 carry;
	add_cc_64(tmp[0], val1[0], val2[0]);
	addc_cc_64(tmp[1], val1[1], val2[1]);
	addc_cc_64(tmp[2], val1[2], val2[2]);
	addc_cc_64(tmp[3], val1[3], val2[3]);	
	addc_32(carry, 0, 0);
	Copy_u64_x4(res, tmp);

	sub_cc_64(res[0], res[0], P_0);
	subc_cc_64(res[1], res[1], P_123);
	subc_cc_64(res[2], res[2], P_123);
	subc_cc_64(res[3], res[3], P_123);
	subc_cc_32(carry, carry, 0);
	subc_32(carry, 0, 0);
	if (carry)
		Copy_u64_x4(res, tmp);
}

__device__ __forceinline__ void add_320_to_256(u64* res, u64* val)
{
	add_cc_64(res[0], res[0], val[0]);
	addc_cc_64(res[1], res[1], val[1]);
	addc_cc_64(res[2], res[2], val[2]);
	addc_cc_64(res[3], res[3], val[3]);
	addc_64(res[4], val[4], 0ull);
}

//mul 256bit by 0x1000003D1
__device__ __forceinline__ void mul_256_by_P0inv(u32* res, u32* val)
{
	u64 tmp64[7];
	u32* tmp = (u32*)tmp64;
	mul_wide_32(*(u64*)res, val[0], P_INV32);
	mul_wide_32(tmp64[0], val[1], P_INV32);
	mul_wide_32(tmp64[1], val[2], P_INV32);
	mul_wide_32(tmp64[2], val[3], P_INV32);
	mul_wide_32(tmp64[3], val[4], P_INV32);
	mul_wide_32(tmp64[4], val[5], P_INV32);
	mul_wide_32(tmp64[5], val[6], P_INV32);
	mul_wide_32(tmp64[6], val[7], P_INV32);

	add_cc_32(res[1], res[1], tmp[0]);
	addc_cc_32(res[2], tmp[1], tmp[2]);
	addc_cc_32(res[3], tmp[3], tmp[4]);
	addc_cc_32(res[4], tmp[5], tmp[6]);
	addc_cc_32(res[5], tmp[7], tmp[8]);
	addc_cc_32(res[6], tmp[9], tmp[10]);
	addc_cc_32(res[7], tmp[11], tmp[12]);
	addc_32(res[8], tmp[13], 0); //t[13] cannot be MAX_UINT so we wont have carry here for r[9]

	add_cc_32(res[1], res[1], val[0]);
	addc_cc_32(res[2], res[2], val[1]);
	addc_cc_32(res[3], res[3], val[2]);
	addc_cc_32(res[4], res[4], val[3]);
	addc_cc_32(res[5], res[5], val[4]);
	addc_cc_32(res[6], res[6], val[5]);
	addc_cc_32(res[7], res[7], val[6]);
	addc_cc_32(res[8], res[8], val[7]);
	addc_32(res[9], 0, 0);
}

//mul 256bit by 64bit
__device__ __forceinline__ void mul_256_by_64(u64* res, u64* val256, u64 val64)
{
	u64 tmp64[7];
	u32* tmp = (u32*)tmp64;
	u32* rs = (u32*)res;
	u32* a = (u32*)val256;
	u32* b = (u32*)&val64;
	mul_wide_32(res[0], a[0], b[0]);
	mul_wide_32(tmp64[0], a[1], b[0]);
	mul_wide_32(tmp64[1], a[2], b[0]);
	mul_wide_32(tmp64[2], a[3], b[0]);
	mul_wide_32(tmp64[3], a[4], b[0]);
	mul_wide_32(tmp64[4], a[5], b[0]);
	mul_wide_32(tmp64[5], a[6], b[0]);
	mul_wide_32(tmp64[6], a[7], b[0]);
	
	add_cc_32(rs[1], rs[1], tmp[0]);
	addc_cc_32(rs[2], tmp[1], tmp[2]);
	addc_cc_32(rs[3], tmp[3], tmp[4]);
	addc_cc_32(rs[4], tmp[5], tmp[6]);
	addc_cc_32(rs[5], tmp[7], tmp[8]);
	addc_cc_32(rs[6], tmp[9], tmp[10]);
	addc_cc_32(rs[7], tmp[11], tmp[12]);
	addc_32(rs[8], tmp[13], 0);

	mul_wide_32(tmp64[0], a[0], b[1]);
	mul_wide_32(tmp64[1], a[1], b[1]);
	mul_wide_32(tmp64[2], a[2], b[1]);
	mul_wide_32(tmp64[3], a[3], b[1]);
	mul_wide_32(tmp64[4], a[4], b[1]);
	mul_wide_32(tmp64[5], a[5], b[1]);
	mul_wide_32(tmp64[6], a[6], b[1]);
	mul_wide_32(tmp64[7], a[7], b[1]);
	
	add_cc_32(rs[1], rs[1], tmp[0]);
	addc_cc_32(rs[2], rs[2], tmp[1]);
	addc_cc_32(rs[3], rs[3], tmp[2]);
	addc_cc_32(rs[4], rs[4], tmp[3]);
	addc_cc_32(rs[5], rs[5], tmp[4]);
	addc_cc_32(rs[6], rs[6], tmp[5]);
	addc_cc_32(rs[7], rs[7], tmp[6]);
	addc_cc_32(rs[8], rs[8], tmp[7]);
	addc_32(rs[9], 0, 0);
}

__device__ __forceinline__ void AddModP_320(u64* res, u64* val1, u64* val2)
{
	u32 carry = 0;
	add_cc_64(res[0], val1[0], val2[0]);
	addc_cc_64(res[1], val1[1], val2[1]);
	addc_cc_64(res[2], val1[2], val2[2]);
	addc_cc_64(res[3], val1[3], val2[3]);
	addc_cc_64(res[4], val1[4], val2[4]);
	addc_32(carry, 0, 0);
	if (carry)
	{
		sub_cc_64(res[0], res[0], P_0);
		subc_cc_64(res[1], res[1], P_123);
		subc_cc_64(res[2], res[2], P_123);
		subc_cc_64(res[3], res[3], P_123);
		subc_64(res[4], res[4], 0);
	}
}

__device__ __forceinline__ void MulModP(u64* res, u64* a, u64* b)
{
	u64 m = 0;
	u64 t[8] = { 0 };
	u64 c = 0;
	
	//t[0], c = a[0]*b[0] + 0
	mul_lo_64(t[0], a[0], b[0]);
	mul_hi_64(c, a[0], b[0]);
	
	//t[1], c = a[0]*b[1] + a[1]*b[0] + c
	u64 tmp;
	mad_lo_cc_64(tmp, a[0], b[1], c);
	u64 hi_tmp;
	madc_hi_cc_64(hi_tmp, a[0], b[1], 0);
	madc_lo_cc_64(t[1], a[1], b[0], tmp);
	madc_hi_cc_64(c, a[1], b[0], hi_tmp);

	//t[2], c = a[0]*b[2] + a[1]*b[1] + a[2]*b[0] + c
	madc_lo_cc_64(tmp, a[0], b[2], c);
	madc_hi_cc_64(hi_tmp, a[0], b[2], 0);
	madc_lo_cc_64(tmp, a[1], b[1], tmp);
	madc_hi_cc_64(hi_tmp, a[1], b[1], hi_tmp);
	madc_lo_cc_64(t[2], a[2], b[0], tmp);
	madc_hi_cc_64(c, a[2], b[0], hi_tmp);

	//t[3], c = a[0]*b[3] + a[1]*b[2] + a[2]*b[1] + a[3]*b[0] + c
	madc_lo_cc_64(tmp, a[0], b[3], c);
	madc_hi_cc_64(hi_tmp, a[0], b[3], 0);
	madc_lo_cc_64(tmp, a[1], b[2], tmp);
	madc_hi_cc_64(hi_tmp, a[1], b[2], hi_tmp);
	madc_lo_cc_64(tmp, a[2], b[1], tmp);
	madc_hi_cc_64(hi_tmp, a[2], b[1], hi_tmp);
	madc_lo_cc_64(t[3], a[3], b[0], tmp);
	madc_hi_cc_64(c, a[3], b[0], hi_tmp);

	//t[4], c = a[1]*b[3] + a[2]*b[2] + a[3]*b[1] + c
	madc_lo_cc_64(tmp, a[1], b[3], c);
	madc_hi_cc_64(hi_tmp, a[1], b[3], 0);
	madc_lo_cc_64(tmp, a[2], b[2], tmp);
	madc_hi_cc_64(hi_tmp, a[2], b[2], hi_tmp);
	madc_lo_cc_64(t[4], a[3], b[1], tmp);
	madc_hi_cc_64(c, a[3], b[1], hi_tmp);

	//t[5], c = a[2]*b[3] + a[3]*b[2] + c
	madc_lo_cc_64(tmp, a[2], b[3], c);
	madc_hi_cc_64(hi_tmp, a[2], b[3], 0);
	madc_lo_cc_64(t[5], a[3], b[2], tmp);
	madc_hi_cc_64(c, a[3], b[2], hi_tmp);

	//t[6], c = a[3]*b[3] + c
	madc_lo_cc_64(t[6], a[3], b[3], c);
	madc_hi_cc_64(t[7], a[3], b[3], 0);

	//montgomery reduction
	u32 carry = 0;
	mul_lo_64(m, t[0], ((TKparams*)params)->InvN[0]); //get m from t[0]
	//m = (t[0] * N0') mod 2^64

	//res[0] = (t[0] + m*N0)/2^64
	u64 mN_hi;
	mul_hi_64(mN_hi, m, ((TKparams*)params)->N[0]);
	u64 t_low, t_hi;
	add_cc_64(t_low, t[0], m * ((TKparams*)params)->N[0]);
	addc_cc_64(t_hi, c, mN_hi);
	
	//a_1 = (t[1] + m*N1 + t_hi)
	u64 tmp_a1;
	mul_lo_64(tmp_a1, m, ((TKparams*)params)->N[1]);
	add_cc_64(res[0], t[1], tmp_a1);
	addc_cc_64(res[0], res[0], t_hi);

	//a_2 = (t[2] + m*N2 + a_1_hi)
	u64 tmp_a2;
	mul_lo_64(tmp_a2, m, ((TKparams*)params)->N[2]);
	addc_cc_64(res[1], t[2], tmp_a2);
	addc_cc_64(res[1], res[1], 0);

	//a_3 = (t[3] + m*N3 + a_2_hi)
	u64 tmp_a3;
	mul_lo_64(tmp_a3, m, ((TKparams*)params)->N[3]);
	addc_cc_64(res[2], t[3], tmp_a3);
	addc_cc_64(res[2], res[2], 0);

	//a_4 = (t[4] + m*N4 + a_3_hi)
	u64 tmp_a4;
	mul_lo_64(tmp_a4, m, ((TKparams*)params)->N[4]);
	addc_cc_64(res[3], t[4], tmp_a4);
	addc_cc_64(res[3], res[3], 0);

	//substract p if res > p
	SubModP_320(res, res, ((TKparams*)params)->P);
}

__device__ __forceinline__ void AddPoint(u64* outX, u64* outY, u64* x1, u64* y1, u64* x2, u64* y2, const TKparams* params)
{
	u64 lambda[4], lambda2[4], u[4], v[4];

	if (is_equal(x1, y1, x2, y2))
	{
		// Doubling
		// lambda = (3x1^2 + a) * (2y1)^-1 mod p
		SqrModP(lambda, x1);
		MulModP(lambda2, lambda, 3);
		AddModP(lambda, lambda2, params->A);
		
		MulModP(lambda2, y1, 2);
		InvModP(lambda2);

		MulModP(lambda, lambda, lambda2);
	}
	else
	{
		// Addition
		// lambda = (y2 - y1) * (x2 - x1)^-1 mod p
		SubModP(u, y2, y1);
		SubModP(v, x2, x1);
		InvModP(v);
		MulModP(lambda, u, v);
	}

	// x3 = lambda^2 - x1 - x2 mod p
	SqrModP(lambda2, lambda);
	SubModP(u, lambda2, x1);
	SubModP(outX, u, x2);

	// y3 = lambda(x1 - x3) - y1 mod p
	SubModP(u, x1, outX);
	MulModP(u, lambda, u);
	SubModP(outY, u, y1);
}

// New helper function to compare two elliptic curve points for equality
__device__ __forceinline__ bool is_equal(const u64* val1_x, const u64* val1_y, const u64* val2_x, const u64* val2_y)
{
	for (int i = 0; i < POINT_SIZE_U64S; i++) {
		if (val1_x[i] != val2_x[i]) return false;
	}
	for (int i = 0; i < POINT_SIZE_U64S; i++) {
		if (val1_y[i] != val2_y[i]) return false;
	}
	return true;
}

__device__ __forceinline__ void SqrModP(u64* res, u64* a)
{
	MulModP(res, a, a);
}

__device__ __forceinline__ void InvModP(u64* res)
{
	// Placeholder for modular inverse logic
}
