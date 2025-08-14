// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once 

#pragma warning(disable : 4996)

typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef int i32;
typedef unsigned short u16;
typedef short i16;
typedef unsigned char u8;
typedef char i8;



#define MAX_GPU_CNT			32

//must be divisible by MD_LEN
#define STEP_CNT			1000

#define JMP_CNT				512

//use different options for cards older than RTX 40xx
#ifdef __CUDA_ARCH__
	#if __CUDA_ARCH__ < 890
		#define OLD_GPU
	#endif
	#ifdef OLD_GPU
		#define BLOCK_SIZE			512
		//can be 8, 16, 24, 32, 40, 48, 56, 64
		#define PNT_GROUP_CNT		64	
	#else
		#define BLOCK_SIZE			256
		//can be 8, 16, 24, 32
		#define PNT_GROUP_CNT		24
	#endif
#else //CPU, fake values
	#define BLOCK_SIZE			512
	#define PNT_GROUP_CNT		64
#endif

// kang type
#define TAME				0  // Tame kangs
#define WILD1				1  // Wild kangs1 
#define WILD2				2  // Wild kangs2

#define GPU_DP_SIZE			48
#define MAX_DP_CNT			(256 * 1024)

#define JMP_MASK			(JMP_CNT-1)

#define DPTABLE_MAX_CNT		16

#define MAX_CNT_LIST		(512 * 1024)

#define DP_FLAG				0x8000
#define INV_FLAG			0x4000

// Structure to hold kernel parameters
// This will be passed from the host to the device
typedef struct 
{
	// Pointers to GPU global memory
	u64* Kangs;
	u64* Points;
	u64* HostPoints;
	u64* Distances;
	
	// Elliptic curve parameters
	u64* Gx;
	u64* Gy;
	u64* P;
	u64* N;
	u64* A;
	u64* B;
	u64* R2P; // R^2 mod P
	u64* InvN; // -N^-1 mod 2^64
	u64* R2N; // R^2 mod N
	u64* Inv256; // 256^-1 mod N

	// Host-side collision detection pointers
	u64* HostKangs;
	u64* HostDistances;

	u32 PointsPerKang;
	u32 KangCount;
	u32 BlockCount;
	u32 ThreadCount;
	u32 KPerBlock;
	u32 Iteration;

	u64 StartPoint[4];
	u64 StopPoint[4];

	// New fields for multiple public keys
	u64* Pubkeys;       // Pointer to the array of public keys on the device
	u32  PubkeyCount;   // The number of public keys
	u64* CollisionResults; // Stores the index and position of found keys

	// Existing kernel parameters
	u32 BlockCnt;
	u32 BlockSize;
	u32 KernelA_LDS_Size;
	u32 KernelB_LDS_Size;
	u32 KernelC_LDS_Size;

} TKparams;

// Point and curve parameters
#define POINT_SIZE_BITS 256
#define POINT_SIZE_BYTES 32
#define POINT_SIZE_U64S 4
#define POINT_DATA_SIZE 8 // 4 for X, 4 for Y
#define POINT_SIZE_WORDS 8 // for 256-bit points
#define POINT_SIZE_DOUBLE_WORDS 16 // for 512-bit intermediate
#define POINTS_PER_THREAD 1
#define STEP_CNT 1024

// This is likely already defined in your code
#define KANG_THREADS 1024
#define KANG_BLOCKS 64
#define KANG_GROUPS 16
#define KANG_PER_GROUP 64

// New collision result struct
typedef struct
{
	u32 pubkeyIndex;
	u64 privateKeyPart;
} CollisionResult;
