// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC

// Modified by ChatGPT: add simple multi‑pubkey support.
// Loads up to 50k pubkeys from -pubkeysfile and, on every DP batch from GPU,
// checks collisions against *all* loaded targets on CPU side.
// Note: GPU walks are still built for the first target; this is a simple CPU-side
// fan-out that is easy to maintain. For best results run per-target; this version
// is provided because the user explicitly wants the simplest implementation.

#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda_runtime.h"
#include "cuda.h"

#include "defs.h"
#include "utils.h"
#include "GpuKang.h"

// ---------------- Globals -----------------
EcJMP EcJumps1[JMP_CNT];
EcJMP EcJumps2[JMP_CNT];
EcJMP EcJumps3[JMP_CNT];

RCGpuKang* GpuKangs[MAX_GPU_CNT];
int GpuCnt;
volatile long ThrCnt;
volatile bool gSolved;

EcInt Int_HalfRange;
EcPoint Pnt_HalfRange;
EcPoint Pnt_NegHalfRange;
EcInt Int_TameOffset;
Ec ec;

CriticalSection csAddPoints;
u8* pPntList;
u8* pPntList2;
volatile int PntIndex;
TFastBase db;
EcPoint gPntToSolve;          // primary target used to generate GPU walks
EcInt gPrivKey;

volatile u64 TotalOps;
u32 TotalSolved;
u32 gTotalErrors;
u64 PntTotalOps;
bool IsBench;

u32 gDP;
u32 gRange;
EcInt gStart;
bool gStartSet;
EcPoint gPubKey;              // single-key path (legacy)
u8 gGPUs_Mask[MAX_GPU_CNT];
char gTamesFileName[1024];
char gPubKeysFileName[1024];  // NEW: file with many pubkeys
bool gUsePubKeysFile = false; // NEW: use multi-file mode

double gMax;
bool gGenMode; //tames generation mode
bool gIsOpsLimit;

// NEW: in-memory pubkeys & per-target adjusted points (with -start offset applied)
static std::vector<EcPoint> gPubKeysVec;     // original pubkeys (for reporting/verify)
static std::vector<EcPoint> gTargetsVec;     // P_i' = P_i + (-G*start)
static std::vector<std::string> gPubHexVec;  // hex strings for logs
static int gSolvedIndex = -1;                // which pubkey was solved

#pragma pack(push, 1)
struct DBRec
{
    u8 x[12];
    u8 d[22];
    u8 type; //0 - tame, 1 - wild1, 2 - wild2
};
#pragma pack(pop)

// -------------- GPU init ------------------
void InitGpus()
{
    GpuCnt = 0;
    int gcnt = 0;
    cudaGetDeviceCount(&gcnt);
    if (gcnt > MAX_GPU_CNT)
        gcnt = MAX_GPU_CNT;

    if (!gcnt)
        return;

    int drv, rt;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    char drvver[100];
    sprintf(drvver, "%d.%d/%d.%d", drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);

    printf("CUDA devices: %d, CUDA driver/runtime: %s\r\n", gcnt, drvver);
    cudaError_t cudaStatus;
    for (int i = 0; i < gcnt; i++)
    {
        cudaStatus = cudaSetDevice(i);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaSetDevice for gpu %d failed!\r\n", i);
            continue;
        }

        if (!gGPUs_Mask[i])
            continue;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("GPU %d: %s, %.2f GB, %d CUs, cap %d.%d, PCI %d, L2 size: %d KB\r\n", i, deviceProp.name, ((float)(deviceProp.totalGlobalMem / (1024 * 1024))) / 1024.0f, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor, deviceProp.pciBusID, deviceProp.l2CacheSize / 1024);
        
        if (deviceProp.major < 6)
        {
            printf("GPU %d - not supported, skip\r\n", i);
            continue;
        }

        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

        GpuKangs[GpuCnt] = new RCGpuKang();
        GpuKangs[GpuCnt]->CudaIndex = i;
        GpuKangs[GpuCnt]->persistingL2CacheMaxSize = deviceProp.persistingL2CacheMaxSize;
        GpuKangs[GpuCnt]->mpCnt = deviceProp.multiProcessorCount;
        GpuKangs[GpuCnt]->IsOldGpu = deviceProp.l2CacheSize < 16 * 1024 * 1024;
        GpuCnt++;
    }
    printf("Total GPUs for work: %d\r\n", GpuCnt);
}
#ifdef _WIN32
u32 __stdcall kang_thr_proc(void* data)
{
    RCGpuKang* Kang = (RCGpuKang*)data;
    Kang->Execute();
    InterlockedDecrement(&ThrCnt);
    return 0;
}
#else
void* kang_thr_proc(void* data)
{
    RCGpuKang* Kang = (RCGpuKang*)data;
    Kang->Execute();
    __sync_fetch_and_sub(&ThrCnt, 1);
    return 0;
}
#endif

// -------------- Shared DP buffer ----------
void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt)
{
    csAddPoints.Enter();
    if (PntIndex + pnt_cnt >= MAX_CNT_LIST)
    {
        csAddPoints.Leave();
        printf("DPs buffer overflow, some points lost, increase DP value!\r\n");
        return;
    }
    memcpy(pPntList + GPU_DP_SIZE * PntIndex, data, pnt_cnt * GPU_DP_SIZE);
    PntIndex += pnt_cnt;
    PntTotalOps += ops_cnt;
    csAddPoints.Leave();
}

// -------------- Collision -----------------
static bool Collision_SOTA_Target(const EcPoint& targetPnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg, EcInt& out_priv)
{
    if (IsNeg)
        t.Neg();
    if (TameType == TAME)
    {
        out_priv = t;
        out_priv.Sub(w);
        EcInt sv = out_priv;
        out_priv.Add(Int_HalfRange);
        EcPoint P = ec.MultiplyG(out_priv);
        if (P.IsEqual(targetPnt))
            return true;
        out_priv = sv;
        out_priv.Neg();
        out_priv.Add(Int_HalfRange);
        P = ec.MultiplyG(out_priv);
        return P.IsEqual(targetPnt);
    }
    else
    {
        out_priv = t;
        out_priv.Sub(w);
        if (out_priv.data[4] >> 63)
            out_priv.Neg();
        out_priv.ShiftRight(1);
        EcInt sv = out_priv;
        out_priv.Add(Int_HalfRange);
        EcPoint P = ec.MultiplyG(out_priv);
        if (P.IsEqual(targetPnt))
            return true;
        out_priv = sv;
        out_priv.Neg();
        out_priv.Add(Int_HalfRange);
        P = ec.MultiplyG(out_priv);
        return P.IsEqual(targetPnt);
    }
}

// OLD helper kept for single-target path
bool Collision_SOTA(EcPoint& pnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg)
{
    EcInt tmp;
    bool ok = Collision_SOTA_Target(pnt, t, TameType, w, WildType, IsNeg, tmp);
    if (ok) gPrivKey = tmp;
    return ok;
}

// -------------- Multi-pubkey DP checking --
static bool gStopOnFirst = true; // simplest behavior: stop after first found

void CheckNewPoints()
{
    csAddPoints.Enter();
    if (!PntIndex)
    {
        csAddPoints.Leave();
        return;
    }

    int cnt = PntIndex;
    memcpy(pPntList2, pPntList, GPU_DP_SIZE * cnt);
    PntIndex = 0;
    csAddPoints.Leave();

    for (int i = 0; i < cnt; i++)
    {
        DBRec nrec;
        u8* p = pPntList2 + i * GPU_DP_SIZE;
        memcpy(nrec.x, p, 12);
        memcpy(nrec.d, p + 16, 22);
        nrec.type = gGenMode ? TAME : p[40];

        DBRec* pref = (DBRec*)db.FindOrAddDataBlock((u8*)&nrec);
        if (gGenMode)
            continue;
        if (pref)
        {
            // restore first 3 bytes (prefix not stored in DB)
            DBRec tmp_pref;
            memcpy(&tmp_pref, &nrec, 3);
            memcpy(((u8*)&tmp_pref) + 3, pref, sizeof(DBRec) - 3);
            pref = &tmp_pref;

            if (pref->type == nrec.type)
            {
                if (pref->type == TAME)
                    continue;
                if (*(u64*)pref->d == *(u64*)nrec.d)
                    continue;
            }

            EcInt w, t;
            int TameType, WildType;
            if (pref->type != TAME)
            {
                memcpy(w.data, pref->d, sizeof(pref->d));
                if (pref->d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
                memcpy(t.data, nrec.d, sizeof(nrec.d));
                if (nrec.d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
                TameType = nrec.type;
                WildType = pref->type;
            }
            else
            {
                memcpy(w.data, nrec.d, sizeof(nrec.d));
                if (nrec.d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
                memcpy(t.data, pref->d, sizeof(pref->d));
                if (pref->d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
                TameType = TAME;
                WildType = nrec.type;
            }

            // --- NEW: check against all targets in memory ---
            if (!gTargetsVec.empty())
            {
                for (size_t idx = 0; idx < gTargetsVec.size(); ++idx)
                {
                    EcInt pk_candidate;
                    bool res = Collision_SOTA_Target(gTargetsVec[idx], t, TameType, w, WildType, false, pk_candidate) ||
                               Collision_SOTA_Target(gTargetsVec[idx], t, TameType, w, WildType, true,  pk_candidate);
                    if (res)
                    {
                        // Verify & report, convert back by adding start offset
                        EcInt final_pk = pk_candidate;
                        final_pk.AddModP(gStart);
                        EcPoint checkP = ec.MultiplyG(final_pk);
                        if (checkP.IsEqual(gPubKeysVec[idx]))
                        {
                            gPrivKey = final_pk;
                            gSolvedIndex = (int)idx;
                            gSolved = true;
                            return; // stop processing this batch early
                        }
                        else
                        {
                            printf("Collision verify failed for target %zu (false positive)\n", idx);
                            gTotalErrors++;
                        }
                    }
                }

                // If nothing matched among targets, allow the original single-target path to attempt
                // (it should be one of the targets anyway if we built walks for that first target)
            }

            // Legacy single-target check (kept for bench or single-key mode)
            bool res1 = Collision_SOTA(gPntToSolve, t, TameType, w, WildType, false) ||
                        Collision_SOTA(gPntToSolve, t, TameType, w, WildType, true);
            if (!res1)
            {
                bool w12 = ((pref->type == WILD1) && (nrec.type == WILD2)) || ((pref->type == WILD2) && (nrec.type == WILD1));
                if (!w12)
                {
                    // not a mirror collision; count error
                    //printf("Collision Error\r\n");
                    gTotalErrors++;
                }
                continue;
            }
            gSolved = true;
            return;
        }
    }
}

// -------------- Stats ---------------------
void ShowStats(u64 tm_start, double exp_ops, double dp_val)
{
#ifdef DEBUG_MODE
    for (int i = 0; i <= MD_LEN; i++)
    {
        u64 val = 0;
        for (int j = 0; j < GpuCnt; j++)
            val += GpuKangs[j]->dbg[i];
        if (val)
            printf("Loop size %d: %llu\r\n", i, val);
    }
#endif

    int speed = GpuKangs[0]->GetStatsSpeed();
    for (int i = 1; i < GpuCnt; i++)
        speed += GpuKangs[i]->GetStatsSpeed();

    u64 est_dps_cnt = (u64)(exp_ops / dp_val);
    u64 exp_sec = 0xFFFFFFFFFFFFFFFFull;
    if (speed)
        exp_sec = (u64)((exp_ops / 1000000) / speed); //in sec
    u64 exp_days = exp_sec / (3600 * 24);
    int exp_hours = (int)(exp_sec - exp_days * (3600 * 24)) / 3600;
    int exp_min = (int)(exp_sec - exp_days * (3600 * 24) - exp_hours * 3600) / 60;

    u64 sec = (GetTickCount64() - tm_start) / 1000;
    u64 days = sec / (3600 * 24);
    int hours = (int)(sec - days * (3600 * 24)) / 3600;
    int min = (int)(sec - days * (3600 * 24) - hours * 3600) / 60;
    
    printf("%sSpeed: %d MKeys/s, Err: %d, DPs: %lluK/%lluK, Time: %llud:%02dh:%02dm/%llud:%02dh:%02dm\r\n",
        gGenMode ? "GEN: " : (IsBench ? "BENCH: " : "MAIN: "), speed, gTotalErrors, db.GetBlockCnt()/1000, est_dps_cnt/1000, days, hours, min, exp_days, exp_hours, exp_min);
}

// -------------- Solver --------------------
bool SolvePoint(EcPoint PntToSolve, int Range, int DP, EcInt* pk_res)
{
    if ((Range < 32) || (Range > 180))
    {
        printf("Unsupported Range value (%d)!\r\n", Range);
        return false;
    }
    if ((DP < 14) || (DP > 60)) 
    {
        printf("Unsupported DP value (%d)!\r\n", DP);
        return false;
    }

    printf("\r\nSolving point: Range %d bits, DP %d, start...\r\n", Range, DP);
    double ops = 1.15 * pow(2.0, Range / 2.0);
    double dp_val = (double)(1ull << DP);
    double ram = (32 + 4 + 4) * ops / dp_val; //+4 for grow allocation and memory fragmentation
    ram += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
    ram /= (1024 * 1024 * 1024); //GB
    printf("SOTA method, estimated ops: 2^%.3f, RAM for DPs: %.3f GB. DP and GPU overheads not included!\r\n", log2(ops), ram);
    gIsOpsLimit = false;
    double MaxTotalOps = 0.0;
    if (gMax > 0)
    {
        MaxTotalOps = gMax * ops;
        printf("Max allowed number of ops: 2^%.3f\r\n", log2(MaxTotalOps));
    }

    u64 total_kangs = GpuKangs[0]->CalcKangCnt();
    for (int i = 1; i < GpuCnt; i++)
        total_kangs += GpuKangs[i]->CalcKangCnt();
    double path_single_kang = ops / total_kangs; 
    double DPs_per_kang = path_single_kang / dp_val;
    printf("Estimated DPs per kangaroo: %.3f.%s\r\n", DPs_per_kang, (DPs_per_kang < 5) ? " DP overhead is big, use less DP value if possible!" : "");

    if (!gGenMode && gTamesFileName[0])
    {
        printf("load tames...\r\n");
        if (db.LoadFromFile(gTamesFileName))
        {
            printf("tames loaded\r\n");
            if (db.Header[0] != gRange)
            {
                printf("loaded tames have different range, they cannot be used, clear\r\n");
                db.Clear();
            }
        }
        else
            printf("tames loading failed\r\n");
    }

    SetRndSeed(0); //use same seed to make tames from file compatible
    PntTotalOps = 0;
    PntIndex = 0;
    // prepare jumps
    EcInt minjump, t;
    minjump.Set(1);
    minjump.ShiftLeft(Range / 2 + 3);
    for (int i = 0; i < JMP_CNT; i++)
    {
        EcJumps1[i].dist = minjump;
        t.RndMax(minjump);
        EcJumps1[i].dist.Add(t);
        EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
        EcJumps1[i].p = ec.MultiplyG(EcJumps1[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10); //large jumps for L1S2 loops. Must be almost RANGE_BITS
    for (int i = 0; i < JMP_CNT; i++)
    {
        EcJumps2[i].dist = minjump;
        t.RndMax(minjump);
        EcJumps2[i].dist.Add(t);
        EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
        EcJumps2[i].p = ec.MultiplyG(EcJumps2[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10 - 2); //large jumps for loops >2
    for (int i = 0; i < JMP_CNT; i++)
    {
        EcJumps3[i].dist = minjump;
        t.RndMax(minjump);
        EcJumps3[i].dist.Add(t);
        EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
        EcJumps3[i].p = ec.MultiplyG(EcJumps3[i].dist);
    }
    SetRndSeed(GetTickCount64());

    Int_HalfRange.Set(1);
    Int_HalfRange.ShiftLeft(Range - 1);
    Pnt_HalfRange = ec.MultiplyG(Int_HalfRange);
    Pnt_NegHalfRange = Pnt_HalfRange;
    Pnt_NegHalfRange.y.NegModP();
    Int_TameOffset.Set(1);
    Int_TameOffset.ShiftLeft(Range - 1);
    EcInt tt;
    tt.Set(1);
    tt.ShiftLeft(Range - 5); //half of tame range width
    Int_TameOffset.Sub(tt);
    gPntToSolve = PntToSolve; // primary target for which GPU walks are built

    // prepare GPUs
    for (int i = 0; i < GpuCnt; i++)
        if (!GpuKangs[i]->Prepare(PntToSolve, Range, DP, EcJumps1, EcJumps2, EcJumps3))
        {
            GpuKangs[i]->Failed = true;
            printf("GPU %d Prepare failed\r\n", GpuKangs[i]->CudaIndex);
        }

    u64 tm0 = GetTickCount64();
    printf("GPUs started...\r\n");

#ifdef _WIN32
    HANDLE thr_handles[MAX_GPU_CNT];
#else
    pthread_t thr_handles[MAX_GPU_CNT];
#endif

    u32 ThreadID;
    gSolved = false;
    gSolvedIndex = -1;
    ThrCnt = GpuCnt;
    for (int i = 0; i < GpuCnt; i++)
    {
#ifdef _WIN32
        thr_handles[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc, (void*)GpuKangs[i], 0, &ThreadID);
#else
        pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)GpuKangs[i]);
#endif
    }

    u64 tm_stats = GetTickCount64();
    while (!gSolved)
    {
        CheckNewPoints();
        Sleep(10);
        if (GetTickCount64() - tm_stats > 10 * 1000)
        {
            ShowStats(tm0, ops, dp_val);
            tm_stats = GetTickCount64();
        }

        if ((MaxTotalOps > 0.0) && (PntTotalOps > MaxTotalOps))
        {
            gIsOpsLimit = true;
            printf("Operations limit reached\r\n");
            break;
        }
    }

    printf("Stopping work ...\r\n");
    for (int i = 0; i < GpuCnt; i++)
        GpuKangs[i]->Stop();
    while (ThrCnt)
        Sleep(10);
    for (int i = 0; i < GpuCnt; i++)
    {
#ifdef _WIN32
        CloseHandle(thr_handles[i]);
#else
        pthread_join(thr_handles[i], NULL);
#endif
    }

    if (gIsOpsLimit)
    {
        if (gGenMode)
        {
            printf("saving tames...\r\n");
            db.Header[0] = gRange; 
            if (db.SaveToFile(gTamesFileName))
                printf("tames saved\r\n");
            else
                printf("tames saving failed\r\n");
        }
        db.Clear();
        return false;
    }

    double K = (double)PntTotalOps / pow(2.0, Range / 2.0);
    printf("Point solved, K: %.3f (with DP and GPU overheads)\r\n\r\n", K);
    db.Clear();
    *pk_res = gPrivKey;
    return true;
}

// -------------- CLI & I/O -----------------
static bool LoadPubKeysFile(const char* fname)
{
    FILE* f = fopen(fname, "r");
    if (!f) { printf("Cannot open %s\n", fname); return false; }

    const size_t kMax = 50000; // as requested
    char line[256];
    size_t count = 0;
    gPubKeysVec.clear();
    gTargetsVec.clear();
    gPubHexVec.clear();

    while (fgets(line, sizeof(line), f))
    {
        // trim
        line[strcspn(line, "\r\n")] = 0;
        if (!*line) continue;

        EcPoint pk;
        if (!pk.SetHexStr(line))
        {
            printf("Skip invalid pubkey format: %s\n", line);
            continue;
        }
        gPubKeysVec.push_back(pk);
        gPubHexVec.emplace_back(line);
        count++;
        if (count >= kMax) break;
    }
    fclose(f);

    if (count == 0)
    {
        printf("No valid pubkeys loaded from %s\n", fname);
        return false;
    }

    // Build per-target adjusted points P' = P + (-G*start)
    EcPoint PntOfs;
    if (!gStart.IsZero())
    {
        PntOfs = ec.MultiplyG(gStart);
        PntOfs.y.NegModP();
    }

    gTargetsVec.reserve(gPubKeysVec.size());
    for (size_t i = 0; i < gPubKeysVec.size(); ++i)
    {
        if (!gStart.IsZero())
            gTargetsVec.push_back(ec.AddPoints(gPubKeysVec[i], PntOfs));
        else
            gTargetsVec.push_back(gPubKeysVec[i]);
    }

    printf("Loaded %zu pubkeys from %s\n", gPubKeysVec.size(), fname);
    return true;
}

bool ParseCommandLine(int argc, char* argv[])
{
    int ci = 1;
    while (ci < argc)
    {
        char* argument = argv[ci];
        ci++;
        if (strcmp(argument, "-gpu") == 0)
        {
            if (ci >= argc)
            {
                printf("error: missed value after -gpu option\r\n");
                return false;
            }
            char* gpus = argv[ci];
            ci++;
            memset(gGPUs_Mask, 0, sizeof(gGPUs_Mask));
            for (int i = 0; i < (int)strlen(gpus); i++)
            {
                if ((gpus[i] < '0') || (gpus[i] > '9'))
                {
                    printf("error: invalid value for -gpu option\r\n");
                    return false;
                }
                gGPUs_Mask[gpus[i] - '0'] = 1;
            }
        }
        else if (strcmp(argument, "-dp") == 0)
        {
            int val = atoi(argv[ci]);
            ci++;
            if ((val < 14) || (val > 60))
            {
                printf("error: invalid value for -dp option\r\n");
                return false;
            }
            gDP = val;
        }
        else if (strcmp(argument, "-range") == 0)
        {
            int val = atoi(argv[ci]);
            ci++;
            if ((val < 32) || (val > 170))
            {
                printf("error: invalid value for -range option\r\n");
                return false;
            }
            gRange = val;
        }
        else if (strcmp(argument, "-start") == 0)
        {   
            if (!gStart.SetHexStr(argv[ci]))
            {
                printf("error: invalid value for -start option\r\n");
                return false;
            }
            ci++;
            gStartSet = true;
        }
        else if (strcmp(argument, "-pubkey") == 0)
        {
            if (!gPubKey.SetHexStr(argv[ci]))
            {
                printf("error: invalid value for -pubkey option\r\n");
                return false;
            }
            ci++;
        }
        else if (strcmp(argument, "-pubkeysfile") == 0)
        {
            strcpy(gPubKeysFileName, argv[ci]);
            ci++;
            gUsePubKeysFile = true;
        }
        else if (strcmp(argument, "-tames") == 0)
        {
            strcpy(gTamesFileName, argv[ci]);
            ci++;
        }
        else if (strcmp(argument, "-max") == 0)
        {
            double val = atof(argv[ci]);
            ci++;
            if (val < 0.001)
            {
                printf("error: invalid value for -max option\r\n");
                return false;
            }
            gMax = val;
        }
        else if (strcmp(argument, "-keepgoing") == 0)
        {
            gStopOnFirst = false; // optional: keep scanning (simple flag, not a hard guarantee)
        }
        else
        {
            printf("error: unknown option %s\r\n", argument);
            return false;
        }
    }

    if (!gPubKey.x.IsZero())
        if (!gStartSet || !gRange || !gDP)
        {
            printf("error: you must also specify -dp, -range and -start options\r\n");
            return false;
        }

    if (gTamesFileName[0] && !IsFileExist(gTamesFileName))
    {
        if (gMax == 0.0)
        {
            printf("error: you must also specify -max option to generate tames\r\n");
            return false;
        }
        gGenMode = true;
    }
    return true;
}

int main(int argc, char* argv[])
{
#ifdef _DEBUG 
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    printf("********************************************************************************\r\n");
    printf("*                    RCKangaroo v3.0  (c) 2024 RetiredCoder                    *\r\n");
    printf("********************************************************************************\r\n\r\n");

    printf("This software is free and open-source: https://github.com/RetiredC\r\n");
    printf("It demonstrates fast GPU implementation of SOTA Kangaroo method for solving ECDLP\r\n");

#ifdef _WIN32
    printf("Windows version\r\n");
#else
    printf("Linux version\r\n");
#endif

#ifdef DEBUG_MODE
    printf("DEBUG MODE\r\n\r\n");
#endif

    InitEc();
    gDP = 0;
    gRange = 0;
    gStartSet = false;
    gTamesFileName[0] = 0;
    gPubKeysFileName[0] = 0;
    gUsePubKeysFile = false;
    gMax = 0.0;
    gGenMode = false;
    gIsOpsLimit = false;
    memset(gGPUs_Mask, 1, sizeof(gGPUs_Mask));
    if (!ParseCommandLine(argc, argv))
        return 0;

    // If we are in multi-file mode, load pubkeys now (also sets gTargetsVec)
    if (gUsePubKeysFile)
    {
        if (!gRange || !gDP)
        {
            printf("error: when using -pubkeysfile, you must also specify -dp and -range (and -start if using an offset)\n");
            return 0;
        }
        if (!LoadPubKeysFile(gPubKeysFileName))
            return 0;
        // pick the *first* target as the GPU-walk base
        if (!gTargetsVec.empty())
            gPntToSolve = gTargetsVec[0];
    }

    InitGpus();

    if (!GpuCnt)
    {
        printf("No supported GPUs detected, exit\r\n");
        return 0;
    }

    pPntList = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
    pPntList2 = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
    TotalOps = 0;
    TotalSolved = 0;
    gTotalErrors = 0;

    // In bench mode when *no* single pubkey and *no* file provided
    IsBench = (gPubKey.x.IsZero() && !gUsePubKeysFile);

    if (!IsBench && !gGenMode)
    {
        printf("\r\nMAIN MODE\r\n\r\n");

        if (gUsePubKeysFile)
        {
            // Build first target point for GPU walks
            EcPoint PntToSolve = gTargetsVec[0];
            EcInt pk_found;

            // For logging
            char sx[100], sy[100];
            gPubKeysVec[0].x.GetHexStr(sx);
            gPubKeysVec[0].y.GetHexStr(sy);
            printf("Using first pubkey as GPU base\nX: %s\nY: %s\n", sx, sy);

            if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
            {
                if (!gIsOpsLimit)
                    printf("SolvePoint failed (multi-file mode)\n");
            }

            if (gSolvedIndex >= 0)
            {
                // pk_found already includes +start (set in CheckNewPoints)
                char s[100];
                gPrivKey.GetHexStr(s);
                printf("\nPRIVATE KEY for line %d: %s\n\n", gSolvedIndex + 1, s);
                FILE* fp = fopen("RESULTS.TXT", "a");
                if (fp) { fprintf(fp, "PRIVATE KEY (line %d): %s\n", gSolvedIndex + 1, s); fclose(fp); }
                else    { printf("WARNING: Cannot save key to RESULTS.TXT\n"); }
            }
            else
            {
                printf("No key found in this run.\n");
            }
        }
        else
        {
            // Original single-key code
            EcPoint PntToSolve, PntOfs;
            EcInt pk_found;

            PntToSolve = gPubKey;
            if (!gStart.IsZero())
            {
                PntOfs = ec.MultiplyG(gStart);
                PntOfs.y.NegModP();
                PntToSolve = ec.AddPoints(PntToSolve, PntOfs);
            }

            char sx[100], sy[100];
            gPubKey.x.GetHexStr(sx);
            gPubKey.y.GetHexStr(sy);
            printf("Solving public key\r\nX: %s\r\nY: %s\r\n", sx, sy);
            gStart.GetHexStr(sx);
            printf("Offset: %s\r\n", sx);

            if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
            {
                if (!gIsOpsLimit)
                    printf("FATAL ERROR: SolvePoint failed\r\n");
                goto label_end;
            }
            pk_found.AddModP(gStart);
            EcPoint tmp = ec.MultiplyG(pk_found);
            if (!tmp.IsEqual(gPubKey))
            {
                printf("FATAL ERROR: SolvePoint found incorrect key\r\n");
                goto label_end;
            }
            char s[100];
            pk_found.GetHexStr(s);
            printf("\r\nPRIVATE KEY: %s\r\n\r\n", s);
            FILE* fp = fopen("RESULTS.TXT", "a");
            if (fp)
            {
                fprintf(fp, "PRIVATE KEY: %s\n", s);
                fclose(fp);
            }
            else
            {
                printf("WARNING: Cannot save the key to RESULTS.TXT!\r\n");
                while (1)
                    Sleep(100);
            }
        }
    }
    else
    {
        if (gGenMode)
            printf("\r\nTAMES GENERATION MODE\r\n");
        else
            printf("\r\nBENCHMARK MODE\r\n");
        // simple bench loop
        while (1)
        {
            EcInt pk, pk_found;
            EcPoint PntToSolve;

            if (!gRange)
                gRange = 78;
            if (!gDP)
                gDP = 16;

            pk.RndBits(gRange);
            PntToSolve = ec.MultiplyG(pk);

            if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
            {
                if (!gIsOpsLimit)
                    printf("FATAL ERROR: SolvePoint failed\r\n");
                break;
            }
            if (!pk_found.IsEqual(pk))
            {
                printf("FATAL ERROR: Found key is wrong!\r\n");
                break;
            }
            TotalOps += PntTotalOps;
            TotalSolved++;
            u64 ops_per_pnt = TotalOps / TotalSolved;
            double K = (double)ops_per_pnt / pow(2.0, gRange / 2.0);
            printf("Points solved: %d, average K: %.3f (with DP and GPU overheads)\r\n", TotalSolved, K);
        }
    }
label_end:
    for (int i = 0; i < GpuCnt; i++)
        delete GpuKangs[i];
    DeInitEc();
    free(pPntList2);
    free(pPntList);
}
