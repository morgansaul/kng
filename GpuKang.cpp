// RCKangaroo.cpp  — complete file
// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC

#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "cuda_runtime.h"
#include "cuda.h"

#include "defs.h"
#include "utils.h"
#include "GpuKang.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
// Globals (matching original project layout)

EcJMP EcJumps1[JMP_CNT];
EcJMP EcJumps2[JMP_CNT];
EcJMP EcJumps3[JMP_CNT];

RCGpuKang* GpuKangs[MAX_GPU_CNT];
int GpuCnt = 0;
volatile long ThrCnt = 0;
volatile bool gSolved = false;

EcInt   Int_HalfRange;
EcPoint Pnt_HalfRange;
EcPoint Pnt_NegHalfRange;
EcInt   Int_TameOffset;
Ec      ec;

CriticalSection csAddPoints;
u8* pPntList  = nullptr;
u8* pPntList2 = nullptr;
volatile int PntIndex = 0;
TFastBase db;

EcInt  gPrivKey;          // found private key (relative; add gStart at the end)
u32    gDP = 0;
u32    gRange = 0;
EcInt  gStart;            // start offset
bool   gStartSet = false;
EcPoint gPubKey;          // single pubkey mode

u8   gGPUs_Mask[MAX_GPU_CNT];
char gTamesFileName[1024];
double gMax = 0.0;
bool gGenMode = false;    // tames generation mode
bool gIsOpsLimit = false;

volatile u64 TotalOps = 0;
u32 TotalSolved = 0;
u32 gTotalErrors = 0;
u64 PntTotalOps = 0;
bool IsBench = false;

// NEW: multi-target file support
static char gPubkeysFile[1024] = {0};

// Targets (original and shifted-by-start for solving)
static std::vector<EcPoint> gTargetsOriginal;  // as read from file
static std::vector<EcPoint> gTargetsShifted;   // P - G*start

///////////////////////////////////////////////////////////////////////////////////////////////////
// DB record layout for distinguished points
#pragma pack(push, 1)
struct DBRec
{
    u8 x[12];
    u8 d[22];
    u8 type; //0 - tame, 1 - wild1, 2 - wild2
};
#pragma pack(pop)

///////////////////////////////////////////////////////////////////////////////////////////////////
// GPU init

void InitGpus()
{
    GpuCnt = 0;
    int gcnt = 0;
    cudaGetDeviceCount(&gcnt);
    if (gcnt > MAX_GPU_CNT) gcnt = MAX_GPU_CNT;
    if (!gcnt) return;

    int drv, rt;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    char drvver[100];
    sprintf(drvver, "%d.%d/%d.%d", drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);

    printf("CUDA devices: %d, CUDA driver/runtime: %s\r\n", gcnt, drvver);

    for (int i = 0; i < gcnt; i++)
    {
        if (!gGPUs_Mask[i]) continue;

        cudaError_t cudaStatus = cudaSetDevice(i);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaSetDevice for gpu %d failed!\r\n", i);
            continue;
        }

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("GPU %d: %s, %.2f GB, %d CUs, cap %d.%d, PCI %d, L2 size: %d KB\r\n",
               i, deviceProp.name,
               ((float)(deviceProp.totalGlobalMem / (1024 * 1024))) / 1024.0f,
               deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor,
               deviceProp.pciBusID, deviceProp.l2CacheSize / 1024);

        if (deviceProp.major < 6)
        {
            printf("GPU %d - not supported, skip\r\n", i);
            continue;
        }

        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

        RCGpuKang* g = new RCGpuKang();
        g->CudaIndex = i;
        g->persistingL2CacheMaxSize = deviceProp.persistingL2CacheMaxSize;
        g->mpCnt    = deviceProp.multiProcessorCount;
        g->IsOldGpu = deviceProp.l2CacheSize < 16 * 1024 * 1024;
        GpuKangs[GpuCnt++] = g;
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

///////////////////////////////////////////////////////////////////////////////////////////////////
// Points IO between GPU and host

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

///////////////////////////////////////////////////////////////////////////////////////////////////
// Collision resolution (unchanged SOTA core; we’ll call it per-target)

bool Collision_SOTA(EcPoint& targetP, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg)
{
    if (IsNeg) t.Neg();
    if (TameType == TAME)
    {
        gPrivKey = t;
        gPrivKey.Sub(w);
        EcInt sv = gPrivKey;
        gPrivKey.Add(Int_HalfRange);
        EcPoint P = ec.MultiplyG(gPrivKey);
        if (P.IsEqual(targetP)) return true;
        gPrivKey = sv;
        gPrivKey.Neg();
        gPrivKey.Add(Int_HalfRange);
        P = ec.MultiplyG(gPrivKey);
        return P.IsEqual(targetP);
    }
    else
    {
        gPrivKey = t;
        gPrivKey.Sub(w);
        if (gPrivKey.data[4] >> 63) gPrivKey.Neg();
        gPrivKey.ShiftRight(1);
        EcInt sv = gPrivKey;
        gPrivKey.Add(Int_HalfRange);
        EcPoint P = ec.MultiplyG(gPrivKey);
        if (P.IsEqual(targetP)) return true;
        gPrivKey = sv;
        gPrivKey.Neg();
        gPrivKey.Add(Int_HalfRange);
        P = ec.MultiplyG(gPrivKey);
        return P.IsEqual(targetP);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-target collision checking at every batch

static EcPoint gMatchedTargetOriginal; // the original pubkey solved

void CheckNewPoints_Multi()
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
        if (gGenMode) continue; // generation mode skips solving

        if (pref)
        {
            // restore first 3 bytes not stored in DB
            DBRec tmp_pref;
            memcpy(&tmp_pref, &nrec, 3);
            memcpy(((u8*)&tmp_pref) + 3, pref, sizeof(DBRec) - 3);
            pref = &tmp_pref;

            if (pref->type == nrec.type)
            {
                if (pref->type == TAME) continue;
                if (*(u64*)pref->d == *(u64*)nrec.d) continue; // same wild; ignore
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

            // NEW: test against ALL shifted targets
            bool solved_here = false;
            for (size_t k = 0; k < gTargetsShifted.size(); ++k)
            {
                EcPoint &targetShifted = gTargetsShifted[k];
                if (Collision_SOTA(targetShifted, t, TameType, w, WildType, false) ||
                    Collision_SOTA(targetShifted, t, TameType, w, WildType, true))
                {
                    // found relative key => add start and verify vs original target
                    EcInt fullKey = gPrivKey;
                    fullKey.AddModP(gStart);

                    EcPoint check = ec.MultiplyG(fullKey);
                    if (check.IsEqual(gTargetsOriginal[k]))
                    {
                        // success
                        gPrivKey = fullKey;
                        gMatchedTargetOriginal = gTargetsOriginal[k];
                        gSolved = true;
                        solved_here = true;
                        break;
                    }
                    else
                    {
                        // shouldn’t happen—defensive
                        printf("Collision verification failed for a target\r\n");
                        gTotalErrors++;
                    }
                }
                if (solved_here) break;
            }

            if (!solved_here)
            {
                bool w12 = ((pref->type == WILD1) && (nrec.type == WILD2)) ||
                           ((pref->type == WILD2) && (nrec.type == WILD1));
                if (!w12)
                {
                    printf("Collision Error\r\n");
                    gTotalErrors++;
                }
            }

            if (gSolved) break;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Stats

void ShowStats(u64 tm_start, double exp_ops, double dp_val)
{
#ifdef DEBUG_MODE
    for (int i = 0; i <= MD_LEN; i++)
    {
        u64 val = 0;
        for (int j = 0; j < GpuCnt; j++) val += GpuKangs[j]->dbg[i];
        if (val) printf("Loop size %d: %llu\r\n", i, val);
    }
#endif

    int speed = GpuKangs[0]->GetStatsSpeed();
    for (int i = 1; i < GpuCnt; i++) speed += GpuKangs[i]->GetStatsSpeed();

    u64 est_dps_cnt = (u64)(exp_ops / dp_val);
    u64 exp_sec = 0xFFFFFFFFFFFFFFFFull;
    if (speed) exp_sec = (u64)((exp_ops / 1000000) / speed); // seconds
    u64 exp_days = exp_sec / (3600 * 24);
    int exp_hours = (int)(exp_sec - exp_days * (3600 * 24)) / 3600;
    int exp_min = (int)(exp_sec - exp_days * (3600 * 24) - exp_hours * 3600) / 60;

    u64 sec = (GetTickCount64() - tm_start) / 1000;
    u64 days = sec / (3600 * 24);
    int hours = (int)(sec - days * (3600 * 24)) / 3600;
    int min = (int)(sec - days * (3600 * 24) - hours * 3600) / 60;

    printf("%sSpeed: %d MKeys/s, Err: %d, DPs: %lluK/%lluK, Time: %llud:%02dh:%02dm/%llud:%02dh:%02dm\r\n",
           gGenMode ? "GEN: " : (IsBench ? "BENCH: " : "MAIN: "),
           speed, gTotalErrors,
           db.GetBlockCnt()/1000, est_dps_cnt/1000,
           days, hours, min, exp_days, exp_hours, exp_min);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-target solving (single run, checks all targets every time)

bool SolveMultiTargets(int Range, int DP)
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
    if (gTargetsShifted.empty())
    {
        printf("No targets loaded.\r\n");
        return false;
    }

    printf("\r\nSolving %zu targets simultaneously: Range %d bits, DP %d, start...\r\n",
           gTargetsShifted.size(), Range, DP);

    // complexity estimate based on single-target SOTA; this is just for stats
    double ops = 1.15 * pow(2.0, Range / 2.0);
    double dp_val = (double)(1ull << DP);
    double ram = (32 + 4 + 4) * ops / dp_val;
    ram += sizeof(TListRec) * 256.0 * 256.0 * 256.0;
    ram /= (1024.0 * 1024.0 * 1024.0);
    printf("SOTA method, estimated ops (single-target): 2^%.3f, RAM for DPs: %.3f GB. DP and GPU overheads not included!\r\n",
           log2(ops), ram);

    gIsOpsLimit = false;
    double MaxTotalOps = 0.0;
    if (gMax > 0)
    {
        MaxTotalOps = gMax * ops;
        double ram_max = (32 + 4 + 4) * MaxTotalOps / dp_val;
        ram_max += sizeof(TListRec) * 256.0 * 256.0 * 256.0;
        ram_max /= (1024.0 * 1024.0 * 1024.0);
        printf("Max allowed number of ops: 2^%.3f, max RAM for DPs: %.3f GB\r\n",
               log2(MaxTotalOps), ram_max);
    }

    // Estimate DPs per kangaroo
    u64 total_kangs = GpuKangs[0]->CalcKangCnt();
    for (int i = 1; i < GpuCnt; i++) total_kangs += GpuKangs[i]->CalcKangCnt();
    double path_single_kang = ops / total_kangs;
    double DPs_per_kang = path_single_kang / dp_val;
    printf("Estimated DPs per kangaroo: %.3f.%s\r\n",
           DPs_per_kang, (DPs_per_kang < 5.0) ? " DP overhead is big, use less DP value if possible!" : "");

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

    SetRndSeed(0);
    PntTotalOps = 0;
    PntIndex = 0;

    // Prepare jumps
    EcInt minjump, t;
    minjump.Set(1);
    minjump.ShiftLeft(Range / 2 + 3);
    for (int i = 0; i < JMP_CNT; i++)
    {
        EcJumps1[i].dist = minjump;
        t.RndMax(minjump);
        EcJumps1[i].dist.Add(t);
        EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; // even
        EcJumps1[i].p = ec.MultiplyG(EcJumps1[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10);
    for (int i = 0; i < JMP_CNT; i++)
    {
        EcJumps2[i].dist = minjump;
        t.RndMax(minjump);
        EcJumps2[i].dist.Add(t);
        EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        EcJumps2[i].p = ec.MultiplyG(EcJumps2[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 12); // Range - 10 - 2
    for (int i = 0; i < JMP_CNT; i++)
    {
        EcJumps3[i].dist = minjump;
        t.RndMax(minjump);
        EcJumps3[i].dist.Add(t);
        EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
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
    tt.ShiftLeft(Range - 5);
    Int_TameOffset.Sub(tt);

    // Use the FIRST shifted target to set GPU walk reference (wild offsets).
    // We still test collisions against ALL targets in CheckNewPoints_Multi().
    EcPoint refTarget = gTargetsShifted[0];

    // Prepare GPUs
    for (int i = 0; i < GpuCnt; i++)
    {
        if (!GpuKangs[i]->Prepare(refTarget, Range, DP, EcJumps1, EcJumps2, EcJumps3))
        {
            GpuKangs[i]->Failed = true;
            printf("GPU %d Prepare failed\r\n", GpuKangs[i]->CudaIndex);
        }
    }

    u64 tm0 = GetTickCount64();
    printf("GPUs started...\r\n");

#ifdef _WIN32
    HANDLE thr_handles[MAX_GPU_CNT];
#else
    pthread_t thr_handles[MAX_GPU_CNT];
#endif

    gSolved = false;
    ThrCnt = GpuCnt;
    for (int i = 0; i < GpuCnt; i++)
    {
#ifdef _WIN32
        u32 ThreadID;
        thr_handles[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc, (void*)GpuKangs[i], 0, &ThreadID);
#else
        pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)GpuKangs[i]);
#endif
    }

    u64 tm_stats = GetTickCount64();
    while (!gSolved)
    {
        CheckNewPoints_Multi();
        Sleep(10);
        if (GetTickCount64() - tm_stats > 10 * 1000)
        {
            ShowStats(tm0, ops, dp_val);
            tm_stats = GetTickCount64();
        }
        if ((gMax > 0.0) && (PntTotalOps > (u64)(gMax * ops)))
        {
            gIsOpsLimit = true;
            printf("Operations limit reached\r\n");
            break;
        }
    }

    printf("Stopping work ...\r\n");
    for (int i = 0; i < GpuCnt; i++) GpuKangs[i]->Stop();
    while (ThrCnt) Sleep(10);
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
            if (db.SaveToFile(gTamesFileName)) printf("tames saved\r\n");
            else printf("tames saving failed\r\n");
        }
        db.Clear();
        return false;
    }

    double K = (double)PntTotalOps / pow(2.0, gRange / 2.0);
    printf("Point solved, K: %.3f (with DP and GPU overheads)\r\n\r\n", K);
    db.Clear();

    // Output the solved key
    if (gSolved)
    {
        char s[100];
        gPrivKey.GetHexStr(s);
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
            while (1) Sleep(100);
        }
    }
    return gSolved;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// CLI

static bool LoadTargetsFromFile(const char* filePath)
{
    gTargetsOriginal.clear();
    gTargetsShifted.clear();

    FILE* fp = fopen(filePath, "r");
    if (!fp)
    {
        printf("error: cannot open pubkeys file: %s\r\n", filePath);
        return false;
    }

    char line[4096];
    int lineNo = 0;
    while (fgets(line, sizeof(line), fp))
    {
        // strip whitespace
        size_t n = strlen(line);
        while (n && (line[n-1] == '\n' || line[n-1] == '\r' || line[n-1] == ' ' || line[n-1] == '\t')) { line[--n] = 0; }
        char* s = line;
        while (*s == ' ' || *s == '\t') ++s;
        if (!*s) continue; // skip empty

        EcPoint P;
        if (!P.SetHexStr(s))
        {
            printf("warning: invalid pubkey at line %d: %s\r\n", lineNo+1, s);
        }
        else
        {
            gTargetsOriginal.push_back(P);
        }
        lineNo++;
    }
    fclose(fp);

    if (gTargetsOriginal.empty())
    {
        printf("error: no valid public keys in file.\r\n");
        return false;
    }

    // shift by start (P - G*start)
    for (size_t i = 0; i < gTargetsOriginal.size(); ++i)
    {
        EcPoint PntToSolve = gTargetsOriginal[i];
        if (!gStart.IsZero())
        {
            EcPoint PntOfs = ec.MultiplyG(gStart);
            PntOfs.y.NegModP();
            PntToSolve = ec.AddPoints(PntToSolve, PntOfs);
        }
        gTargetsShifted.push_back(PntToSolve);
    }

    printf("Loaded %zu public keys from file.\r\n", gTargetsOriginal.size());
    return true;
}

bool ParseCommandLine(int argc, char* argv[])
{
    int ci = 1;
    while (ci < argc)
    {
        char* argument = argv[ci++];
        if (strcmp(argument, "-gpu") == 0)
        {
            if (ci >= argc) { printf("error: missed value after -gpu option\r\n"); return false; }
            char* gpus = argv[ci++];
            memset(gGPUs_Mask, 0, sizeof(gGPUs_Mask));
            for (int i = 0; i < (int)strlen(gpus); i++)
            {
                if (gpus[i] < '0' || gpus[i] > '9')
                {
                    printf("error: invalid value for -gpu option\r\n");
                    return false;
                }
                gGPUs_Mask[gpus[i] - '0'] = 1;
            }
        }
        else if (strcmp(argument, "-dp") == 0)
        {
            if (ci >= argc) { printf("error: missed value after -dp\r\n"); return false; }
            int val = atoi(argv[ci++]);
            if (val < 14 || val > 60) { printf("error: invalid value for -dp option\r\n"); return false; }
            gDP = val;
        }
        else if (strcmp(argument, "-range") == 0)
        {
            if (ci >= argc) { printf("error: missed value after -range\r\n"); return false; }
            int val = atoi(argv[ci++]);
            if (val < 32 || val > 170) { printf("error: invalid value for -range option\r\n"); return false; }
            gRange = val;
        }
        else if (strcmp(argument, "-start") == 0)
        {
            if (ci >= argc) { printf("error: missed value after -start\r\n"); return false; }
            if (!gStart.SetHexStr(argv[ci++]))
            {
                printf("error: invalid value for -start option\r\n");
                return false;
            }
            gStartSet = true;
        }
        else if (strcmp(argument, "-pubkey") == 0)
        {
            if (ci >= argc) { printf("error: missed value after -pubkey\r\n"); return false; }
            if (!gPubKey.SetHexStr(argv[ci++]))
            {
                printf("error: invalid value for -pubkey option\r\n");
                return false;
            }
        }
        else if (strcmp(argument, "-pubkeysfile") == 0)
        {
            if (ci >= argc) { printf("error: missed value after -pubkeysfile\r\n"); return false; }
            strncpy(gPubkeysFile, argv[ci++], sizeof(gPubkeysFile)-1);
            gPubkeysFile[sizeof(gPubkeysFile)-1] = 0;
        }
        else if (strcmp(argument, "-tames") == 0)
        {
            if (ci >= argc) { printf("error: missed value after -tames\r\n"); return false; }
            strcpy(gTamesFileName, argv[ci++]);
        }
        else if (strcmp(argument, "-max") == 0)
        {
            if (ci >= argc) { printf("error: missed value after -max\r\n"); return false; }
            double val = atof(argv[ci++]);
            if (val < 0.001) { printf("error: invalid value for -max option\r\n"); return false; }
            gMax = val;
        }
        else
        {
            printf("error: unknown option %s\r\n", argument);
            return false;
        }
    }

    // Basic validations
    if (!gPubkeysFile[0] && !gPubKey.x.IsZero())
    {
        if (!gStartSet || !gRange || !gDP)
        {
            printf("error: you must also specify -dp, -range and -start options\r\n");
            return false;
        }
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

///////////////////////////////////////////////////////////////////////////////////////////////////
// main

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
    gMax = 0.0;
    gGenMode = false;
    gIsOpsLimit = false;
    memset(gGPUs_Mask, 1, sizeof(gGPUs_Mask));

    if (!ParseCommandLine(argc, argv)) return 0;

    InitGpus();

    if (!GpuCnt)
    {
        printf("No supported GPUs detected, exit\r\n");
        return 0;
    }

    pPntList  = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
    pPntList2 = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
    TotalOps = 0;
    TotalSolved = 0;
    gTotalErrors = 0;

    const bool multiTargets = (gPubkeysFile[0] != 0);
    IsBench = (!multiTargets && gPubKey.x.IsZero());

    if (multiTargets)
    {
        if (!gRange || !gDP)
        {
            printf("error: you must specify -range and -dp for -pubkeysfile mode\r\n");
            goto label_end;
        }
        if (!LoadTargetsFromFile(gPubkeysFile)) goto label_end;

        printf("\r\nMAIN MULTI-TARGET MODE (simultaneous check at every jump)\r\n\r\n");

        if (!SolveMultiTargets(gRange, gDP))
        {
            if (!gIsOpsLimit)
                printf("No solution found before stop/limit.\r\n");
        }
    }
    else if (!IsBench && !gGenMode)
    {
        printf("\r\nMAIN MODE (single key)\r\n\r\n");
        // Keep single-key path for completeness; reuse existing logic by calling multi with one target
        gTargetsOriginal.clear();
        gTargetsShifted.clear();
        gTargetsOriginal.push_back(gPubKey);

        EcPoint PntToSolve = gPubKey;
        if (!gStart.IsZero())
        {
            EcPoint PntOfs = ec.MultiplyG(gStart);
            PntOfs.y.NegModP();
            PntToSolve = ec.AddPoints(PntToSolve, PntOfs);
        }
        gTargetsShifted.push_back(PntToSolve);

        if (!gRange || !gDP)
        {
            printf("error: you must specify -range and -dp\r\n");
            goto label_end;
        }

        if (!SolveMultiTargets(gRange, gDP))
        {
            if (!gIsOpsLimit)
                printf("FATAL ERROR: Solve failed\r\n");
            goto label_end;
        }
    }
    else
    {
        if (gGenMode) printf("\r\nTAMES GENERATION MODE\r\n");
        else printf("\r\nBENCHMARK MODE\r\n");

        while (1)
        {
            EcInt pk, pk_found;
            EcPoint PntToSolve;

            if (!gRange) gRange = 78;
            if (!gDP)    gDP    = 16;

            pk.RndBits(gRange);
            PntToSolve = ec.MultiplyG(pk);

            // Reuse multi-solver with single synthetic target
            gTargetsOriginal.clear();
            gTargetsShifted.clear();
            gTargetsOriginal.push_back(PntToSolve);
            gTargetsShifted.push_back(PntToSolve);

            if (!SolveMultiTargets(gRange, gDP))
            {
                if (!gIsOpsLimit) printf("FATAL ERROR: Solve failed\r\n");
                break;
            }

            // gPrivKey already final (no offset in bench loop)
            if (!gPrivKey.IsEqual(pk))
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
    for (int i = 0; i < GpuCnt; i++) delete GpuKangs[i];
    DeInitEc();
    free(pPntList2);
    free(pPntList);
    return 0;
}
