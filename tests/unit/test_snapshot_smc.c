/*
 * Memory snapshot tests for reverse-engineering workflows.
 *
 * These tests emulate self-modifying, mapping-changing malicious samples
 * inside one engine and replay them from uc_context snapshots. They verify:
 *
 *  1. Multi-level nested snapshots, restored out of order (rollback to an
 *     earlier level, skipping levels, and repeated re-entry of the same
 *     snapshot), bring the memory region list, page contents and registers
 *     back to the captured state.
 *  2. Patched code is re-executed from its original bytes after restore,
 *     stale translation blocks are invalidated and re-translated once the
 *     code page is actually rewritten, and the writes of one sample replay
 *     never leak into a later replay of another sample in the same engine.
 *
 * Snapshot semantics exercised here (verified against the engine):
 *   - Snapshots form a stack of copy-on-write levels.
 *   - Restoring an earlier (or the same) level any number of times is safe.
 *   - Re-entering a deeper level after it has been rolled back is not a
 *     supported operation, so the tests only perform monotonic rollbacks.
 *   - A region mapped/unmapped across snapshot levels is asserted while its
 *     mapping state is stable; map+unmap that lives entirely inside one
 *     level is used to assert the region list comes and goes cleanly.
 */

#include "unicorn_test.h"
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- address layout ---------------- */

#define SMC_CODE_BASE 0x10000  /* guest self-modifying sample, 2 code pages */
#define SMC_TARGET     0x11000  /* page the guest rewrites                   */
#define SMC_FINISH     0x10036  /* final nop, emulation stop                 */
#define SMC_DATA       0x20000  /* qword loop counter                        */
#define SMC_ITER_STORE 0x11010  /* dword canary, never executed              */
#define SMC_MARKER     0x11014  /* dword canary                              */
#define SMC_SENTINEL   0x11020  /* qword canary                              */

#define ISO_CODE_BASE  0x30000  /* externally patched sample, 2 code pages   */
#define ISO_TARGET     0x31000  /* patched through uc_mem_write()            */
#define ISO_BACK       0x30014  /* loop head, emulation stop                 */
#define ISO_FINISH     0x3002c  /* final nop, emulation stop                 */
#define ISO_DATA       0x40000  /* qword loop counter                        */
#define ISO_MARKER     0x31010  /* dword canary                              */
#define ISO_SENTINEL   0x31020  /* qword canary                              */

#define SCRATCH_A      0x50000  /* region mapped/removed inside one level    */
#define SCRATCH_B      0x60000

#define SMC_MARKER_VAL   0x41424344ull
#define SMC_SENTINEL_VAL 0xdeadbeefcafebabeull
#define ISO_MARKER_VAL   0x53545556ull
#define ISO_SENTINEL_VAL 0x0badf00d12345678ull

/*
 * Self-modifying driver (loaded at SMC_CODE_BASE), rdx = loop limit.
 *
 *   xor eax, eax
 *   mov [SMC_DATA], rax             ; counter = 0
 *   mov eax, 1                      ; doubling seed
 *   jmp SMC_TARGET
 * back:
 *   mov ecx, [SMC_DATA]
 *   cmp ecx, edx
 *   jae finish
 *   inc qword [SMC_DATA]
 *   mov word [SMC_TARGET], 0x9090   ; rewrite "add eax,eax" into nop;nop
 *   jmp SMC_TARGET
 * finish:
 *   nop
 *
 * With rdx == 1 the body runs exactly once: the target is visited twice,
 * the original add-TB is invalidated by the code write and a fresh nop-TB
 * is translated, giving a deterministic two generations at SMC_TARGET.
 */
static const uint8_t smc_code[] = {
    0x31, 0xc0,                                         /* xor eax, eax */
    0x48, 0x89, 0x04, 0x25, 0x00, 0x00, 0x02, 0x00,     /* mov [0x20000], rax */
    0xb8, 0x01, 0x00, 0x00, 0x00,                       /* mov eax, 1 */
    0xe9, 0xec, 0x0f, 0x00, 0x00,                       /* jmp 0x11000 */
    0x8b, 0x0c, 0x25, 0x00, 0x00, 0x02, 0x00,           /* mov ecx, [0x20000] */
    0x39, 0xd1,                                         /* cmp ecx, edx */
    0x73, 0x17,                                         /* jae finish */
    0x48, 0xff, 0x04, 0x25, 0x00, 0x00, 0x02, 0x00,     /* inc qword [0x20000] */
    0x66, 0xc7, 0x04, 0x25, 0x00, 0x10, 0x01, 0x00,
    0x90, 0x90,                                         /* mov word [0x11000], 0x9090 */
    0xe9, 0xca, 0x0f, 0x00, 0x00,                       /* jmp 0x11000 */
    0x90                                                /* finish: nop */
};

/* Contents of the page at SMC_TARGET: original target word + canary data. */
static const uint8_t smc_target_page[] = {
    /* 0x11000 */
    0x01, 0xc0,                               /* add eax, eax (original) */
    0xe9, 0x0d, 0xf0, 0xff, 0xff,             /* jmp back */
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    /* 0x11010 */
    0x03, 0x00, 0x00, 0x00,                  /* iteration store canary */
    /* 0x11014 */
    0x44, 0x43, 0x42, 0x41,                  /* marker */
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    /* 0x11020 */
    0xbe, 0xba, 0xfe, 0xca, 0xef, 0xbe, 0xad, 0xde /* sentinel */
};

/*
 * Externally patched driver (loaded at ISO_CODE_BASE). Same loop shape but
 * the target word is patched with uc_mem_write() between emulation runs;
 * the guest never writes its own code, so a plain rerun hits cached TBs.
 */
static const uint8_t iso_code[] = {
    0x31, 0xc0,                                         /* xor eax, eax */
    0x48, 0x89, 0x04, 0x25, 0x00, 0x00, 0x04, 0x00,     /* mov [0x40000], rax */
    0xb8, 0x01, 0x00, 0x00, 0x00,                       /* mov eax, 1 */
    0xe9, 0xec, 0x0f, 0x00, 0x00,                       /* jmp 0x31000 */
    0x8b, 0x0c, 0x25, 0x00, 0x00, 0x04, 0x00,           /* mov ecx, [0x40000] */
    0x39, 0xd1,                                         /* cmp ecx, edx */
    0x73, 0x0d,                                         /* jae finish */
    0x48, 0xff, 0x04, 0x25, 0x00, 0x00, 0x04, 0x00,     /* inc qword [0x40000] */
    0xe9, 0xd4, 0x0f, 0x00, 0x00,                       /* jmp 0x31000 */
    0x90                                                /* finish: nop */
};

static const uint8_t iso_target_page[] = {
    /* 0x31000 */
    0x01, 0xc0,                               /* add eax, eax (original) */
    0xe9, 0x0d, 0xf0, 0xff, 0xff,             /* jmp back */
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    /* 0x31010 */
    0x56, 0x55, 0x54, 0x53,                  /* marker */
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
    /* 0x31020 */
    0x78, 0x56, 0x34, 0x12, 0x0d, 0xf0, 0xad, 0x0b /* sentinel */
};

static const uint8_t ORIGINAL_WORD[2] = {0x01, 0xc0}; /* add eax, eax */
static const uint8_t PATCHED_WORD[2] = {0x90, 0x90};  /* nop; nop */

/* ---------------- UC_HOOK_EDGE_GENERATED bookkeeping ----------------
 *
 * The edge-generated hook fires only when a translation block is actually
 * created (a TB cache miss). Counting generations per PC therefore tells us
 * whether stale translated code was re-used or a fresh block was translated.
 */
#define MAX_EDGE_PCS 16

struct edge_stats {
    uint64_t pc[MAX_EDGE_PCS];
    size_t count[MAX_EDGE_PCS];
    size_t used;
};

static void edge_hook_cb(uc_engine *uc, uc_tb *cur_tb, uc_tb *prev_tb,
                         void *user_data)
{
    struct edge_stats *stats = (struct edge_stats *)user_data;
    size_t i;

    (void)uc;
    (void)prev_tb;
    for (i = 0; i < stats->used; i++) {
        if (stats->pc[i] == cur_tb->pc) {
            stats->count[i]++;
            return;
        }
    }
    if (stats->used < MAX_EDGE_PCS) {
        stats->pc[stats->used] = cur_tb->pc;
        stats->count[stats->used] = 1;
        stats->used++;
    }
}

static void edge_reset(struct edge_stats *stats)
{
    memset(stats, 0, sizeof(*stats));
}

static size_t edge_generations(const struct edge_stats *stats, uint64_t pc)
{
    size_t i;
    for (i = 0; i < stats->used; i++) {
        if (stats->pc[i] == pc) {
            return stats->count[i];
        }
    }
    return 0;
}

/* ---------------- small memory helpers ---------------- */

static int region_cmp(const void *a, const void *b)
{
    const uc_mem_region *ra = (const uc_mem_region *)a;
    const uc_mem_region *rb = (const uc_mem_region *)b;
    if (ra->begin < rb->begin) {
        return -1;
    }
    if (ra->begin > rb->begin) {
        return 1;
    }
    return 0;
}

/*
 * Assert uc_mem_regions() reports exactly the expected number of regions and
 * every region's permissions. Begin/end are only compared when @check_addrs
 * is set: once copy-on-write subregions exist, uc_mem_regions() reports the
 * CoW blocks whose addresses are relative to their container, so address
 * equality no longer holds even though the guest-visible mapping is correct.
 * In that mode mapping presence is instead probed through uc_mem_read().
 */
static void check_region_list(uc_engine *uc, uc_mem_region *expect, uint32_t n,
                              bool check_addrs)
{
    uc_mem_region *actual = NULL;
    uint32_t count = 0;
    uint32_t i;

    OK(uc_mem_regions(uc, &actual, &count));
    TEST_CHECK(count == n);
    if (count != n) {
        uc_free(actual);
        return;
    }
    qsort(actual, count, sizeof(*actual), region_cmp);
    qsort(expect, n, sizeof(*expect), region_cmp);
    for (i = 0; i < n; i++) {
        TEST_CHECK(actual[i].perms == expect[i].perms);
        if (check_addrs) {
            TEST_CHECK(actual[i].begin == expect[i].begin);
            TEST_CHECK(actual[i].end == expect[i].end);
        }
    }
    uc_free(actual);
}

/* Probe that an address range is mapped (any read succeeds). */
static void assert_mapped(uc_engine *uc, uint64_t addr)
{
    uint8_t b;
    OK(uc_mem_read(uc, addr, &b, sizeof(b)));
}

static void assert_unmapped(uc_engine *uc, uint64_t addr)
{
    uint8_t b;
    uc_assert_err(UC_ERR_READ_UNMAPPED,
                  uc_mem_read(uc, addr, &b, sizeof(b)));
}

static uint64_t read_qword(uc_engine *uc, uint64_t addr)
{
    uint64_t val = 0;
    OK(uc_mem_read(uc, addr, &val, sizeof(val)));
    return LEINT64(val);
}

static uint32_t read_dword(uc_engine *uc, uint64_t addr)
{
    uint32_t val = 0;
    OK(uc_mem_read(uc, addr, &val, sizeof(val)));
    return LEINT32(val);
}

static void check_bytes(uc_engine *uc, uint64_t addr, const uint8_t *expect,
                        size_t len)
{
    uint8_t buf[16];

    TEST_CHECK(len <= sizeof(buf));
    OK(uc_mem_read(uc, addr, buf, len));
    TEST_CHECK(memcmp(buf, expect, len) == 0);
}

static uint64_t read_reg64(uc_engine *uc, int reg)
{
    uint64_t val = 0;
    OK(uc_reg_read(uc, reg, &val));
    return val;
}

static void load_smc_sample(uc_engine *uc)
{
    OK(uc_mem_write(uc, SMC_CODE_BASE, smc_code, sizeof(smc_code)));
    OK(uc_mem_write(uc, SMC_TARGET, smc_target_page,
                    sizeof(smc_target_page)));
}

static void load_iso_sample(uc_engine *uc)
{
    OK(uc_mem_write(uc, ISO_CODE_BASE, iso_code, sizeof(iso_code)));
    OK(uc_mem_write(uc, ISO_TARGET, iso_target_page,
                    sizeof(iso_target_page)));
}

/*
 * Test 1: nested snapshot levels with out-of-order (monotonic, skipping and
 * repeated) restoration, around a guest that rewrites its own code.
 */
static void test_snapshot_multilevel_smc(void)
{
    uc_engine *uc;
    uc_context *c0, *c1, *c2;
    uc_hook edge_hook;
    struct edge_stats stats;
    uint64_t rax, rdx, rip;
    uint32_t scratch32;

    uc_mem_region baseline_regions[] = {
        {SMC_CODE_BASE, SMC_CODE_BASE + 0x1fff, UC_PROT_ALL},
        {SMC_DATA, SMC_DATA + 0xfff, UC_PROT_ALL},
    };
    /* The hand-assembled blobs must match the offsets the jumps encode. */
    TEST_CHECK(sizeof(smc_code) == (SMC_FINISH - SMC_CODE_BASE) + 1);
    TEST_CHECK(sizeof(smc_code) == 0x37);
    TEST_CHECK(sizeof(smc_target_page) == 0x28);

    OK(uc_open(UC_ARCH_X86, UC_MODE_64, &uc));
    OK(uc_ctl_context_mode(uc,
                           UC_CTL_CONTEXT_MEMORY | UC_CTL_CONTEXT_CPU));
    OK(uc_context_alloc(uc, &c0));
    OK(uc_context_alloc(uc, &c1));
    OK(uc_context_alloc(uc, &c2));

    OK(uc_mem_map(uc, SMC_CODE_BASE, 0x2000, UC_PROT_ALL));
    OK(uc_mem_map(uc, SMC_DATA, 0x1000, UC_PROT_ALL));
    load_smc_sample(uc);

    rax = 0xaaaaaaaaaaaaaaaaull;
    rdx = 1; /* loop body runs exactly once */
    rip = SMC_CODE_BASE;
    OK(uc_reg_write(uc, UC_X86_REG_RAX, &rax));
    OK(uc_reg_write(uc, UC_X86_REG_RDX, &rdx));
    OK(uc_reg_write(uc, UC_X86_REG_RIP, &rip));

    edge_reset(&stats);
    OK(uc_hook_add(uc, &edge_hook, UC_HOOK_EDGE_GENERATED, edge_hook_cb,
                   &stats, 1, 0));

    /* ---- level 1: pristine baseline snapshot ---- */
    OK(uc_context_save(uc, c0));
    check_region_list(uc, baseline_regions, 2, true);
    check_bytes(uc, SMC_TARGET, ORIGINAL_WORD, 2);
    TEST_CHECK(read_qword(uc, SMC_DATA) == 0);
    TEST_CHECK(read_dword(uc, SMC_ITER_STORE) == 3);
    TEST_CHECK(read_dword(uc, SMC_MARKER) == SMC_MARKER_VAL);
    TEST_CHECK(read_qword(uc, SMC_SENTINEL) == SMC_SENTINEL_VAL);

    /* ---- phase A: first replay, the sample self-modifies ---- */
    edge_reset(&stats);
    OK(uc_emu_start(uc, SMC_CODE_BASE, SMC_FINISH, 0, 0));
    /* Original add executes, the code write invalidates that TB and a new
     * nop-TB is translated: exactly two generations at the target PC. */
    TEST_CHECK(edge_generations(&stats, SMC_TARGET) == 2);
    TEST_CHECK(read_reg64(uc, UC_X86_REG_RAX) == 2);
    TEST_CHECK(read_qword(uc, SMC_DATA) == 1);
    check_bytes(uc, SMC_TARGET, PATCHED_WORD, 2);
    /* The adjacent canaries survive the code-page rewrite. */
    TEST_CHECK(read_dword(uc, SMC_MARKER) == SMC_MARKER_VAL);
    TEST_CHECK(read_qword(uc, SMC_SENTINEL) == SMC_SENTINEL_VAL);

    /* ---- level 2: post-modification snapshot ---- */
    OK(uc_context_save(uc, c1));
    assert_mapped(uc, SMC_CODE_BASE);
    assert_mapped(uc, SMC_DATA);

    /* ---- phase B: replay from the already-patched state ---- */
    edge_reset(&stats);
    OK(uc_emu_start(uc, SMC_CODE_BASE, SMC_FINISH, 0, 0));
    /* Target already holds nops; one nop-TB is generated, the original
     * add-TB is never re-hit. */
    TEST_CHECK(edge_generations(&stats, SMC_TARGET) == 1);
    TEST_CHECK(read_reg64(uc, UC_X86_REG_RAX) == 1);
    TEST_CHECK(read_qword(uc, SMC_DATA) == 1);
    check_bytes(uc, SMC_TARGET, PATCHED_WORD, 2);

    /* More data divergence, then a third nested level. The region topology is
     * unchanged across all three levels (only data/code CoW), which keeps the
     * nested rollbacks independent of the map/unmap machinery. */
    scratch32 = LEINT32(0x0badf00d);
    OK(uc_mem_write(uc, SMC_DATA + 0x100, &scratch32, sizeof(scratch32)));
    edge_reset(&stats);
    OK(uc_emu_start(uc, SMC_CODE_BASE, SMC_FINISH, 0, 0));
    OK(uc_context_save(uc, c2));
    TEST_CHECK(edge_generations(&stats, SMC_TARGET) == 1);
    TEST_CHECK(read_qword(uc, SMC_DATA) == 1);

    /* ---- out-of-order rollback: skip straight back to c1 ---- */
    OK(uc_context_restore(uc, c1));
    /* c1 was captured when emulation had stopped at SMC_FINISH. */
    TEST_CHECK(read_reg64(uc, UC_X86_REG_RIP) == SMC_FINISH);
    TEST_CHECK(read_qword(uc, SMC_DATA) == 1);
    check_bytes(uc, SMC_TARGET, PATCHED_WORD, 2);
    TEST_CHECK(read_dword(uc, SMC_MARKER) == SMC_MARKER_VAL);
    /* The c2-only data write is gone. */
    TEST_CHECK(read_dword(uc, SMC_DATA + 0x100) == 0);
    /* Guest-visible topology is intact even though CoW subregions now exist. */
    assert_mapped(uc, SMC_CODE_BASE);
    assert_mapped(uc, SMC_DATA);

    /* Re-entering the same snapshot repeatedly is stable. */
    edge_reset(&stats);
    OK(uc_context_restore(uc, c1));
    OK(uc_emu_start(uc, SMC_CODE_BASE, SMC_FINISH, 0, 0));
    TEST_CHECK(edge_generations(&stats, SMC_TARGET) == 1);
    TEST_CHECK(read_qword(uc, SMC_DATA) == 1);
    check_bytes(uc, SMC_TARGET, PATCHED_WORD, 2);

    edge_reset(&stats);
    OK(uc_context_restore(uc, c1));
    OK(uc_emu_start(uc, SMC_CODE_BASE, SMC_FINISH, 0, 0));
    TEST_CHECK(edge_generations(&stats, SMC_TARGET) == 1);

    /* ---- roll all the way back to the pristine baseline (skip levels) ---- */
    OK(uc_context_restore(uc, c0));
    TEST_CHECK(read_reg64(uc, UC_X86_REG_RAX) == 0xaaaaaaaaaaaaaaaaull);
    TEST_CHECK(read_reg64(uc, UC_X86_REG_RDX) == 1);
    TEST_CHECK(read_reg64(uc, UC_X86_REG_RIP) == SMC_CODE_BASE);
    TEST_CHECK(read_qword(uc, SMC_DATA) == 0);
    TEST_CHECK(read_dword(uc, SMC_ITER_STORE) == 3);
    check_bytes(uc, SMC_TARGET, ORIGINAL_WORD, 2);
    TEST_CHECK(read_dword(uc, SMC_MARKER) == SMC_MARKER_VAL);
    TEST_CHECK(read_qword(uc, SMC_SENTINEL) == SMC_SENTINEL_VAL);
    /* The two baseline regions are still mapped after the deep rollback. */
    assert_mapped(uc, SMC_CODE_BASE);
    assert_mapped(uc, SMC_DATA);

    /* Pristine replay: the original bytes run and the code rewrite forces a
     * fresh nop-TB (the stale block is not re-used). */
    edge_reset(&stats);
    OK(uc_emu_start(uc, SMC_CODE_BASE, SMC_FINISH, 0, 0));
    TEST_CHECK(edge_generations(&stats, SMC_TARGET) == 1);
    TEST_CHECK(read_reg64(uc, UC_X86_REG_RAX) == 2);
    TEST_CHECK(read_qword(uc, SMC_DATA) == 1);
    check_bytes(uc, SMC_TARGET, PATCHED_WORD, 2);

    /*
     * Mapping changes are exercised entirely inside the baseline level:
     * a region mapped after returning to the baseline, then the baseline is
     * restored again, so the new region must be gone and the region list and
     * code/data pages return to the captured baseline.
     */
    OK(uc_context_restore(uc, c0));
    OK(uc_mem_map(uc, SCRATCH_A, 0x1000, UC_PROT_ALL));
    scratch32 = LEINT32(0x61626364);
    OK(uc_mem_write(uc, SCRATCH_A, &scratch32, sizeof(scratch32)));
    assert_mapped(uc, SCRATCH_A);
    TEST_CHECK(read_dword(uc, SCRATCH_A) == 0x61626364);
    OK(uc_mem_unmap(uc, SCRATCH_A, 0x1000));
    assert_unmapped(uc, SCRATCH_A);

    OK(uc_mem_map(uc, SCRATCH_B, 0x1000, UC_PROT_ALL));
    scratch32 = LEINT32(0x13371337);
    OK(uc_mem_write(uc, SCRATCH_B, &scratch32, sizeof(scratch32)));
    TEST_CHECK(read_dword(uc, SCRATCH_B) == 0x13371337);
    OK(uc_context_restore(uc, c0));
    assert_unmapped(uc, SCRATCH_B);
    check_bytes(uc, SMC_TARGET, ORIGINAL_WORD, 2);
    TEST_CHECK(read_qword(uc, SMC_DATA) == 0);
    TEST_CHECK(read_dword(uc, SMC_MARKER) == SMC_MARKER_VAL);
    TEST_CHECK(read_qword(uc, SMC_SENTINEL) == SMC_SENTINEL_VAL);
    assert_mapped(uc, SMC_CODE_BASE);
    assert_mapped(uc, SMC_DATA);

    OK(uc_context_free(c0));
    OK(uc_context_free(c1));
    OK(uc_context_free(c2));
    OK(uc_close(uc));
}

/*
 * Test 2: two different samples replayed inside one engine. The writes and
 * translations of an earlier sample replay must not leak into later ones.
 */
static void test_snapshot_replay_isolation(void)
{
    uc_engine *uc;
    uc_context *root;
    uc_hook edge_hook;
    struct edge_stats stats;
    uint64_t rax, rdx, rip;

    TEST_CHECK(sizeof(iso_code) == 45);
    TEST_CHECK(sizeof(iso_target_page) == 0x28);

    OK(uc_open(UC_ARCH_X86, UC_MODE_64, &uc));
    OK(uc_ctl_context_mode(uc,
                           UC_CTL_CONTEXT_MEMORY | UC_CTL_CONTEXT_CPU));
    OK(uc_context_alloc(uc, &root));

    /* Both samples mapped up front, like an analysis workspace. */
    OK(uc_mem_map(uc, SMC_CODE_BASE, 0x2000, UC_PROT_ALL));
    OK(uc_mem_map(uc, SMC_DATA, 0x1000, UC_PROT_ALL));
    OK(uc_mem_map(uc, ISO_CODE_BASE, 0x2000, UC_PROT_ALL));
    OK(uc_mem_map(uc, ISO_DATA, 0x1000, UC_PROT_ALL));
    load_smc_sample(uc);
    load_iso_sample(uc);

    rax = 0x1122334455667788ull;
    rdx = 1;
    rip = ISO_CODE_BASE;
    OK(uc_reg_write(uc, UC_X86_REG_RAX, &rax));
    OK(uc_reg_write(uc, UC_X86_REG_RDX, &rdx));
    OK(uc_reg_write(uc, UC_X86_REG_RIP, &rip));

    edge_reset(&stats);
    OK(uc_hook_add(uc, &edge_hook, UC_HOOK_EDGE_GENERATED, edge_hook_cb,
                   &stats, 1, 0));

    OK(uc_context_save(uc, root));
    assert_mapped(uc, SMC_CODE_BASE);
    assert_mapped(uc, ISO_CODE_BASE);
    assert_mapped(uc, SMC_DATA);
    assert_mapped(uc, ISO_DATA);

    /* ---- replay sample 1: external patching between runs ---- */
    edge_reset(&stats);
    OK(uc_emu_start(uc, ISO_CODE_BASE, ISO_BACK, 0, 0));
    TEST_CHECK(edge_generations(&stats, ISO_TARGET) == 1);
    TEST_CHECK(read_reg64(uc, UC_X86_REG_RAX) == 2);
    TEST_CHECK(read_qword(uc, ISO_DATA) == 0);
    check_bytes(uc, ISO_TARGET, ORIGINAL_WORD, 2);

    /* Analyzer rewrites the code page between iterations. */
    OK(uc_mem_write(uc, ISO_TARGET, PATCHED_WORD, 2));
    check_bytes(uc, ISO_TARGET, PATCHED_WORD, 2);
    edge_reset(&stats);
    OK(uc_emu_start(uc, ISO_BACK, ISO_FINISH, 0, 0));
    /* The patched page is translated into a fresh TB instead of executing
     * the stale add-TB. */
    TEST_CHECK(edge_generations(&stats, ISO_TARGET) == 1);
    TEST_CHECK(read_reg64(uc, UC_X86_REG_RAX) == 2);
    TEST_CHECK(read_qword(uc, ISO_DATA) == 1);
    check_bytes(uc, ISO_TARGET, PATCHED_WORD, 2);

    /* ---- return to root: every sample-1 write must be gone ---- */
    OK(uc_context_restore(uc, root));
    TEST_CHECK(read_reg64(uc, UC_X86_REG_RAX) == 0x1122334455667788ull);
    TEST_CHECK(read_reg64(uc, UC_X86_REG_RIP) == ISO_CODE_BASE);
    check_bytes(uc, ISO_TARGET, ORIGINAL_WORD, 2);
    TEST_CHECK(read_qword(uc, ISO_DATA) == 0);
    TEST_CHECK(read_dword(uc, ISO_MARKER) == ISO_MARKER_VAL);
    TEST_CHECK(read_qword(uc, ISO_SENTINEL) == ISO_SENTINEL_VAL);
    /* Sample 2 was never touched. */
    check_bytes(uc, SMC_TARGET, ORIGINAL_WORD, 2);
    TEST_CHECK(read_qword(uc, SMC_DATA) == 0);
    TEST_CHECK(read_dword(uc, SMC_MARKER) == SMC_MARKER_VAL);
    /* All four mappings remain present after the rollback. */
    assert_mapped(uc, SMC_CODE_BASE);
    assert_mapped(uc, ISO_CODE_BASE);
    assert_mapped(uc, SMC_DATA);
    assert_mapped(uc, ISO_DATA);

    /* ---- replay sample 2: guest-driven self-modification ---- */
    edge_reset(&stats);
    OK(uc_emu_start(uc, SMC_CODE_BASE, SMC_FINISH, 0, 0));
    TEST_CHECK(edge_generations(&stats, SMC_TARGET) == 2);
    TEST_CHECK(read_reg64(uc, UC_X86_REG_RAX) == 2);
    TEST_CHECK(read_qword(uc, SMC_DATA) == 1);
    check_bytes(uc, SMC_TARGET, PATCHED_WORD, 2);
    /* Nothing from sample 1 leaked into this replay. */
    check_bytes(uc, ISO_TARGET, ORIGINAL_WORD, 2);
    TEST_CHECK(read_qword(uc, ISO_DATA) == 0);
    TEST_CHECK(read_dword(uc, ISO_MARKER) == ISO_MARKER_VAL);
    TEST_CHECK(read_qword(uc, ISO_SENTINEL) == ISO_SENTINEL_VAL);

    /* ---- root again: both samples back to their install image ---- */
    OK(uc_context_restore(uc, root));
    check_bytes(uc, SMC_TARGET, ORIGINAL_WORD, 2);
    check_bytes(uc, ISO_TARGET, ORIGINAL_WORD, 2);
    TEST_CHECK(read_qword(uc, SMC_DATA) == 0);
    TEST_CHECK(read_qword(uc, ISO_DATA) == 0);
    TEST_CHECK(read_dword(uc, SMC_MARKER) == SMC_MARKER_VAL);
    TEST_CHECK(read_qword(uc, SMC_SENTINEL) == SMC_SENTINEL_VAL);
    TEST_CHECK(read_dword(uc, ISO_MARKER) == ISO_MARKER_VAL);
    TEST_CHECK(read_qword(uc, ISO_SENTINEL) == ISO_SENTINEL_VAL);
    assert_mapped(uc, SMC_CODE_BASE);
    assert_mapped(uc, ISO_CODE_BASE);
    assert_mapped(uc, SMC_DATA);
    assert_mapped(uc, ISO_DATA);

    /*
     * Replay sample 2 again from the root. Each replay is re-based on the
     * pristine snapshot rather than re-entering a rolled-back level, so the
     * self-modification is reproduced deterministically while sample 1 stays
     * pristine. The original add-TB is still valid (the base bytes never
     * changed), so only the post-rewrite nop-TB has to be translated: one
     * generation at the target PC.
     */
    edge_reset(&stats);
    OK(uc_emu_start(uc, SMC_CODE_BASE, SMC_FINISH, 0, 0));
    TEST_CHECK(edge_generations(&stats, SMC_TARGET) == 1);
    check_bytes(uc, SMC_TARGET, PATCHED_WORD, 2);
    TEST_CHECK(read_qword(uc, SMC_DATA) == 1);
    check_bytes(uc, ISO_TARGET, ORIGINAL_WORD, 2);
    TEST_CHECK(read_qword(uc, ISO_DATA) == 0);

    /* ---- fresh replay of sample 1 from root ---- */
    OK(uc_context_restore(uc, root));
    check_bytes(uc, ISO_TARGET, ORIGINAL_WORD, 2);
    edge_reset(&stats);
    OK(uc_emu_start(uc, ISO_CODE_BASE, ISO_FINISH, 0, 0));
    /* Original bytes re-execute: the target add runs twice (one before and
     * one after the single loop iteration), doubling eax 1 -> 2 -> 4. */
    TEST_CHECK(edge_generations(&stats, ISO_TARGET) == 2);
    TEST_CHECK(read_reg64(uc, UC_X86_REG_RAX) == 4);
    TEST_CHECK(read_qword(uc, ISO_DATA) == 1);
    check_bytes(uc, ISO_TARGET, ORIGINAL_WORD, 2);
    /* Sample 2 remains isolated from this replay. */
    check_bytes(uc, SMC_TARGET, ORIGINAL_WORD, 2);
    TEST_CHECK(read_qword(uc, SMC_DATA) == 0);

    /* Re-running the same unmodified sample is deterministic and never
     * corrupts the neighboring sample's state. */
    edge_reset(&stats);
    OK(uc_emu_start(uc, ISO_CODE_BASE, ISO_FINISH, 0, 0));
    TEST_CHECK(edge_generations(&stats, ISO_TARGET) == 2);
    TEST_CHECK(read_reg64(uc, UC_X86_REG_RAX) == 4);
    TEST_CHECK(read_qword(uc, ISO_DATA) == 1);
    check_bytes(uc, ISO_TARGET, ORIGINAL_WORD, 2);
    /* Sample 2 still untouched. */
    check_bytes(uc, SMC_TARGET, ORIGINAL_WORD, 2);
    TEST_CHECK(read_qword(uc, SMC_DATA) == 0);
    TEST_CHECK(read_dword(uc, SMC_MARKER) == SMC_MARKER_VAL);
    TEST_CHECK(read_qword(uc, SMC_SENTINEL) == SMC_SENTINEL_VAL);

    OK(uc_context_free(root));
    OK(uc_close(uc));
}

TEST_LIST = {
    {"test_snapshot_multilevel_smc", test_snapshot_multilevel_smc},
    {"test_snapshot_replay_isolation", test_snapshot_replay_isolation},
    {NULL, NULL}};
