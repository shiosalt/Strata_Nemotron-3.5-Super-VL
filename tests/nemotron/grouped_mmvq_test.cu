// tests/nemotron/grouped_mmvq_test.cu - the decode path's grouped expert products (src/nemotron/grouped_mmvq.cu) per weight type, at the
// real model's widths: three experts' matrices and two input rows (q8_1), five entries in one launch, against the float product of the
// dequantized weights with the float input (the q8_1 rounding of the input is the only difference: rel. L2 well under 1%), and against
// the per-entry native MMVQ (the same blocks, so within float rounding).  No model, a few MB of VRAM.
#include "grouped_mmvq.cuh"

#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"

#include <cmath>
#include <exception>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace G = strata::nemotron::gpu;
namespace K = strata::kernels;

namespace {
int failed = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-110s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++failed;
}
#define CK(x)                                                                                                      \
    do {                                                                                                           \
        const cudaError_t e_ = (x);                                                                                \
        if (e_ != cudaSuccess) {                                                                                   \
            std::printf("CUDA: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);                     \
            std::exit(1);                                                                                          \
        }                                                                                                          \
    } while (0)

float rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        n += (double) (a[i] - b[i]) * (a[i] - b[i]);
        d += (double) b[i] * b[i];
    }
    return (float) std::sqrt(n / std::max(d, 1e-30));
}

void type_case(ggml_type type, int n_in, int n_out) {
    const int E = 3, X = 2;
    std::mt19937 rng(1234 + (int) type);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // the experts' matrices, quantized; their dequantized values for the reference
    const size_t row_bytes = ggml_row_size(type, n_in), mat_bytes = row_bytes * (size_t) n_out;
    std::vector<uint8_t> q((size_t) E * mat_bytes);
    std::vector<float> wf((size_t) E * n_out * n_in);
    for (float& v : wf) v = nd(rng);
    std::vector<float> imat((size_t) n_in, 1.0f);
    ggml_quantize_chunk(type, wf.data(), q.data(), 0, (int64_t) E * n_out, n_in, ggml_quantize_requires_imatrix(type) ? imat.data() : nullptr);
    ggml_get_type_traits(type)->to_float(q.data(), wf.data(), (int64_t) wf.size());
    std::vector<float> x((size_t) X * n_in);
    for (float& v : x) v = nd(rng);
    // five entries: (expert, input row) -> output row
    const int ent_e[5] = {0, 2, 1, 2, 0}, ent_x[5] = {0, 0, 1, 1, 1};
    std::vector<float> want((size_t) 5 * n_out);
    for (int i = 0; i < 5; ++i)
        for (int r = 0; r < n_out; ++r) {
            double s = 0;
            const float* w = wf.data() + ((size_t) ent_e[i] * n_out + r) * n_in;
            const float* xv = x.data() + (size_t) ent_x[i] * n_in;
            for (int k = 0; k < n_in; ++k) s += (double) w[k] * xv[k];
            want[(size_t) i * n_out + r] = (float) s;
        }
    uint8_t* dq = nullptr;
    float *dx = nullptr, *dy = nullptr, *dy1 = nullptr;
    void* dx8 = nullptr;
    G::GroupedEntry* dent = nullptr;
    const size_t x8 = K::native_q8_1_bytes(n_in, 1);
    cudaStream_t st = nullptr;
    CK(cudaStreamCreate(&st));
    CK(cudaMalloc(&dq, q.size() + 4096));
    CK(cudaMemset(dq, 0, q.size() + 4096));
    CK(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dx, x.size() * 4));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dx8, x8 * X));
    K::native_quantize_q8_1(dx, dx8, n_in, X, st);
    CK(cudaMalloc(&dy, want.size() * 4));
    CK(cudaMalloc(&dy1, want.size() * 4));
    std::vector<G::GroupedEntry> ent(5);
    for (int i = 0; i < 5; ++i) ent[(size_t) i] = G::GroupedEntry{dq + (size_t) ent_e[i] * mat_bytes, ent_x[i], i};
    CK(cudaMalloc(&dent, sizeof(G::GroupedEntry) * 5));
    CK(cudaMemcpy(dent, ent.data(), sizeof(G::GroupedEntry) * 5, cudaMemcpyHostToDevice));
    G::grouped_mmvq((int) type, dent, 5, 0, dx8, x8, false, dy, (size_t) n_out, n_in, n_out, st);
    for (int i = 0; i < 5; ++i)
        K::native_mmvq((int) type, dq + (size_t) ent_e[i] * mat_bytes, (const char*) dx8 + (size_t) ent_x[i] * x8, dy1 + (size_t) i * n_out, n_in,
                       n_out, 1, st);
    CK(cudaStreamSynchronize(st));
    CK(cudaStreamDestroy(st));
    std::vector<float> got(want.size()), one(want.size());
    CK(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(one.data(), dy1, one.size() * 4, cudaMemcpyDeviceToHost));
    const float e_ref = rel(got, want), e_one = rel(got, one);
    char buf[200];
    std::snprintf(buf, sizeof buf, "%s, %d x %d, 5 entries of 3 experts: vs the float product rel. L2 %.2e, vs the per-entry MMVQ %.2e",
                  ggml_type_name(type), n_out, n_in, (double) e_ref, (double) e_one);
    // (Q5_0: ggml's dot subtracts 16 x the input block's sum, which q8_1 keeps in fp16: about 1.4% here, the per-entry MMVQ the same)
    const float lim = type == GGML_TYPE_Q5_0 ? 2e-2f : 1e-2f;
    check(e_ref < lim && e_one < 1e-5f, buf);
    for (void* p : {(void*) dq, (void*) dx, (void*) dy, (void*) dy1, dx8, (void*) dent}) CK(cudaFree(p));
}
}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("  no CUDA GPU: skipped\n");
        return 77;
    }
    // the up experts (latent 1024 in) and the down experts (2688 in) of the three Unsloth sets
    try {
    for (ggml_type t : {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ2_XS, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K})
        type_case(t, 1024, 96);
    for (ggml_type t : {GGML_TYPE_IQ4_NL, GGML_TYPE_Q2_0, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0}) type_case(t, 2688, 96);
    } catch (const std::exception& e) {
        std::printf("  exception: %s\n", e.what());
        ++failed;
    }
    std::printf(failed ? "FAILED: %d\n" : "all ok\n", failed);
    return failed ? 1 : 0;
}
