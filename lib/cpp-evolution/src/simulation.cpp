#include "mapping.h"

#include <cstdlib>
#include <algorithm>
#include <unordered_map>

#if USE_AVX || USE_PEXT
#include <immintrin.h>
#define W 8
#endif


#if USE_AVX
#ifdef DEBUG_AVX
namespace {
// AVX debugging functions

/// Dump an AVX(2) vector of 32-bit floats
void dump_vec_ps(__m256 v) {
    float* buf = (float*)valloc(W * sizeof(float));
    _mm256_stream_ps(buf, v);
    std::cerr << "fvec:";
    for (int i = W-1; i >= 0; --i) {
        std::cerr << " " << buf[i];
    }
    std::cerr << "\n";

    free(buf);
}

/// Dump an AVX(2) vector of 32-bit ints
void dump_vec_epi32(__m256i v) {
    int size = 12;
    unsigned* buf = (unsigned*)valloc(W * sizeof(int));
    _mm256_stream_si256((__m256i*)buf, v);
    fprintf(stderr, "ivec:");
    for (int i = W-1; i >= 0; --i) {
        fprintf(stderr, " ");
        int val = buf[i];
        for (int j = size; j >= 0; --j) {
            fprintf(stderr, "%d", !!(val & 1 << j) );
        }
    }
    fprintf(stderr, "\n");

    free(buf);
}

}
#endif
#endif

#if !USE_GUROBI && !USE_Z3
double Mapping::simulateExperiment(const Architecture &arch, const Experiment &e) const {
    std::unordered_map<Uop, num_type> uop_map;

#if USE_PEXT
    Uop used_ports = 0;
#endif

    for (const auto& i : e.getInsnSeq()) {
        const auto &used_uops = *UopMap_.at(i);
        for (const auto &[u, n] : used_uops) {
            uop_map[u] += n;
#if USE_PEXT
            used_ports |= u;
#endif
        }
    }

    Uop max_uop = arch.getLargestUop();


#ifdef USE_AVX
    // Manually vectorized simulation code. The non-vectorized reference is
    // below.
    static thread_local Uop *uops = nullptr;
    static thread_local float *vals = nullptr;

    if (uops == nullptr) {
        uops = (Uop*)malloc((max_uop + 1) * sizeof(*uops));
        vals = (float*)malloc((max_uop + 1) * sizeof(*vals));
    }

    int_fast32_t num_uops = uop_map.size();
    int_fast32_t i = 0;

#if USE_PEXT
    if (max_uop & ~used_ports) { // there are unused ports, compress
        max_uop = std::max(_pext_u32(max_uop, used_ports), (1u << W) - 1);
        for (auto &[u, n] : uop_map) {
            uops[i] = _pext_u32(u, used_ports);
            vals[i] = (float)n;
            ++i;
        }
    } else
#endif
    {
        for (auto &[u, n] : uop_map) {
            uops[i] = u;
            vals[i] = (float)n;
            ++i;
        }
    }

    assert((max_uop + 1 ) % W == 0);
    assert((max_uop + 1) >= W);

    __m256 max_vals = _mm256_setzero_ps();

    __m256i q_offsets = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256 popcnt_offsets_first = _mm256_set_ps(1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 3.0); // popcnts of q_offsets + 1 for the zero uop

#ifdef DEBUG_AVX
    std::vector<std::pair<Uop, num_type>> uop_vec(uop_map.begin(), uop_map.end());

    float ref_val = 0.0;
#endif

    auto loop_it = [&](Uop current_q, __m256 popcnt_offsets) {
        float popcnt_scalar = (float)__builtin_popcount(current_q);
        __m256i q_base = _mm256_set1_epi32(current_q);
        __m256i q = _mm256_or_si256(q_base, q_offsets);

        __m256 popcnt_base = _mm256_set1_ps(popcnt_scalar);
        __m256 popcnt = _mm256_add_ps(popcnt_base, popcnt_offsets);

        __m256 val = _mm256_setzero_ps(); // float val = 0.0;

        for (int_fast32_t j = 0; j < num_uops; ++j) {
            __m256i uop = _mm256_set1_epi32(uops[j]); // load uop entry
            __m256i cond = _mm256_andnot_si256(q, uop);
            __m256i zero = _mm256_setzero_si256();
            __m256 mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cond, zero));
            __m256 n_vals = _mm256_broadcast_ss(vals+j);
            __m256 sum = _mm256_add_ps(val, n_vals);
            val = _mm256_blendv_ps(val, sum, mask);
        }

        __m256 div_val = _mm256_div_ps(val, popcnt);
        // 1 2 3 4 5 6 7 8

        max_vals = _mm256_max_ps(max_vals, div_val);
#ifdef DEBUG_AVX
        // std::cerr << "vec_val: ";
        // dump_vec_epi32(q_base);
        // std::cerr << "vec_val: ";
        // dump_vec_epi32(q_offsets);
        // dump_vec_epi32(q);
        // std::cerr << "vec_val: ";
        // dump_vec_ps(popcnt);
        // dump_vec_ps(val);
        // dump_vec_ps(div_val);

        // std::cerr << "scalar_val:   ";
        for (Uop local_q = current_q; local_q < current_q + W; ++local_q) {
            float val = 0.0;
            for (const auto &[u, n] : uop_vec) {
                if ((~local_q & u) == 0){
                    val += n;
                }
            }
            // std::cerr << " " << local_q;
            // std::cerr << " " << val;
            val = val / ((local_q == 0) ? 1 : __builtin_popcount(local_q));
            // std::cerr << " " << val;
            if (val > ref_val) {
                ref_val = val;
            }
        }
        // std::cerr << std::endl;
#endif
    };

    loop_it(0, popcnt_offsets_first);

    __m256 popcnt_offsets_later = _mm256_set_ps(0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 3.0); // popcnts of q_offsets

    for (Uop current_q = W; current_q <= max_uop; current_q += W) {
        loop_it(current_q, popcnt_offsets_later);
    }

    // compute maximum by shuffling and vertical maxing
    __m256 tmp1 = _mm256_permute2f128_ps(max_vals, max_vals, 1);
    // 5 6 7 8 1 2 3 4
    __m256 tmp2 = _mm256_max_ps(max_vals, tmp1);
    // (1,5) (2,6) (3,7) (4,8) (1,5) (2,6) (3,7) (4,8)
    __m256 tmp3 = _mm256_permute_ps(tmp2, (2 << 0) | (3 << 2) | (0 << 4) | (1 << 6));
    // (3,7) (4,8) (1,5) (2,6) (3,7) (4,8) (1,5) (2,6)
    __m256 tmp4 = _mm256_max_ps(tmp2, tmp3);
    // (1,3,5,7) (2,4,6,8) (1,3,5,7) (2,4,6,8) (1,3,5,7) (2,4,6,8) (1,3,5,7) (2,4,6,8)
    __m256 tmp5 = _mm256_permute_ps(tmp4, (1 << 0) | (0 << 2) | (3 << 4) | (2 << 6));
    // (2,4,6,8) (1,3,5,7) (2,4,6,8) (1,3,5,7) (2,4,6,8) (1,3,5,7) (2,4,6,8) (1,3,5,7)
    __m256 tmp6 = _mm256_max_ps(tmp4, tmp5); // all max

    float res = _mm256_cvtss_f32(tmp6);

#ifdef DEBUG_AVX
        if (res != ref_val) {
            std::cerr << "res: " << res << ", ref: " << ref_val << "\n";
        }

        assert(res == ref_val);
#endif

    return (double)res;
#else
    // This is the non-vectorized version.
    std::vector<std::pair<Uop, num_type>> uop_vec(uop_map.begin(), uop_map.end());

    double max_val = 0.0;

    for (Uop current_q = 1; current_q <= max_uop; ++current_q) {
        double val = 0.0;
        for (const auto &[u, n] : uop_vec) {
            if ((~current_q & u) == 0){
                val += n;
            }
        }
        val = val / __builtin_popcount(current_q);
        if (val > max_val) {
            max_val = val;
        }
    }
    return max_val;
#endif
}
#endif

