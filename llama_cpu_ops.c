/*
 * llama_cpu_ops.c — Accelerate-optimized CPU ops for Llama-3.2-1B
 *
 * Replaces numpy CPU ops in run_llama_fused.py with vDSP/BLAS calls.
 * ANE dispatch stays in Python via ct.predict (MIL IR 2-input fusion).
 *
 * Functions:
 *   llama_rms_norm      — RMSNorm via vDSP_vsq + vDSP_meanv + rsqrt + vDSP_vmul
 *   llama_rope           — RoPE (rotate_half, theta=500000) via vDSP
 *   llama_gqa_attention   — GQA 32Q/8KV heads via cblas_sgemm
 *   llama_softmax         — Stable softmax via vDSP + vvexpf
 *   llama_embedding       — FP32 embed lookup → FP16 output
 *   llama_argmax          — Argmax via vDSP_maxvi
 *   llama_fp16_to_fp32    — vImage conversion
 *   llama_fp32_to_fp16    — vImage conversion
 *
 * Build:
 *   xcrun clang -O2 -shared -framework Accelerate -fobjc-arc \
 *     -o libllama_cpu_ops.dylib llama_cpu_ops.c
 *
 * Copyright 2026 Nick Lo. MIT License.
 */

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

// ─── FP16 ↔ FP32 conversion (vImage) ───

void llama_fp16_to_fp32(const uint16_t *in, float *out, int n) {
    vImage_Buffer src = {(void *)in, 1, (vImagePixelCount)n, n * 2};
    vImage_Buffer dst = {out, 1, (vImagePixelCount)n, n * 4};
    vImageConvert_Planar16FtoPlanarF(&src, &dst, 0);
}

void llama_fp32_to_fp16(const float *in, uint16_t *out, int n) {
    vImage_Buffer src = {(void *)in, 1, (vImagePixelCount)n, n * 4};
    vImage_Buffer dst = {out, 1, (vImagePixelCount)n, n * 2};
    vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0);
}

// ─── RMSNorm ───
// RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
// Input:  x_fp16[dim], weight_fp32[dim]
// Output: out_fp16[dim]

void llama_rms_norm(const uint16_t *x_fp16, const float *weight_fp32,
                    uint16_t *out_fp16, int dim, float eps) {
    // Allocate FP32 scratch
    float *x_f32 = (float *)malloc(dim * sizeof(float));
    float *tmp   = (float *)malloc(dim * sizeof(float));

    // Convert input to FP32
    llama_fp16_to_fp32(x_fp16, x_f32, dim);

    // tmp = x^2
    vDSP_vsq(x_f32, 1, tmp, 1, dim);

    // mean(x^2)
    float mean_sq;
    vDSP_meanv(tmp, 1, &mean_sq, dim);

    // rsqrt(mean_sq + eps)
    float rms_inv = 1.0f / sqrtf(mean_sq + eps);

    // tmp = x * rms_inv
    vDSP_vsmul(x_f32, 1, &rms_inv, tmp, 1, dim);

    // tmp = tmp * weight
    vDSP_vmul(tmp, 1, weight_fp32, 1, tmp, 1, dim);

    // Convert to FP16
    llama_fp32_to_fp16(tmp, out_fp16, dim);

    free(x_f32);
    free(tmp);
}

// ─── RoPE ───
// Llama rotate_half convention: split first/second half, not interleaved.
// x1 = x[:half_dim], x2 = x[half_dim:]
// rotated = concat(-x2, x1)
// output = x * cos + rotated * sin
//
// q: [n_q_heads * head_dim] FP16 (contiguous, head-major)
// k: [n_kv_heads * head_dim] FP16

void llama_rope(const uint16_t *q_fp16, const uint16_t *k_fp16,
                uint16_t *q_out_fp16, uint16_t *k_out_fp16,
                int n_q_heads, int n_kv_heads, int head_dim,
                int position, double theta) {
    int half_dim = head_dim / 2;

    // Precompute cos/sin table for this position
    float *cos_vals = (float *)malloc(half_dim * sizeof(float));
    float *sin_vals = (float *)malloc(half_dim * sizeof(float));
    for (int i = 0; i < half_dim; i++) {
        double freq = 1.0 / pow(theta, (double)(i * 2) / (double)head_dim);
        double angle = (double)position * freq;
        cos_vals[i] = (float)cos(angle);
        sin_vals[i] = (float)sin(angle);
    }

    // Full cos/sin arrays [head_dim] = concat(cos, cos) and concat(sin, sin)
    float *cos_full = (float *)malloc(head_dim * sizeof(float));
    float *sin_full = (float *)malloc(head_dim * sizeof(float));
    memcpy(cos_full, cos_vals, half_dim * sizeof(float));
    memcpy(cos_full + half_dim, cos_vals, half_dim * sizeof(float));
    memcpy(sin_full, sin_vals, half_dim * sizeof(float));
    memcpy(sin_full + half_dim, sin_vals, half_dim * sizeof(float));

    // Scratch buffers
    float *x_f32     = (float *)malloc(head_dim * sizeof(float));
    float *rot_f32   = (float *)malloc(head_dim * sizeof(float));
    float *term1     = (float *)malloc(head_dim * sizeof(float));
    float *term2     = (float *)malloc(head_dim * sizeof(float));
    float *result    = (float *)malloc(head_dim * sizeof(float));

    // Helper: apply RoPE to one head
    #define APPLY_ROPE(in_fp16, out_fp16, offset)                              \
    do {                                                                        \
        llama_fp16_to_fp32((in_fp16) + (offset), x_f32, head_dim);            \
        /* rotate_half: rot = concat(-x[half:], x[:half]) */                    \
        float neg_one = -1.0f;                                                  \
        /* rot[:half] = -x[half:] */                                            \
        vDSP_vsmul(x_f32 + half_dim, 1, &neg_one, rot_f32, 1, half_dim);      \
        /* rot[half:] = x[:half] */                                             \
        memcpy(rot_f32 + half_dim, x_f32, half_dim * sizeof(float));           \
        /* term1 = x * cos */                                                   \
        vDSP_vmul(x_f32, 1, cos_full, 1, term1, 1, head_dim);                 \
        /* term2 = rot * sin */                                                 \
        vDSP_vmul(rot_f32, 1, sin_full, 1, term2, 1, head_dim);               \
        /* result = term1 + term2 */                                            \
        vDSP_vadd(term1, 1, term2, 1, result, 1, head_dim);                   \
        llama_fp32_to_fp16(result, (out_fp16) + (offset), head_dim);           \
    } while(0)

    // Apply to all Q heads
    for (int h = 0; h < n_q_heads; h++) {
        APPLY_ROPE(q_fp16, q_out_fp16, h * head_dim);
    }

    // Apply to all KV heads
    for (int h = 0; h < n_kv_heads; h++) {
        APPLY_ROPE(k_fp16, k_out_fp16, h * head_dim);
    }

    #undef APPLY_ROPE

    free(cos_vals);
    free(sin_vals);
    free(cos_full);
    free(sin_full);
    free(x_f32);
    free(rot_f32);
    free(term1);
    free(term2);
    free(result);
}

// ─── Softmax ───
// Numerically stable: subtract max, exp, normalize.

void llama_softmax(float *x, int n) {
    float max_val;
    vDSP_maxv(x, 1, &max_val, n);
    float neg_max = -max_val;
    vDSP_vsadd(x, 1, &neg_max, x, 1, n);

    int n_int = n;
    vvexpf(x, x, &n_int);

    float sum;
    vDSP_sve(x, 1, &sum, n);
    float inv_sum = 1.0f / sum;
    vDSP_vsmul(x, 1, &inv_sum, x, 1, n);
}

// ─── GQA Attention ───
// q_fp16:    [n_heads * head_dim] FP16 — current token's Q after RoPE
// cached_k:  [seq_len * n_kv_heads * head_dim] FP16 — full KV cache K
// cached_v:  [seq_len * n_kv_heads * head_dim] FP16 — full KV cache V
// out_fp16:  [n_heads * head_dim] FP16 — attention output
//
// GQA: n_heads=32, n_kv_heads=8, n_rep=4
// Each Q head h uses KV head h // n_rep

void llama_gqa_attention(const uint16_t *q_fp16,
                         const uint16_t *cached_k_fp16,
                         const uint16_t *cached_v_fp16,
                         uint16_t *out_fp16,
                         int n_heads, int n_kv_heads, int head_dim,
                         int seq_len) {
    int n_rep = n_heads / n_kv_heads;
    int dim = n_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    // Convert Q to FP32
    float *q_f32 = (float *)malloc(dim * sizeof(float));
    llama_fp16_to_fp32(q_fp16, q_f32, dim);

    // Convert full KV cache to FP32
    int kv_size = seq_len * n_kv_heads * head_dim;
    float *k_f32 = (float *)malloc(kv_size * sizeof(float));
    float *v_f32 = (float *)malloc(kv_size * sizeof(float));
    llama_fp16_to_fp32(cached_k_fp16, k_f32, kv_size);
    llama_fp16_to_fp32(cached_v_fp16, v_f32, kv_size);

    // Output in FP32
    float *out_f32 = (float *)malloc(dim * sizeof(float));
    memset(out_f32, 0, dim * sizeof(float));

    // Scratch for attention scores
    float *scores = (float *)malloc(seq_len * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / n_rep;
        float *q_h = &q_f32[h * head_dim];                 // [head_dim]
        float *out_h = &out_f32[h * head_dim];              // [head_dim]

        // Compute scores = Q @ K^T for this head
        // K for kv_h is stored as [seq_len, n_kv_heads, head_dim]
        // K[s, kv_h] is at k_f32[s * n_kv_heads * head_dim + kv_h * head_dim]
        for (int s = 0; s < seq_len; s++) {
            float dot;
            vDSP_dotpr(q_h, 1,
                       &k_f32[s * n_kv_heads * head_dim + kv_h * head_dim], 1,
                       &dot, head_dim);
            scores[s] = dot * scale;
        }

        // Softmax
        llama_softmax(scores, seq_len);

        // attn_out[h] = scores @ V
        // V[s, kv_h] at v_f32[s * n_kv_heads * head_dim + kv_h * head_dim]
        memset(out_h, 0, head_dim * sizeof(float));
        for (int s = 0; s < seq_len; s++) {
            float w = scores[s];
            float *v_s = &v_f32[s * n_kv_heads * head_dim + kv_h * head_dim];
            vDSP_vsma(v_s, 1, &w, out_h, 1, out_h, 1, head_dim);
        }
    }

    // Convert to FP16
    llama_fp32_to_fp16(out_f32, out_fp16, dim);

    free(q_f32);
    free(k_f32);
    free(v_f32);
    free(out_f32);
    free(scores);
}

// ─── Embedding Lookup ───
// embed_table: [vocab_size * dim] FP32
// out_fp16:    [dim] FP16

void llama_embedding(const float *embed_table, int token_id,
                     uint16_t *out_fp16, int dim) {
    llama_fp32_to_fp16(&embed_table[token_id * dim], out_fp16, dim);
}

// ─── Argmax ───
// logits_fp32: [vocab_size] FP32
// Returns token ID

int llama_argmax(const float *logits_fp32, int vocab_size) {
    float max_val;
    vDSP_Length max_idx = 0;
    vDSP_maxvi(logits_fp32, 1, &max_val, &max_idx, vocab_size);
    return (int)max_idx;
}

// ─── Concatenate logit chunks ───
// Converts each FP16 chunk to FP32 and places at correct offset.

void llama_concat_logits_fp16_to_fp32(const uint16_t *chunk_fp16,
                                       float *logits_fp32,
                                       int offset, int chunk_size) {
    llama_fp16_to_fp32(chunk_fp16, &logits_fp32[offset], chunk_size);
}
