#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <VapourSynth.h>
#include <VSHelper.h>


#ifdef _WIN32
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif


static FORCE_INLINE int process_chroma_pixel_scalar(
        int pixel_PP,
        int pixel_P,
        int pixel_C,
        int pixel_N,
        int pixel_NN,
        int chroma_t1,
        int chroma_t2) {

    bool lteq_t1 =
            std::abs(pixel_P - pixel_N) <= chroma_t1 &&
            std::abs(pixel_C - pixel_PP) <= chroma_t1 &&
            std::abs(pixel_C - pixel_NN) <= chroma_t1;

    int abs_diff_CP = std::abs(pixel_C - pixel_P);
    int abs_diff_CN = std::abs(pixel_C - pixel_N);

    bool lteq_t1_gt_t2 = lteq_t1 &&
            abs_diff_CP > chroma_t2 &&
            abs_diff_CN > chroma_t2;

    int avg_pc = (pixel_P + pixel_C + 1) >> 1;
    int avg_nc = (pixel_N + pixel_C + 1) >> 1;

    int avg_nc_or_pc = abs_diff_CN <= abs_diff_CP ? avg_nc : avg_pc;

    return lteq_t1_gt_t2 ? avg_nc_or_pc : pixel_C;
}


static FORCE_INLINE int process_luma_pixel_scalar(
        int pixel_current_left,
        int pixel_current,
        int pixel_current_right,
        int pixel_current_2above,
        int pixel_current_2below,
        int pixel_2previous,
        int pixel_previous,
        int pixel_next,
        int pixel_2next,
        int luma_2d,
        int luma_t) {

    int left_right = pixel_current_left + pixel_current_right;
    int above_below = pixel_current_2above + pixel_current_2below;
    int center_center = pixel_current * 2;

    int abs_diff_horizontal = std::abs(left_right - center_center);
    int abs_diff_vertical = std::abs(above_below - center_center);

    int result = pixel_current;

    if (abs_diff_horizontal > luma_2d || abs_diff_vertical > luma_2d) {
        bool temporal_okay =
                std::abs(pixel_previous - pixel_next) <= luma_t &&
                std::abs(pixel_current - pixel_2previous) <= luma_t &&
                std::abs(pixel_current - pixel_2next) <= luma_t;

        if (temporal_okay) {
            int avg_pc = (pixel_previous + pixel_current + 1) >> 1;
            int avg_nc = (pixel_next + pixel_current + 1) >> 1;

            int abs_diff_pc = std::abs(pixel_previous - pixel_current);
            int abs_diff_nc = std::abs(pixel_next - pixel_current);

            int avg_nc_or_avg_pc = abs_diff_nc <= abs_diff_pc ? avg_nc : avg_pc;

            result = avg_nc_or_avg_pc;
        }
    }

    return result;
}


#if defined (DEDOT_X86)

#include <emmintrin.h>


#define zeroes _mm_setzero_si128()


static FORCE_INLINE __m128i mm_abs_diff_epu8(const __m128i &a, const __m128i &b) {
    return _mm_max_epu8(_mm_subs_epu8(a, b),
                        _mm_subs_epu8(b, a));
}


static FORCE_INLINE __m128i mm_abs_diff_epu16(const __m128i &a, const __m128i &b) {
    return _mm_or_si128(_mm_subs_epu16(a, b),
                        _mm_subs_epu16(b, a));
}


static FORCE_INLINE __m128i abs_diff_lteq_threshold_mask(
        const __m128i &a,
        const __m128i &b,
        const __m128i &threshold) {

    __m128i abs_diff = mm_abs_diff_epu8(a, b);
    abs_diff = _mm_subs_epu8(abs_diff, threshold);

    // Pixels less than or equal to threshold.
    __m128i lteq_mask = _mm_cmpeq_epi8(abs_diff, zeroes);

    return lteq_mask;
}


/// Currently 1 sub, 2 cmpeqb
/// Could be 1 sub, 1 cmpgtb if the threshold was moved into signed byte range in advance
static FORCE_INLINE __m128i abs_diff_gt_threshold_mask(
        const __m128i &mmmabs,
        const __m128i &threshold) {

    __m128i abs_diff = _mm_subs_epu8(mmmabs, threshold);

    // Pixels less than or equal to threshold.
    __m128i lteq_mask = _mm_cmpeq_epi8(abs_diff, zeroes);
    // Pixels greater than threshold.
    __m128i gt_mask = _mm_cmpeq_epi8(lteq_mask, zeroes);

    return gt_mask;
}


static void process_chroma_plane_sse2(
        const uint8_t *pPP,
        const uint8_t *pP,
        const uint8_t *pC,
        const uint8_t *pN,
        const uint8_t *pNN,
        uint8_t *pD,
        const int width_U,
        const int height_U,
        const int stride_U,
        const int chroma_t1,
        const int chroma_t2) {

    const int pixels_in_xmm = 16;

    int width_U_simd = width_U / pixels_in_xmm * pixels_in_xmm;

    __m128i bytes_chroma_t1 = _mm_set1_epi8(chroma_t1);
    __m128i bytes_chroma_t2 = _mm_set1_epi8(chroma_t2);

    for (int y = 0; y < height_U; y++) {
        for (int x = 0; x < width_U_simd; x += pixels_in_xmm) {
            __m128i pixel_PP = _mm_load_si128((const __m128i *)&pPP[x]);
            __m128i pixel_P = _mm_load_si128((const __m128i *)&pP[x]);
            __m128i pixel_C = _mm_load_si128((const __m128i *)&pC[x]);
            __m128i pixel_N = _mm_load_si128((const __m128i *)&pN[x]);
            __m128i pixel_NN = _mm_load_si128((const __m128i *)&pNN[x]);

            __m128i lteq_T1_mask = _mm_and_si128(_mm_and_si128(abs_diff_lteq_threshold_mask(pixel_P, pixel_N, bytes_chroma_t1),
                                                               abs_diff_lteq_threshold_mask(pixel_C, pixel_PP, bytes_chroma_t1)),
                                                 abs_diff_lteq_threshold_mask(pixel_C, pixel_NN, bytes_chroma_t1));

            __m128i abs_diff_CP = mm_abs_diff_epu8(pixel_C, pixel_P);
            __m128i abs_diff_CN = mm_abs_diff_epu8(pixel_C, pixel_N);

            __m128i lteq_T1_gt_T2_mask = _mm_and_si128(_mm_and_si128(lteq_T1_mask,
                                                                     abs_diff_gt_threshold_mask(abs_diff_CP, bytes_chroma_t2)),
                                                       abs_diff_gt_threshold_mask(abs_diff_CN, bytes_chroma_t2));

            __m128i avg_nc_or_pc_mask = _mm_cmpeq_epi8(_mm_subs_epu8(abs_diff_CN, abs_diff_CP),
                                                       zeroes);

            __m128i avg_pc = _mm_avg_epu8(pixel_P, pixel_C);
            __m128i avg_nc = _mm_avg_epu8(pixel_N, pixel_C);

            __m128i avg_nc_or_pc = _mm_or_si128(_mm_and_si128(avg_nc, avg_nc_or_pc_mask),
                                                _mm_andnot_si128(avg_nc_or_pc_mask, avg_pc));

            __m128i result = _mm_or_si128(_mm_and_si128(avg_nc_or_pc, lteq_T1_gt_T2_mask),
                                          _mm_andnot_si128(lteq_T1_gt_T2_mask, pixel_C));

            _mm_store_si128((__m128i *)&pD[x], result);
        }

        for (int x = width_U_simd; x < width_U; x++) {
            int pixel_PP = pPP[x];
            int pixel_P = pP[x];
            int pixel_C = pC[x];
            int pixel_N = pN[x];
            int pixel_NN = pNN[x];

            pD[x] = process_chroma_pixel_scalar(pixel_PP, pixel_P, pixel_C, pixel_N, pixel_NN, chroma_t1, chroma_t2);
        }

        pPP += stride_U;
        pP += stride_U;
        pC += stride_U;
        pN += stride_U;
        pNN += stride_U;
        pD += stride_U;
    }
}


static void process_luma_plane_sse2(
        const uint8_t *pPP,
        const uint8_t *pP,
        const uint8_t *pC,
        const uint8_t *pN,
        const uint8_t *pNN,
        uint8_t *pD,
        const int width,
        const int height,
        const int stride,
        const int bytesPerSample,
        const int luma_2d,
        const int luma_t) {

    const int pixels_in_xmm = 16;

    int width_simd = (width - 1 * 2) / pixels_in_xmm * pixels_in_xmm;

    __m128i words_luma_2d = _mm_set1_epi16(luma_2d);
    __m128i bytes_luma_t = _mm_set1_epi8(luma_t);


    for (int y = 0; y < 2; y++) {
        memcpy(pD, pC, width * bytesPerSample);

        pD += stride;
        pC += stride;
    }

    for (int y = 2; y < height - 2; y++) {
        pD[0] = pC[0];

        for (int x = 1; x < 1 + width_simd; x += pixels_in_xmm) {
            // luma2d
            __m128i pixel_current_left  = _mm_loadu_si128((const __m128i *)&pC[x - 1]);
            __m128i pixel_current =       _mm_loadu_si128((const __m128i *)&pC[x]);
            __m128i pixel_current_right = _mm_loadu_si128((const __m128i *)&pC[x + 1]);

            __m128i pixel_current_2above = _mm_loadu_si128((const __m128i *)&pC[x - stride * 2]);
            __m128i pixel_current_2below = _mm_loadu_si128((const __m128i *)&pC[x + stride * 2]);


            __m128i left_right_lo = _mm_add_epi16(_mm_unpacklo_epi8(pixel_current_left, zeroes),
                                                  _mm_unpacklo_epi8(pixel_current_right, zeroes));
            __m128i left_right_hi = _mm_add_epi16(_mm_unpackhi_epi8(pixel_current_left, zeroes),
                                                  _mm_unpackhi_epi8(pixel_current_right, zeroes));

            __m128i above_below_lo = _mm_add_epi16(_mm_unpacklo_epi8(pixel_current_2above, zeroes),
                                                   _mm_unpacklo_epi8(pixel_current_2below, zeroes));
            __m128i above_below_hi = _mm_add_epi16(_mm_unpackhi_epi8(pixel_current_2above, zeroes),
                                                   _mm_unpackhi_epi8(pixel_current_2below, zeroes));

            __m128i center_center_lo = _mm_slli_epi16(_mm_unpacklo_epi8(pixel_current, zeroes), 1);
            __m128i center_center_hi = _mm_slli_epi16(_mm_unpackhi_epi8(pixel_current, zeroes), 1);


            __m128i abs_diff_horizontal_lo = mm_abs_diff_epu16(left_right_lo, center_center_lo);
            __m128i abs_diff_vertical_lo = mm_abs_diff_epu16(above_below_lo, center_center_lo);

            __m128i abs_diff_horizontal_hi = mm_abs_diff_epu16(left_right_hi, center_center_hi);
            __m128i abs_diff_vertical_hi = mm_abs_diff_epu16(above_below_hi, center_center_hi);

            __m128i spatial_mask_lo = _mm_or_si128(_mm_cmpgt_epi16(abs_diff_horizontal_lo, words_luma_2d),
                                                   _mm_cmpgt_epi16(abs_diff_vertical_lo, words_luma_2d));

            __m128i spatial_mask_hi = _mm_or_si128(_mm_cmpgt_epi16(abs_diff_horizontal_hi, words_luma_2d),
                                                   _mm_cmpgt_epi16(abs_diff_vertical_hi, words_luma_2d));

            __m128i spatial_mask = _mm_packs_epi16(spatial_mask_lo, spatial_mask_hi);


            __m128i result = pixel_current;

            __m128i packed_spatial_mask = _mm_packs_epi16(spatial_mask, spatial_mask);
#if defined (DEDOT_32_BITS)
            int all_pixels = _mm_cvtsi128_si32(_mm_packs_epi16(packed_spatial_mask, packed_spatial_mask));
#else
            long long all_pixels = _mm_cvtsi128_si64(packed_spatial_mask);
#endif

            // Don't do the temporal stuff if all 16 pixels fail the spatial test.
            if (all_pixels != 0) {
                __m128i pixel_previous =  _mm_loadu_si128((const __m128i *)&pP[x]);
                __m128i pixel_next =      _mm_loadu_si128((const __m128i *)&pN[x]);
                __m128i pixel_2previous = _mm_loadu_si128((const __m128i *)&pPP[x]);
                __m128i pixel_2next =     _mm_loadu_si128((const __m128i *)&pNN[x]);


                // lumaT
                __m128i temporal_mask_pn = abs_diff_lteq_threshold_mask(pixel_previous, pixel_next, bytes_luma_t);
                __m128i st_mask = _mm_and_si128(spatial_mask, temporal_mask_pn);

                __m128i temporal_mask_cpp = abs_diff_lteq_threshold_mask(pixel_current, pixel_2previous, bytes_luma_t);
                st_mask = _mm_and_si128(st_mask, temporal_mask_cpp);

                __m128i temporal_mask_cnn = abs_diff_lteq_threshold_mask(pixel_current, pixel_2next, bytes_luma_t);
                st_mask = _mm_and_si128(st_mask, temporal_mask_cnn);

                // luma avg
                __m128i avg_pc = _mm_avg_epu8(pixel_previous, pixel_current);
                __m128i avg_nc = _mm_avg_epu8(pixel_next, pixel_current);

                __m128i abs_diff_pc = mm_abs_diff_epu8(pixel_previous, pixel_current);
                __m128i abs_diff_nc = mm_abs_diff_epu8(pixel_next, pixel_current);

                __m128i abs_diff_nc_lteq_abs_diff_pc_mask = _mm_cmpeq_epi8(_mm_subs_epu8(abs_diff_nc, abs_diff_pc),
                                                                           zeroes);
                __m128i avg_nc_or_avg_pc = _mm_or_si128(_mm_and_si128(abs_diff_nc_lteq_abs_diff_pc_mask, avg_nc),
                                                        _mm_andnot_si128(abs_diff_nc_lteq_abs_diff_pc_mask, avg_pc));

                result = _mm_or_si128(_mm_and_si128(avg_nc_or_avg_pc, st_mask),
                                      _mm_andnot_si128(st_mask, pixel_current));
            }

            _mm_storeu_si128((__m128i *)&pD[x], result);
        }

        for (int x = 1 + width_simd; x < width - 1; x++) {
            int pixel_current_left  = pC[x - 1];
            int pixel_current =       pC[x];
            int pixel_current_right = pC[x + 1];

            int pixel_current_2above = pC[x - stride * 2];
            int pixel_current_2below = pC[x + stride * 2];

            int pixel_previous =  pP[x];
            int pixel_next =      pN[x];
            int pixel_2previous = pPP[x];
            int pixel_2next =     pNN[x];

            pD[x] = process_luma_pixel_scalar(pixel_current_left, pixel_current, pixel_current_right,
                                              pixel_current_2above, pixel_current_2below,
                                              pixel_2previous, pixel_previous, pixel_next, pixel_2next,
                                              luma_2d, luma_t);
        }

        pD[width - 1] = pC[width - 1];

        pPP += stride;
        pP += stride;
        pC += stride;
        pN += stride;
        pNN += stride;
        pD += stride;
    }

    for (int y = height - 2; y < height; y++) {
        memcpy(pD, pC, width * bytesPerSample);

        pD += stride;
        pC += stride;
    }
}


#else // DEDOT_X86


static void process_chroma_plane_scalar(
        const uint8_t *pPP,
        const uint8_t *pP,
        const uint8_t *pC,
        const uint8_t *pN,
        const uint8_t *pNN,
        uint8_t *pD,
        const int width_U,
        const int height_U,
        const int stride_U,
        const int chroma_t1,
        const int chroma_t2) {

    for (int y = 0; y < height_U; y++) {
        for (int x = 0; x < width_U; x++) {
            int pixel_PP = pPP[x];
            int pixel_P = pP[x];
            int pixel_C = pC[x];
            int pixel_N = pN[x];
            int pixel_NN = pNN[x];

            pD[x] = process_chroma_pixel_scalar(pixel_PP, pixel_P, pixel_C, pixel_N, pixel_NN, chroma_t1, chroma_t2);
        }

        pPP += stride_U;
        pP += stride_U;
        pC += stride_U;
        pN += stride_U;
        pNN += stride_U;
        pD += stride_U;
    }
}


static void process_luma_plane_scalar(
        const uint8_t *pPP,
        const uint8_t *pP,
        const uint8_t *pC,
        const uint8_t *pN,
        const uint8_t *pNN,
        uint8_t *pD,
        const int width,
        const int height,
        const int stride,
        const int bytesPerSample,
        const int luma_2d,
        const int luma_t) {

    for (int y = 0; y < 2; y++) {
        memcpy(pD, pC, width * bytesPerSample);

        pD += stride;
        pC += stride;
    }

    for (int y = 2; y < height - 2; y++) {
        pD[0] = pC[0];

        for (int x = 1; x < width - 1; x++) {
            int pixel_current_left  = pC[x - 1];
            int pixel_current =       pC[x];
            int pixel_current_right = pC[x + 1];

            int pixel_current_2above = pC[x - stride * 2];
            int pixel_current_2below = pC[x + stride * 2];

            int pixel_previous =  pP[x];
            int pixel_next =      pN[x];
            int pixel_2previous = pPP[x];
            int pixel_2next =     pNN[x];

            pD[x] = process_luma_pixel_scalar(pixel_current_left, pixel_current, pixel_current_right,
                                              pixel_current_2above, pixel_current_2below,
                                              pixel_2previous, pixel_previous, pixel_next, pixel_2next,
                                              luma_2d, luma_t);
        }

        pD[width - 1] = pC[width - 1];

        pPP += stride;
        pP += stride;
        pC += stride;
        pN += stride;
        pNN += stride;
        pD += stride;
    }

    for (int y = height - 2; y < height; y++) {
        memcpy(pD, pC, width * bytesPerSample);

        pD += stride;
        pC += stride;
    }
}


#endif // DEDOT_X86


typedef struct DedotData {
    VSNodeRef *clip;
    const VSVideoInfo *vi;

    int process[3];

    int chroma_t1;
    int chroma_t2;
    int luma_2d;
    int luma_t;

} DedotData;


static void VS_CC dedotInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    (void)in;
    (void)out;
    (void)core;

    DedotData *d = (DedotData *) *instanceData;

    vsapi->setVideoInfo(d->vi, 1, node);
}


static const VSFrameRef *VS_CC dedotGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    (void)frameData;

    const DedotData *d = (const DedotData *) *instanceData;

    if (activationReason == arInitial) {
        for (int i = std::max(0, n - 2); i <= std::min(n + 2, d->vi->numFrames - 1); i++)
            vsapi->requestFrameFilter(i, d->clip, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef *srcPP = vsapi->getFrameFilter(std::max(0, n - 2), d->clip, frameCtx);
        const VSFrameRef *srcP = vsapi->getFrameFilter(std::max(0, n - 1), d->clip, frameCtx);
        const VSFrameRef *srcC = vsapi->getFrameFilter(n, d->clip, frameCtx);
        const VSFrameRef *srcN = vsapi->getFrameFilter(std::min(n + 1, d->vi->numFrames - 1), d->clip, frameCtx);
        const VSFrameRef *srcNN = vsapi->getFrameFilter(std::min(n + 2, d->vi->numFrames - 1), d->clip, frameCtx);

        const VSFrameRef *plane_src[3] = {
            d->process[0] ? nullptr : srcC,
            d->process[1] ? nullptr : srcC,
            d->process[2] ? nullptr : srcC
        };

        int planes[3] = { 0, 1, 2 };

        VSFrameRef *dst = vsapi->newVideoFrame2(d->vi->format, d->vi->width, d->vi->height, plane_src, planes, srcC, core);

        if (d->vi->format->colorFamily != cmGray && d->process[1] && d->process[2]) {
            for (int plane = 1; plane < 3; plane++) {
                const uint8_t *pPP = vsapi->getReadPtr(srcPP, plane);
                const uint8_t *pP = vsapi->getReadPtr(srcP, plane);
                const uint8_t *pC = vsapi->getReadPtr(srcC, plane);
                const uint8_t *pN = vsapi->getReadPtr(srcN, plane);
                const uint8_t *pNN = vsapi->getReadPtr(srcNN, plane);
                uint8_t *pD = vsapi->getWritePtr(dst, plane);

                int width = vsapi->getFrameWidth(srcC, plane);
                int height = vsapi->getFrameHeight(srcC, plane);
                int stride = vsapi->getStride(srcC, plane);

#if defined (DEDOT_X86)
                process_chroma_plane_sse2(
#else
                process_chroma_plane_scalar(
#endif
                            pPP, pP, pC, pN, pNN, pD,
                            width, height, stride,
                            d->chroma_t1, d->chroma_t2);
            }
        }

        if (d->process[0]) {
            int width = vsapi->getFrameWidth(srcC, 0);
            int height = vsapi->getFrameHeight(srcC, 0);
            int stride = vsapi->getStride(srcC, 0);

            const uint8_t *pPP = vsapi->getReadPtr(srcPP, 0) + 2 * stride;
            const uint8_t *pP = vsapi->getReadPtr(srcP, 0) + 2 * stride;
            const uint8_t *pC = vsapi->getReadPtr(srcC, 0);
            const uint8_t *pN = vsapi->getReadPtr(srcN, 0) + 2 * stride;
            const uint8_t *pNN = vsapi->getReadPtr(srcNN, 0) + 2 * stride;
            uint8_t *pD = vsapi->getWritePtr(dst, 0);

#if defined (DEDOT_X86)
            process_luma_plane_sse2(
#else
            process_luma_plane_scalar(
#endif
                        pPP, pP, pC, pN, pNN, pD,
                        width, height, stride,
                        d->vi->format->bytesPerSample,
                        d->luma_2d, d->luma_t);
        }

        vsapi->freeFrame(srcPP);
        vsapi->freeFrame(srcP);
        vsapi->freeFrame(srcC);
        vsapi->freeFrame(srcN);
        vsapi->freeFrame(srcNN);

        return dst;
    }

    return NULL;
}


static void VS_CC dedotFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    (void)core;

    DedotData *d = (DedotData *)instanceData;

    vsapi->freeNode(d->clip);
    free(d);
}


static void VS_CC dedotCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    (void)userData;

    DedotData d;
    memset(&d, 0, sizeof(d));

    int err;

    d.luma_2d = int64ToIntS(vsapi->propGetInt(in, "luma_2d", 0, &err));
    if (err)
        d.luma_2d = 20;

    d.luma_t = int64ToIntS(vsapi->propGetInt(in, "luma_t", 0, &err));
    if (err)
        d.luma_t = 20;

    d.chroma_t1 = int64ToIntS(vsapi->propGetInt(in, "chroma_t1", 0, &err));
    if (err)
        d.chroma_t1 = 15;

    d.chroma_t2 = int64ToIntS(vsapi->propGetInt(in, "chroma_t2", 0, &err));
    if (err)
        d.chroma_t2 = 5;


    if (d.luma_2d < 0 || d.luma_2d > 510) {
        vsapi->setError(out, "Dedot: luma_2d must be between 0 and 510 (inclusive).");
        return;
    }

    if (d.luma_t < 0 || d.luma_t > 255) {
        vsapi->setError(out, "Dedot: luma_t must be between 0 and 255 (inclusive).");
        return;
    }

    if (d.chroma_t1 < 0 || d.chroma_t1 > 255) {
        vsapi->setError(out, "Dedot: chroma_t1 must be between 0 and 255 (inclusive).");
        return;
    }

    if (d.chroma_t2 < 0 || d.chroma_t2 > 255) {
        vsapi->setError(out, "Dedot: chroma_t2 must be between 0 and 255 (inclusive).");
        return;
    }

    if ((d.luma_2d == 510 || d.luma_t == 0) && d.chroma_t2 == 255) {
        vsapi->setError(out, "Dedot: chroma_t2 can't be 255 when luma_2d is 510 or when luma_t is 0 because then all the planes would be returned unchanged.");
        return;
    }


    d.clip = vsapi->propGetNode(in, "clip", 0, NULL);
    d.vi = vsapi->getVideoInfo(d.clip);


    if (!d.vi->format ||
        (d.vi->format->colorFamily != cmGray && d.vi->format->colorFamily != cmYUV) ||
        d.vi->format->bitsPerSample > 8 ||
        d.vi->width == 0 ||
        d.vi->height == 0) {
        vsapi->setError(out, "Dedot: the input clip must be 8 bit YUV or Gray with constant format and dimensions.");
        vsapi->freeNode(d.clip);
        return;
    }


    d.process[0] = d.luma_2d < 510 && d.luma_t > 0;
    d.process[1] = d.process[2] = d.chroma_t2 < 255 && d.vi->format->colorFamily != cmGray;


    DedotData *data = (DedotData *)malloc(sizeof(d));
    *data = d;

    vsapi->createFilter(in, out, "Dedot", dedotInit, dedotGetFrame, dedotFree, fmParallel, 0, data, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.nodame.dedot", "dedot", "Temporal dotcrawl and rainbow remover", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("Dedot",
            "clip:clip;"
            "luma_2d:int:opt;"
            "luma_t:int:opt;"
            "chroma_t1:int:opt;"
            "chroma_t2:int:opt;"
            , dedotCreate, 0, plugin);
}
