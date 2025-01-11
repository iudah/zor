#ifndef ZOR_H
#define ZOR_H

#include <stdint.h>

typedef struct zor zor;

zor *zor_init(uint8_t rank, uint32_t *restrict shape);

void zor_free(zor *restrict tensor);

zor *zor_zeros(uint8_t rank, uint32_t *restrict shape);

void zor_srandom(uint64_t seed);

zor *zor_random(uint8_t rank, uint32_t *restrict shape, float min, float max);

zor *zor_reshape(zor *restrict tensor, uint8_t rank, uint32_t *shape);

zor *zor_transpose(zor *restrict tensor, int32_t *restrict axes);

#define ELLIPSIS ((int)2147483647)
zor *zor_slice(zor *tensor, uint32_t n_slice_triples,
               int32_t **restrict slice_triples);

__attribute__((
    warn_unused_result("Ensure to check return value of function"))) bool
zor_get_element(zor *restrict tensor, const int *restrict indices,
                float *restrict value);

__attribute__((
    warn_unused_result("Ensure to check return value of function"))) bool bool
zor_set_element(zor *restrict tensor, const int *restrict indices, float value);

zor *zor_add(zor *a, zor *b);

zor *zor_subtract(zor *a, zor *b);

zor *zor_multiply(zor *a, zor *b);

zor *zor_divide(zor *a, zor *b);

zor *zor_sum(zor *restrict tensor, int axis);

zor *zor_mean(zor *restrict tensor, int axis);

zor *zor_max(zor *restrict tensor, int axis);

zor *zor_min(zor *restrict tensor, int axis);

#endif