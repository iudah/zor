#ifndef ZOR_H
#define ZOR_H

#include <stdint.h>

typedef struct zor zor;

void *zor_init(uint8_t rank, uint32_t *restrict shape);

uint8_t zor_rank(void *restrict tensor);

uint32_t *zor_shape(void *restrict self, uint32_t *shape);

void zor_swap_repr(void *restrict tensor, void *restrict swap);

void zor_swap_array(void *restrict tensor, void *restrict swap);

void zor_free(void *restrict tensor);

void *zor_ones(uint8_t rank, uint32_t *restrict shape);

void *zor_zeros(uint8_t rank, uint32_t *restrict shape);

void *zor_fill(uint8_t rank, uint32_t *restrict shape, float value);

void *zor_from_array(uint8_t rank, uint32_t *shape, float *numbers);

void zor_srandom(uint64_t seed);

void *zor_random(uint8_t rank, uint32_t *restrict shape, float min, float max);

void *zor_reshape(void *restrict tensor, uint8_t rank, uint32_t *shape);

void *zor_transpose(void *restrict tensor, int32_t *restrict axes);

#define ELLIPSIS ((void *)0x1)
void *zor_slice(void *tensor, uint32_t n_slice_triples,
                int32_t **restrict slice_triples);

__attribute__((warn_unused_result(
#ifdef __clang__
    "Ensure to check return value of function"
#endif
    ))) bool
zor_get_element(void *restrict tensor, const int *restrict indices,
                float *restrict value);

__attribute__((warn_unused_result(
#ifdef __clang__
    "Ensure to check return value of function"
#endif
    ))) bool
zor_set_element(void *restrict tensor, const int *restrict indices,
                float value);

void *zor_add(void *a, void *b);

void *zor_subtract(void *a, void *b);

void *zor_multiply(void *a, void *b);

void *zor_divide(void *a, void *b);

#define ZOR_REDUCE_AXIS_NONE ((uint8_t)-1)
void *zor_sum(void *restrict tensor, int axis);

void *zor_mean(void *restrict tensor, int axis);

void *zor_max(void *restrict tensor, int axis);

void *zor_min(void *restrict tensor, int axis);

void *zor_matmul(void *a, void *b);

void *zor_tensordot(void *a, void *b, int32_t n_axes, int32_t *a_axes,
                    int32_t *b_axes);

void *zor_copy(void *restrict tensor);

void *zor_negative(void *restrict tensor);

void *zor_relu(void *restrict tensor);

uint64_t zor_to_string(void *tensor, char *buffer, uint64_t buffer_limit);

void *zor_sigmoid(void *restrict tensor);

uint64_t zor_size(void *restrict tensor);

void *zor_embed(void *table, void *toks);

void *zor_softmax(void *restrict tensor, int axis);

void *zor_softmax_backward(void *restrict upstream_grad, void *restrict softmax,
                           int axis);

void *zor_log_softmax(void *restrict tensor, int axis);

void *zor_log_softmax_backward(void *upstream_grad, void *log_softmax,
                               int axis);

void *zor_negative_log_likelihood_loss(void *restrict pred,
                                       void *restrict distr, int axis);

void *
zor_negative_log_likelihood_loss_backward_pred(void *restrict upstream_grad,
                                               void *restrict nlll, int axis);

void *
zor_negative_log_likelihood_loss_backward_distr(void *restrict upstream_grad,
                                                void *restrict nlll, int axis);

void *zor_cross_entropy_loss(void *restrict pred, void *restrict distr,
                             int axis);

void *zor_cross_entropy_backward_pred(void *restrict upstream_grad,
                                      void *restrict x_entropy, int axis);

void *zor_cross_entropy_backward_distr(void *restrict upstream_grad,
                                       void *restrict x_entropy, int axis);

#endif