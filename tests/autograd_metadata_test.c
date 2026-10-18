#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <php.h>
#include "../src/autograd.h"
#undef ecalloc
#undef emalloc
#undef erealloc
#undef efree
#define ecalloc(count, size) calloc((count), (size))
#define emalloc(size) malloc(size)
#define erealloc(ptr, size) realloc((ptr), (size))
#define efree(ptr) free(ptr)
#include "../src/autograd.c"

void cuda_tensor_destroy(tensor_t *tensor)
{
    if (tensor && tensor->ref_count > 0)
        tensor->ref_count--;
}

int main(void)
{
    tensor_t root;
    tensor_t view;
    memset(&root, 0, sizeof(root));
    memset(&view, 0, sizeof(view));
    view.base_tensor = &root;

    assert(root.grad_fn == NULL);
    assert(root.grad == NULL);
    assert(root.requires_grad == 0);
    assert(root.autograd_consumed == 0);
    assert(autograd_tensor_version(&root) == 0);
    assert(autograd_tensor_version(&view) == 0);

    autograd_mark_mutated(&view);
    assert(autograd_tensor_version(&root) == 1);
    assert(autograd_tensor_version(&view) == 1);

    autograd_mark_mutated(&root);
    assert(autograd_tensor_version(&view) == 2);

    tensor_t gradient;
    memset(&gradient, 0, sizeof(gradient));
    gradient.ref_count = 1;
    root.grad = &gradient;
    autograd_clear_gradient(&root);
    assert(root.grad == NULL);

    return 0;
}
