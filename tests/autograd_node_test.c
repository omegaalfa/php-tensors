#include <assert.h>
#include <stdlib.h>
#include <string.h>

/* The native test exercises the current node API without linking the PHP
 * extension.  Unused autograd sections are discarded by the test compiler. */
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

static int destroyed_count;

void cuda_tensor_destroy(tensor_t *tensor)
{
    if (tensor && tensor->ref_count > 0)
    {
        tensor->ref_count--;
        destroyed_count++;
    }
}

int main(void)
{
    tensor_t parent;
    tensor_t other;
    tensor_t output;
    memset(&parent, 0, sizeof(parent));
    memset(&other, 0, sizeof(other));
    memset(&output, 0, sizeof(output));

    parent.dtype = other.dtype = output.dtype = DTYPE_FLOAT32;
    parent.ref_count = other.ref_count = output.ref_count = 1;
    parent.requires_grad = 1;
    other.requires_grad = 1;

    assert(autograd_record_binary(&output, &parent, &other, OP_ADD));
    assert(output.grad_fn != NULL);
    assert(output.grad_fn->kind == AUTOGRAD_BINARY);
    assert(output.grad_fn->op == OP_ADD);
    assert(output.grad_fn->a == &parent);
    assert(output.grad_fn->b == &other);
    assert(output.grad_fn->a_version == 0);
    assert(output.grad_fn->b_version == 0);
    assert(parent.ref_count == 2);
    assert(other.ref_count == 2);

    autograd_mark_mutated(&parent);
    assert(autograd_tensor_version(&parent) == 1);
    assert(output.grad_fn->a_version != autograd_tensor_version(&parent));

    autograd_release_node(&output);
    assert(output.grad_fn == NULL);
    assert(parent.ref_count == 1);
    assert(other.ref_count == 1);
    assert(destroyed_count == 2);

    /* A shared parent is retained once per edge and released once per edge. */
    output.grad_fn = NULL;
    assert(autograd_record_binary(&output, &parent, &parent, OP_ADD));
    assert(output.grad_fn != NULL);
    assert(parent.ref_count == 3);
    autograd_release_node(&output);
    assert(parent.ref_count == 1);
    assert(destroyed_count == 4);

    return 0;
}
