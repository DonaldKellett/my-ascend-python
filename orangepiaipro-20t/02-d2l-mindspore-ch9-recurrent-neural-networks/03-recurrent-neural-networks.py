import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

from mindspore import dtype as mstype

def main():
    mindspore.set_device(device_target='Ascend', device_id=0)

    X, W_xh = ops.randn(3, 1, dtype=mstype.float16), ops.randn(1, 4, dtype=mstype.float16)
    H, W_hh = ops.randn(3, 4, dtype=mstype.float16), ops.randn(4, 4, dtype=mstype.float16)
    sum_prod = ops.matmul(X, W_xh) + ops.matmul(H, W_hh)
    prod_cat = ops.matmul(ops.cat((X, H), axis=1), ops.cat((W_xh, W_hh), axis=0))
    loss_fn = nn.MSELoss()
    loss = loss_fn(sum_prod, prod_cat).item()
    print(f'Sum of products: {sum_prod}')
    print(f'Product of concatenation: {prod_cat}')
    print(f'Results {'match' if loss < 1e-3 else 'differ'}!')

if __name__ == '__main__':
    main()
