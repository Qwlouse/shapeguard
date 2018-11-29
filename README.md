# Shape Guard

ShapeGuard is a tool to help with handling shapes in Tensorflow.



## Basic Usage
```python
import tensorflow as tf
from shapeguard import ShapeGuard

sg = ShapeGuard()

img = tf.ones([64, 32, 32, 3])
flat_img = tf.ones([64, 1024])
labels = tf.ones([64])

# check shape consistency
sg.guard(img, "B, H, W, C")
sg.guard(labels, "B, 1")  # raises error because of rank mismatch
sg.guard(flat_img, "B, H*W*C")  # raises error because 1024 != 32*32*3

# guard also returns the tensor, so it can be inlined
mean_img = sg.guard(tf.reduce_mean(img, axis=0), "H, W, C")

# more readable reshapes
flat_img = sg.reshape(img, 'B, H*W*C')

# evaluate templates
assert sg['H, W*C+1'] == [32, 97]

# attribute access to inferred dimensions
assert sg.B == 64

```


## Shape Template Syntax
The shape template mini-DSL supports many different ways of specifying shapes:

  * numbers: `"64, 32, 32, 3"`
  * named dimensions: `"B, width, height2, channels"`
  * wildcards: `"B, *, *, *"`
  * ellipsis: `"B, ..., 3"`
  * addition, subtraction, multiplication, division: `"B*N, W/2, H*(C+1)"`
  * dynamic dimensions: `"?, H, W, C"`  *(only matches `[None, H, W, C]`)*

---
**DISCLAIMER**

This is not an officially supported Google product.

---

