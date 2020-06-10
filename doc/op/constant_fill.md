
# ConstantFill

Fills the elements of the output tensor with a constant value specified by 
the `value` argument.

- The data type is specified by the `dtype` argument

- Currently, the data types supported are *float*, *int32*, *int64*, and *bool*

- If the `dtype` argument is not provided, the data type of `value` is used

- The output tensor shape is either specified by the `shape` argument or will
  match the shape of the input tensor if one is provided (if an input tensor is
  provided, a `shape` argument should not be set)

- Optional additional dimensions can be appended at the end as specified by
  `extraShape` argument

## Interface

### Input

**input**

>[OPTIONAL] Input tensor to provide shape information

### Output

**output**

>Output tensor of constant values

### Attributes

**shape**

>*(type: []int)* Shape of the output tensor

**extraShape**

>*(type: []int)* Additional dimensions appended at the end of the shape indicated by the input tensor


**dtype**

>*(type: enum)* The data type for the elements of the output tensor. Must be one of the supported types listed above.


**value**

>*(type: dynamic; default: 0.0)* value to populate output tensor with


## Constructors

### Operator


```
ConstantFillOp(
    input *core.Tensor,
    output *core.Tensor,
    shape []int,
    extraShape []int,
    dtype core.Type,
    value interface{}) core.Operator
```


### Layer


```
func(f *Fragment) ConstantFill(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func ConstantFill(g api.Graph, t core.Type, shape []int, value interface{}) {
    output := g.NewTensor(t)
    g.ConstantFillOp(nil, output, shape, nil, t, value)
    g.Return(output)
}
```


### Layer


```
import "fragata/arhat/front/models"

func ConstantFill(m *models.Model, t string, shape []int, value interface{}) {
    output := m.ConstantFill(nil, "dtype", t, "shape", shape, "value", value)
    m.Return(output, 0)
}
```

