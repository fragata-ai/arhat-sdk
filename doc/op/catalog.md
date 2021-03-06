# Operator Catalog

This catalog documents all currently implemented Arhat operators.
Constructors and examples are provided for both low level (Operator)
and high level (Layer) interfaces. For operators that support gradients,
constructors of respective gradient operators are included.
Layer constructors and examples are provided only for operators that
can be represented as layers. Operator constuctors are methods
of the interface `Graph` declared in `"fragata/arhat/graph/api"` package.
Layer constructors are methods of the structure `Fragment` declared in
`"fragata/arhat/front/layers"` package.

- [Abs](./abs.md)
- [Acos](./acos.md)
- [Adadelta](./adadelta.md)
- [Adagrad](./adagrad.md)
- [Adam](./adam.md)
- [Add](./add.md)
- [And](./and.md)
- [ArgMax](./arg_max.md)
- [ArgMin](./arg_min.md)
- [Asin](./asin.md)
- [Atan](./atan.md)
- [AveragePool](./average_pool.md)
- [AveragedLoss](./averaged_loss.md)
- [BitwiseAnd](./bitwise_and.md)
- [BitwiseOr](./bitwise_or.md)
- [BitwiseXor](./bitwise_xor.md)
- [Cbrt](./cbrt.md)
- [Ceil](./ceil.md)
- [ColwiseMax](./colwise_max.md)
- [Concat](./concat.md)
- [ConstantFill](./constant_fill.md)
- [Conv](./conv.md)
- [Cos](./cos.md)
- [Cosh](./cosh.md)
- [Cube](./cube.md)
- [Div](./div.md)
- [Elu](./elu.md)
- [Eq](./eq.md)
- [Erf](./erf.md)
- [Exp](./exp.md)
- [FloatToHalf](./float_to_half.md)
- [Floor](./floor.md)
- [Ftlr](./ftlr.md)
- [FullyConnected](./fully_connected.md)
- [FullyConnectedTranspose](./fully_connected_transpose.md)
- [Ge](./ge.md)
- [Gt](./gt.md)
- [HalfToFloat](./half_to_float.md)
- [HardSigmoid](./hard_sigmoid.md)
- [LabelCrossEntropy](./label_cross_entropy.md)
- [Lars](./lars.md)
- [Le](./le.md)
- [LearningRate](./learning_rate.md)
- [LearningRateAdaption](./learning_rate_adaption.md)
- [Log](./log.md)
- [Logit](./logit.md)
- [Lrn](./lrn.md)
- [Lt](./lt.md)
- [MakeTwoClass](./make_two_class.md)
- [Matmul](./matmul.md)
- [Max](./max.md)
- [MaxPool](./max_pool.md)
- [Min](./min.md)
- [MomentumSgdUpdate](./momentum_sgd_update.md)
- [Mul](./mul.md)
- [Ne](./ne.md)
- [Negative](./negative.md)
- [Not](./not.md)
- [Or](./or.md)
- [PadImage](./pad_image.md)
- [Pow](./pow.md)
- [Reciprocal](./reciprocal.md)
- [ReduceMax](./reduce_max.md)
- [ReduceMean](./reduce_mean.md)
- [ReduceMin](./reduce_min.md)
- [ReduceSum](./reduce_sum.md)
- [Relu](./relu.md)
- [ReluN](./relu_n.md)
- [Reshape](./reshape.md)
- [RowwiseMax](./rowwise_max.md)
- [Rsqrt](./rsqrt.md)
- [Scale](./scale.md)
- [Sigmoid](./sigmoid.md)
- [SigmoidCrossEntropyWithLogits](./sigmoid_cross_entropy_with_logits.md)
- [Sign](./sign.md)
- [Sin](./sin.md)
- [Sinh](./sinh.md)
- [Slice](./slice.md)
- [Softmax](./softmax.md)
- [Softplus](./softplus.md)
- [Softsign](./softsign.md)
- [SpatialBn](./spatial_bn.md)
- [Split](./split.md)
- [Sqr](./sqr.md)
- [Sqrt](./sqrt.md)
- [Sub](./sub.md)
- [Sum](./sum.md)
- [SumElements](./sum_elements.md)
- [SumElementsInt](./sum_elements_int.md)
- [SumSqrElements](./sum_sqr_elements.md)
- [Swish](./swish.md)
- [Tan](./tan.md)
- [Tanh](./tanh.md)
- [Tile](./tile.md)
- [Transpose](./transpose.md)
- [WeightedSigmoidCrossEntropyWithLogits](./weighted_sigmoid_cross_entropy_with_logits.md)
- [WeightedSum](./weighted_sum.md)
- [Xor](./xor.md)


