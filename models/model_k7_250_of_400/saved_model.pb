На
љн
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02unknown8Љц
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	dє*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	dє*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:є*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:є*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
єє*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
єє*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:є*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:є*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
єЌ*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
єЌ*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ќ*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:Ќ*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ќ1*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	Ќ1*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:1*
dtype0
l
Adagrad/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdagrad/iter
e
 Adagrad/iter/Read/ReadVariableOpReadVariableOpAdagrad/iter*
_output_shapes
: *
dtype0	
n
Adagrad/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdagrad/decay
g
!Adagrad/decay/Read/ReadVariableOpReadVariableOpAdagrad/decay*
_output_shapes
: *
dtype0
~
Adagrad/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdagrad/learning_rate
w
)Adagrad/learning_rate/Read/ReadVariableOpReadVariableOpAdagrad/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

 Adagrad/dense/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	dє*1
shared_name" Adagrad/dense/kernel/accumulator

4Adagrad/dense/kernel/accumulator/Read/ReadVariableOpReadVariableOp Adagrad/dense/kernel/accumulator*
_output_shapes
:	dє*
dtype0

Adagrad/dense/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:є*/
shared_name Adagrad/dense/bias/accumulator

2Adagrad/dense/bias/accumulator/Read/ReadVariableOpReadVariableOpAdagrad/dense/bias/accumulator*
_output_shapes	
:є*
dtype0
Ђ
"Adagrad/dense_1/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
єє*3
shared_name$"Adagrad/dense_1/kernel/accumulator

6Adagrad/dense_1/kernel/accumulator/Read/ReadVariableOpReadVariableOp"Adagrad/dense_1/kernel/accumulator* 
_output_shapes
:
єє*
dtype0

 Adagrad/dense_1/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:є*1
shared_name" Adagrad/dense_1/bias/accumulator

4Adagrad/dense_1/bias/accumulator/Read/ReadVariableOpReadVariableOp Adagrad/dense_1/bias/accumulator*
_output_shapes	
:є*
dtype0
Ђ
"Adagrad/dense_2/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
єЌ*3
shared_name$"Adagrad/dense_2/kernel/accumulator

6Adagrad/dense_2/kernel/accumulator/Read/ReadVariableOpReadVariableOp"Adagrad/dense_2/kernel/accumulator* 
_output_shapes
:
єЌ*
dtype0

 Adagrad/dense_2/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ќ*1
shared_name" Adagrad/dense_2/bias/accumulator

4Adagrad/dense_2/bias/accumulator/Read/ReadVariableOpReadVariableOp Adagrad/dense_2/bias/accumulator*
_output_shapes	
:Ќ*
dtype0
Ё
"Adagrad/dense_3/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ќ1*3
shared_name$"Adagrad/dense_3/kernel/accumulator

6Adagrad/dense_3/kernel/accumulator/Read/ReadVariableOpReadVariableOp"Adagrad/dense_3/kernel/accumulator*
_output_shapes
:	Ќ1*
dtype0

 Adagrad/dense_3/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*1
shared_name" Adagrad/dense_3/bias/accumulator

4Adagrad/dense_3/bias/accumulator/Read/ReadVariableOpReadVariableOp Adagrad/dense_3/bias/accumulator*
_output_shapes
:1*
dtype0

NoOpNoOp
-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Х,
valueЛ,BИ, BБ,
ш
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
І

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
І

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
І

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses*
І

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses*
Ф
.iter
	/decay
0learning_rateaccumulatorPaccumulatorQaccumulatorRaccumulatorSaccumulatorTaccumulatorU&accumulatorV'accumulatorW*
<
0
1
2
3
4
5
&6
'7*
<
0
1
2
3
4
5
&6
'7*
* 
А
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

6serving_default* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*

&0
'1*
* 

Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*
* 
* 
OI
VARIABLE_VALUEAdagrad/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEAdagrad/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdagrad/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

K0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	Ltotal
	Mcount
N	variables
O	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

L0
M1*

N	variables*

VARIABLE_VALUE Adagrad/dense/kernel/accumulator\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdagrad/dense/bias/accumulatorZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adagrad/dense_1/kernel/accumulator\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adagrad/dense_1/bias/accumulatorZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adagrad/dense_2/kernel/accumulator\layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adagrad/dense_2/bias/accumulatorZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adagrad/dense_3/kernel/accumulator\layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adagrad/dense_3/bias/accumulatorZlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
z
serving_default_input_1Placeholder*'
_output_shapes
:џџџџџџџџџd*
dtype0*
shape:џџџџџџџџџd
К
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ1**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_3000173
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ђ	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp Adagrad/iter/Read/ReadVariableOp!Adagrad/decay/Read/ReadVariableOp)Adagrad/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp4Adagrad/dense/kernel/accumulator/Read/ReadVariableOp2Adagrad/dense/bias/accumulator/Read/ReadVariableOp6Adagrad/dense_1/kernel/accumulator/Read/ReadVariableOp4Adagrad/dense_1/bias/accumulator/Read/ReadVariableOp6Adagrad/dense_2/kernel/accumulator/Read/ReadVariableOp4Adagrad/dense_2/bias/accumulator/Read/ReadVariableOp6Adagrad/dense_3/kernel/accumulator/Read/ReadVariableOp4Adagrad/dense_3/bias/accumulator/Read/ReadVariableOpConst*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_3000339
љ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasAdagrad/iterAdagrad/decayAdagrad/learning_ratetotalcount Adagrad/dense/kernel/accumulatorAdagrad/dense/bias/accumulator"Adagrad/dense_1/kernel/accumulator Adagrad/dense_1/bias/accumulator"Adagrad/dense_2/kernel/accumulator Adagrad/dense_2/bias/accumulator"Adagrad/dense_3/kernel/accumulator Adagrad/dense_3/bias/accumulator*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_3000412лњ
+
Л
"__inference__wrapped_model_2999772
input_1B
/sequential_dense_matmul_readvariableop_resource:	dє?
0sequential_dense_biasadd_readvariableop_resource:	єE
1sequential_dense_1_matmul_readvariableop_resource:
єєA
2sequential_dense_1_biasadd_readvariableop_resource:	єE
1sequential_dense_2_matmul_readvariableop_resource:
єЌA
2sequential_dense_2_biasadd_readvariableop_resource:	ЌD
1sequential_dense_3_matmul_readvariableop_resource:	Ќ1@
2sequential_dense_3_biasadd_readvariableop_resource:1
identityЂ'sequential/dense/BiasAdd/ReadVariableOpЂ&sequential/dense/MatMul/ReadVariableOpЂ)sequential/dense_1/BiasAdd/ReadVariableOpЂ(sequential/dense_1/MatMul/ReadVariableOpЂ)sequential/dense_2/BiasAdd/ReadVariableOpЂ(sequential/dense_2/MatMul/ReadVariableOpЂ)sequential/dense_3/BiasAdd/ReadVariableOpЂ(sequential/dense_3/MatMul/ReadVariableOp
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	dє*
dtype0
sequential/dense/MatMulMatMulinput_1.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџє
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:є*
dtype0Њ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєs
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџє
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
єє*
dtype0­
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџє
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:є*
dtype0А
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєw
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџє
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
єЌ*
dtype0Џ
sequential/dense_2/MatMulMatMul%sequential/dense_1/Relu:activations:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype0А
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌw
sequential/dense_2/ReluRelu#sequential/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
(sequential/dense_3/MatMul/ReadVariableOpReadVariableOp1sequential_dense_3_matmul_readvariableop_resource*
_output_shapes
:	Ќ1*
dtype0Ў
sequential/dense_3/MatMulMatMul%sequential/dense_2/Relu:activations:00sequential/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ1
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0Џ
sequential/dense_3/BiasAddBiasAdd#sequential/dense_3/MatMul:product:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ1|
sequential/dense_3/SoftmaxSoftmax#sequential/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ1s
IdentityIdentity$sequential/dense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ1
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp)^sequential/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџd: : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2V
)sequential/dense_3/BiasAdd/ReadVariableOp)sequential/dense_3/BiasAdd/ReadVariableOp2T
(sequential/dense_3/MatMul/ReadVariableOp(sequential/dense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:џџџџџџџџџd
!
_user_specified_name	input_1
Ї

ј
D__inference_dense_2_layer_call_and_return_conditional_losses_2999824

inputs2
matmul_readvariableop_resource:
єЌ.
biasadd_readvariableop_resource:	Ќ
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
єЌ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџє: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџє
 
_user_specified_nameinputs
г	
Х
,__inference_sequential_layer_call_fn_2999867
input_1
unknown:	dє
	unknown_0:	є
	unknown_1:
єє
	unknown_2:	є
	unknown_3:
єЌ
	unknown_4:	Ќ
	unknown_5:	Ќ1
	unknown_6:1
identityЂStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ1**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_2999848o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџd: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџd
!
_user_specified_name	input_1
Є

і
D__inference_dense_3_layer_call_and_return_conditional_losses_2999841

inputs1
matmul_readvariableop_resource:	Ќ1-
biasadd_readvariableop_resource:1
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ќ1*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ1V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ1`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ1w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџЌ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_nameinputs
и
њ
G__inference_sequential_layer_call_and_return_conditional_losses_3000042
input_1 
dense_3000021:	dє
dense_3000023:	є#
dense_1_3000026:
єє
dense_1_3000028:	є#
dense_2_3000031:
єЌ
dense_2_3000033:	Ќ"
dense_3_3000036:	Ќ1
dense_3_3000038:1
identityЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallь
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_3000021dense_3000023*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_2999790
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_3000026dense_1_3000028*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_2999807
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_3000031dense_2_3000033*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџЌ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_2999824
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_3000036dense_3_3000038*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_2999841w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ1Ь
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџd: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџd
!
_user_specified_name	input_1
ѕW
В
#__inference__traced_restore_3000412
file_prefix0
assignvariableop_dense_kernel:	dє,
assignvariableop_1_dense_bias:	є5
!assignvariableop_2_dense_1_kernel:
єє.
assignvariableop_3_dense_1_bias:	є5
!assignvariableop_4_dense_2_kernel:
єЌ.
assignvariableop_5_dense_2_bias:	Ќ4
!assignvariableop_6_dense_3_kernel:	Ќ1-
assignvariableop_7_dense_3_bias:1)
assignvariableop_8_adagrad_iter:	 *
 assignvariableop_9_adagrad_decay: 3
)assignvariableop_10_adagrad_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: G
4assignvariableop_13_adagrad_dense_kernel_accumulator:	dєA
2assignvariableop_14_adagrad_dense_bias_accumulator:	єJ
6assignvariableop_15_adagrad_dense_1_kernel_accumulator:
єєC
4assignvariableop_16_adagrad_dense_1_bias_accumulator:	єJ
6assignvariableop_17_adagrad_dense_2_kernel_accumulator:
єЌC
4assignvariableop_18_adagrad_dense_2_bias_accumulator:	ЌI
6assignvariableop_19_adagrad_dense_3_kernel_accumulator:	Ќ1B
4assignvariableop_20_adagrad_dense_3_bias_accumulator:1
identity_22ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Є
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ъ
valueРBНB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adagrad_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp assignvariableop_9_adagrad_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp)assignvariableop_10_adagrad_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_13AssignVariableOp4assignvariableop_13_adagrad_dense_kernel_accumulatorIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_14AssignVariableOp2assignvariableop_14_adagrad_dense_bias_accumulatorIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_15AssignVariableOp6assignvariableop_15_adagrad_dense_1_kernel_accumulatorIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_16AssignVariableOp4assignvariableop_16_adagrad_dense_1_bias_accumulatorIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adagrad_dense_2_kernel_accumulatorIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adagrad_dense_2_bias_accumulatorIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adagrad_dense_3_kernel_accumulatorIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adagrad_dense_3_bias_accumulatorIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_22IdentityIdentity_21:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_22Identity_22:output:0*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ї

ј
D__inference_dense_2_layer_call_and_return_conditional_losses_3000233

inputs2
matmul_readvariableop_resource:
єЌ.
biasadd_readvariableop_resource:	Ќ
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
єЌ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџє: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџє
 
_user_specified_nameinputs
Ё

ѕ
B__inference_dense_layer_call_and_return_conditional_losses_2999790

inputs1
matmul_readvariableop_resource:	dє.
biasadd_readvariableop_resource:	є
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	dє*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:є*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџєb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџєw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ї

ј
D__inference_dense_1_layer_call_and_return_conditional_losses_3000213

inputs2
matmul_readvariableop_resource:
єє.
biasadd_readvariableop_resource:	є
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
єє*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:є*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџєb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџєw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџє: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџє
 
_user_specified_nameinputs
Ь

)__inference_dense_2_layer_call_fn_3000222

inputs
unknown:
єЌ
	unknown_0:	Ќ
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџЌ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_2999824p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЌ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџє: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџє
 
_user_specified_nameinputs
Ї

ј
D__inference_dense_1_layer_call_and_return_conditional_losses_2999807

inputs2
matmul_readvariableop_resource:
єє.
biasadd_readvariableop_resource:	є
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
єє*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:є*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџєb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџєw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџє: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџє
 
_user_specified_nameinputs
Х

'__inference_dense_layer_call_fn_3000182

inputs
unknown:	dє
	unknown_0:	є
identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_2999790p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџє`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
г	
Х
,__inference_sequential_layer_call_fn_2999994
input_1
unknown:	dє
	unknown_0:	є
	unknown_1:
єє
	unknown_2:	є
	unknown_3:
єЌ
	unknown_4:	Ќ
	unknown_5:	Ќ1
	unknown_6:1
identityЂStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ1**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_2999954o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџd: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџd
!
_user_specified_name	input_1
Ї	
О
%__inference_signature_wrapper_3000173
input_1
unknown:	dє
	unknown_0:	є
	unknown_1:
єє
	unknown_2:	є
	unknown_3:
єЌ
	unknown_4:	Ќ
	unknown_5:	Ќ1
	unknown_6:1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ1**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_2999772o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџd: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџd
!
_user_specified_name	input_1
н#
Џ
G__inference_sequential_layer_call_and_return_conditional_losses_3000118

inputs7
$dense_matmul_readvariableop_resource:	dє4
%dense_biasadd_readvariableop_resource:	є:
&dense_1_matmul_readvariableop_resource:
єє6
'dense_1_biasadd_readvariableop_resource:	є:
&dense_2_matmul_readvariableop_resource:
єЌ6
'dense_2_biasadd_readvariableop_resource:	Ќ9
&dense_3_matmul_readvariableop_resource:	Ќ15
'dense_3_biasadd_readvariableop_resource:1
identityЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpЂdense_3/BiasAdd/ReadVariableOpЂdense_3/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	dє*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџє
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:є*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџє]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџє
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
єє*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџє
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:є*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєa
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџє
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
єЌ*
dtype0
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌa
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	Ќ1*
dtype0
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ1
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ1f
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ1h
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ1Ц
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџd: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
е
љ
G__inference_sequential_layer_call_and_return_conditional_losses_2999954

inputs 
dense_2999933:	dє
dense_2999935:	є#
dense_1_2999938:
єє
dense_1_2999940:	є#
dense_2_2999943:
єЌ
dense_2_2999945:	Ќ"
dense_3_2999948:	Ќ1
dense_3_2999950:1
identityЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallы
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2999933dense_2999935*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_2999790
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2999938dense_1_2999940*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_2999807
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_2999943dense_2_2999945*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџЌ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_2999824
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_2999948dense_3_2999950*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_2999841w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ1Ь
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџd: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ш

)__inference_dense_3_layer_call_fn_3000242

inputs
unknown:	Ќ1
	unknown_0:1
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_2999841o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџЌ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_nameinputs
а	
Ф
,__inference_sequential_layer_call_fn_3000065

inputs
unknown:	dє
	unknown_0:	є
	unknown_1:
єє
	unknown_2:	є
	unknown_3:
єЌ
	unknown_4:	Ќ
	unknown_5:	Ќ1
	unknown_6:1
identityЂStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ1**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_2999848o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџd: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ё

ѕ
B__inference_dense_layer_call_and_return_conditional_losses_3000193

inputs1
matmul_readvariableop_resource:	dє.
biasadd_readvariableop_resource:	є
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	dє*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:є*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџєb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџєw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ь

)__inference_dense_1_layer_call_fn_3000202

inputs
unknown:
єє
	unknown_0:	є
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_2999807p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџє`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџє: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџє
 
_user_specified_nameinputs
а	
Ф
,__inference_sequential_layer_call_fn_3000086

inputs
unknown:	dє
	unknown_0:	є
	unknown_1:
єє
	unknown_2:	є
	unknown_3:
єЌ
	unknown_4:	Ќ
	unknown_5:	Ќ1
	unknown_6:1
identityЂStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ1**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_2999954o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџd: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
е
љ
G__inference_sequential_layer_call_and_return_conditional_losses_2999848

inputs 
dense_2999791:	dє
dense_2999793:	є#
dense_1_2999808:
єє
dense_1_2999810:	є#
dense_2_2999825:
єЌ
dense_2_2999827:	Ќ"
dense_3_2999842:	Ќ1
dense_3_2999844:1
identityЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallы
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2999791dense_2999793*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_2999790
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2999808dense_1_2999810*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_2999807
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_2999825dense_2_2999827*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџЌ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_2999824
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_2999842dense_3_2999844*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_2999841w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ1Ь
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџd: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ч4
Т	
 __inference__traced_save_3000339
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop+
'savev2_adagrad_iter_read_readvariableop	,
(savev2_adagrad_decay_read_readvariableop4
0savev2_adagrad_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop?
;savev2_adagrad_dense_kernel_accumulator_read_readvariableop=
9savev2_adagrad_dense_bias_accumulator_read_readvariableopA
=savev2_adagrad_dense_1_kernel_accumulator_read_readvariableop?
;savev2_adagrad_dense_1_bias_accumulator_read_readvariableopA
=savev2_adagrad_dense_2_kernel_accumulator_read_readvariableop?
;savev2_adagrad_dense_2_bias_accumulator_read_readvariableopA
=savev2_adagrad_dense_3_kernel_accumulator_read_readvariableop?
;savev2_adagrad_dense_3_bias_accumulator_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ё
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ъ
valueРBНB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B Ц	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop'savev2_adagrad_iter_read_readvariableop(savev2_adagrad_decay_read_readvariableop0savev2_adagrad_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop;savev2_adagrad_dense_kernel_accumulator_read_readvariableop9savev2_adagrad_dense_bias_accumulator_read_readvariableop=savev2_adagrad_dense_1_kernel_accumulator_read_readvariableop;savev2_adagrad_dense_1_bias_accumulator_read_readvariableop=savev2_adagrad_dense_2_kernel_accumulator_read_readvariableop;savev2_adagrad_dense_2_bias_accumulator_read_readvariableop=savev2_adagrad_dense_3_kernel_accumulator_read_readvariableop;savev2_adagrad_dense_3_bias_accumulator_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Е
_input_shapesЃ
 : :	dє:є:
єє:є:
єЌ:Ќ:	Ќ1:1: : : : : :	dє:є:
єє:є:
єЌ:Ќ:	Ќ1:1: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	dє:!

_output_shapes	
:є:&"
 
_output_shapes
:
єє:!

_output_shapes	
:є:&"
 
_output_shapes
:
єЌ:!

_output_shapes	
:Ќ:%!

_output_shapes
:	Ќ1: 

_output_shapes
:1:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	dє:!

_output_shapes	
:є:&"
 
_output_shapes
:
єє:!

_output_shapes	
:є:&"
 
_output_shapes
:
єЌ:!

_output_shapes	
:Ќ:%!

_output_shapes
:	Ќ1: 

_output_shapes
:1:

_output_shapes
: 
и
њ
G__inference_sequential_layer_call_and_return_conditional_losses_3000018
input_1 
dense_2999997:	dє
dense_2999999:	є#
dense_1_3000002:
єє
dense_1_3000004:	є#
dense_2_3000007:
єЌ
dense_2_3000009:	Ќ"
dense_3_3000012:	Ќ1
dense_3_3000014:1
identityЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallь
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_2999997dense_2999999*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_2999790
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_3000002dense_1_3000004*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_2999807
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_3000007dense_2_3000009*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџЌ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_2999824
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_3000012dense_3_3000014*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_2999841w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ1Ь
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџd: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџd
!
_user_specified_name	input_1
Є

і
D__inference_dense_3_layer_call_and_return_conditional_losses_3000253

inputs1
matmul_readvariableop_resource:	Ќ1-
biasadd_readvariableop_resource:1
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ќ1*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ1V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ1`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ1w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџЌ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_nameinputs
н#
Џ
G__inference_sequential_layer_call_and_return_conditional_losses_3000150

inputs7
$dense_matmul_readvariableop_resource:	dє4
%dense_biasadd_readvariableop_resource:	є:
&dense_1_matmul_readvariableop_resource:
єє6
'dense_1_biasadd_readvariableop_resource:	є:
&dense_2_matmul_readvariableop_resource:
єЌ6
'dense_2_biasadd_readvariableop_resource:	Ќ9
&dense_3_matmul_readvariableop_resource:	Ќ15
'dense_3_biasadd_readvariableop_resource:1
identityЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpЂdense_3/BiasAdd/ReadVariableOpЂdense_3/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	dє*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџє
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:є*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџє]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџє
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
єє*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџє
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:є*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєa
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџє
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
єЌ*
dtype0
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌa
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	Ќ1*
dtype0
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ1
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ1f
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ1h
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ1Ц
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџd: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs"лL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Њ
serving_default
;
input_10
serving_default_input_1:0џџџџџџџџџd;
dense_30
StatefulPartitionedCall:0џџџџџџџџџ1tensorflow/serving/predict:ЕY

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Л

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Л

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Л

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
Л

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
г
.iter
	/decay
0learning_rateaccumulatorPaccumulatorQaccumulatorRaccumulatorSaccumulatorTaccumulatorU&accumulatorV'accumulatorW"
	optimizer
X
0
1
2
3
4
5
&6
'7"
trackable_list_wrapper
X
0
1
2
3
4
5
&6
'7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ў2ћ
,__inference_sequential_layer_call_fn_2999867
,__inference_sequential_layer_call_fn_3000065
,__inference_sequential_layer_call_fn_3000086
,__inference_sequential_layer_call_fn_2999994Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ъ2ч
G__inference_sequential_layer_call_and_return_conditional_losses_3000118
G__inference_sequential_layer_call_and_return_conditional_losses_3000150
G__inference_sequential_layer_call_and_return_conditional_losses_3000018
G__inference_sequential_layer_call_and_return_conditional_losses_3000042Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЭBЪ
"__inference__wrapped_model_2999772input_1"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
,
6serving_default"
signature_map
:	dє2dense/kernel
:є2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
б2Ю
'__inference_dense_layer_call_fn_3000182Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_dense_layer_call_and_return_conditional_losses_3000193Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
": 
єє2dense_1/kernel
:є2dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
г2а
)__inference_dense_1_layer_call_fn_3000202Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_dense_1_layer_call_and_return_conditional_losses_3000213Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
": 
єЌ2dense_2/kernel
:Ќ2dense_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
г2а
)__inference_dense_2_layer_call_fn_3000222Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_dense_2_layer_call_and_return_conditional_losses_3000233Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
!:	Ќ12dense_3/kernel
:12dense_3/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
г2а
)__inference_dense_3_layer_call_fn_3000242Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_dense_3_layer_call_and_return_conditional_losses_3000253Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
:	 (2Adagrad/iter
: (2Adagrad/decay
: (2Adagrad/learning_rate
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
K0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЬBЩ
%__inference_signature_wrapper_3000173input_1"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ltotal
	Mcount
N	variables
O	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
L0
M1"
trackable_list_wrapper
-
N	variables"
_generic_user_object
1:/	dє2 Adagrad/dense/kernel/accumulator
+:)є2Adagrad/dense/bias/accumulator
4:2
єє2"Adagrad/dense_1/kernel/accumulator
-:+є2 Adagrad/dense_1/bias/accumulator
4:2
єЌ2"Adagrad/dense_2/kernel/accumulator
-:+Ќ2 Adagrad/dense_2/bias/accumulator
3:1	Ќ12"Adagrad/dense_3/kernel/accumulator
,:*12 Adagrad/dense_3/bias/accumulator
"__inference__wrapped_model_2999772o&'0Ђ-
&Ђ#
!
input_1џџџџџџџџџd
Њ "1Њ.
,
dense_3!
dense_3џџџџџџџџџ1І
D__inference_dense_1_layer_call_and_return_conditional_losses_3000213^0Ђ-
&Ђ#
!
inputsџџџџџџџџџє
Њ "&Ђ#

0џџџџџџџџџє
 ~
)__inference_dense_1_layer_call_fn_3000202Q0Ђ-
&Ђ#
!
inputsџџџџџџџџџє
Њ "џџџџџџџџџєІ
D__inference_dense_2_layer_call_and_return_conditional_losses_3000233^0Ђ-
&Ђ#
!
inputsџџџџџџџџџє
Њ "&Ђ#

0џџџџџџџџџЌ
 ~
)__inference_dense_2_layer_call_fn_3000222Q0Ђ-
&Ђ#
!
inputsџџџџџџџџџє
Њ "џџџџџџџџџЌЅ
D__inference_dense_3_layer_call_and_return_conditional_losses_3000253]&'0Ђ-
&Ђ#
!
inputsџџџџџџџџџЌ
Њ "%Ђ"

0џџџџџџџџџ1
 }
)__inference_dense_3_layer_call_fn_3000242P&'0Ђ-
&Ђ#
!
inputsџџџџџџџџџЌ
Њ "џџџџџџџџџ1Ѓ
B__inference_dense_layer_call_and_return_conditional_losses_3000193]/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "&Ђ#

0џџџџџџџџџє
 {
'__inference_dense_layer_call_fn_3000182P/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџєЖ
G__inference_sequential_layer_call_and_return_conditional_losses_3000018k&'8Ђ5
.Ђ+
!
input_1џџџџџџџџџd
p 

 
Њ "%Ђ"

0џџџџџџџџџ1
 Ж
G__inference_sequential_layer_call_and_return_conditional_losses_3000042k&'8Ђ5
.Ђ+
!
input_1џџџџџџџџџd
p

 
Њ "%Ђ"

0џџџџџџџџџ1
 Е
G__inference_sequential_layer_call_and_return_conditional_losses_3000118j&'7Ђ4
-Ђ*
 
inputsџџџџџџџџџd
p 

 
Њ "%Ђ"

0џџџџџџџџџ1
 Е
G__inference_sequential_layer_call_and_return_conditional_losses_3000150j&'7Ђ4
-Ђ*
 
inputsџџџџџџџџџd
p

 
Њ "%Ђ"

0џџџџџџџџџ1
 
,__inference_sequential_layer_call_fn_2999867^&'8Ђ5
.Ђ+
!
input_1џџџџџџџџџd
p 

 
Њ "џџџџџџџџџ1
,__inference_sequential_layer_call_fn_2999994^&'8Ђ5
.Ђ+
!
input_1џџџџџџџџџd
p

 
Њ "џџџџџџџџџ1
,__inference_sequential_layer_call_fn_3000065]&'7Ђ4
-Ђ*
 
inputsџџџџџџџџџd
p 

 
Њ "џџџџџџџџџ1
,__inference_sequential_layer_call_fn_3000086]&'7Ђ4
-Ђ*
 
inputsџџџџџџџџџd
p

 
Њ "џџџџџџџџџ1Ѓ
%__inference_signature_wrapper_3000173z&';Ђ8
Ђ 
1Њ.
,
input_1!
input_1џџџџџџџџџd"1Њ.
,
dense_3!
dense_3џџџџџџџџџ1