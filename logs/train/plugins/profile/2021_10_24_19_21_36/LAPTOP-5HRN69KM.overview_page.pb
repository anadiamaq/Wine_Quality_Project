?	0?'u$@0?'u$@!0?'u$@	????S8??????S8??!????S8??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$0?'u$@vOjM??AJ+?6$@YEGr????*	??????P@2F
Iterator::Model????????!????B@)??ܵ?|??1^Cy?58@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat???{????!.ȸ ??8@)%u???1?????6@:Preprocessing2?
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceǺ?????!y?5??0@)Ǻ?????1y?5??0@:Preprocessing2S
Iterator::Model::ParallelMap??y?):??!F,?T?*@)??y?):??1F,?T?*@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate???&??!???<@)???Q?~?1?蛣o?&@:Preprocessing2X
!Iterator::Model::ParallelMap::Zipsh??|???!@?L?3O@)"??u??q?1֝VwZ?@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap??A?f??!۶m۶m?@)/n??b?1u?՝Vw
@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor-C??6Z?!?3?τ?@)-C??6Z?1?3?τ?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	vOjM??vOjM??!vOjM??      ??!       "      ??!       *      ??!       2	J+?6$@J+?6$@!J+?6$@:      ??!       B      ??!       J	EGr????EGr????!EGr????R      ??!       Z	EGr????EGr????!EGr????JCPU_ONLY2black"?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: 