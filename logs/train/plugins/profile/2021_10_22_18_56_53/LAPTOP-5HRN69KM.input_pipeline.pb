	M?O?"@M?O?"@!M?O?"@	??f???????f?????!??f?????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$M?O?"@+??	h??A????"@Y??&S??*	?????LX@2F
Iterator::Modelx$(~???!%﯃1?I@)??ʡE???1??w??C@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat+??????!R?n?4@)??y?):??1?? ?P2@:Preprocessing2?
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???????!sᮾI?'@)???????1sᮾI?'@:Preprocessing2S
Iterator::Model::ParallelMap?+e?X??!?a?2?t'@)?+e?X??1?a?2?t'@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate?0?*???!?E??ӭ4@)?? ?rh??1????]}!@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip?3??7??!?P|?TH@)??_?Lu?1???u0f@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMapM??St$??!=??lC@7@){?G?zd?1???]}?@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor_?Q?[?!.??7????)_?Q?[?1.??7????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	+??	h??+??	h??!+??	h??      ??!       "      ??!       *      ??!       2	????"@????"@!????"@:      ??!       B      ??!       J	??&S????&S??!??&S??R      ??!       Z	??&S????&S??!??&S??JCPU_ONLY