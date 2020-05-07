## pre process call stack

```

```

```cpp
//in gstnvinfer.cpp
/* Helper function to queue a batch for inferencing and push it to the element's
 * processing queue. */
static gpointer
gst_nvinfer_input_queue_loop (gpointer data) {
...
  DsNvInferImpl *impl = DS_NVINFER_IMPL (nvinfer);
...
  NvDsInferContextPtr nvdsinfer_ctx = impl->m_InferCtx;
}

//in gstnvinfer_impl.h
using NvDsInferContextPtr = std::shared_ptr<INvDsInferContext>;
```
## post process call stack
```
gst_nvinfer_output_loop -> NvDsInferContextImpl::dequeueOutputBatch(NvDsInferContextBatchOutput &batchOutput)
                                                -> InferPostprocessor::postProcessHost
                                                -> DetectPostprocessor::parseEachBatch
                                                -> DetectPostprocessor::fillDetectionOutput
                                                -> custom parser function or built-in parseBoundingBox                      
                        ->attach_metadata_detector
```
```cpp
gstnvinfer.cpp which reside in nvinfer plugin
/**
 * Output loop used to pop output from inference, attach the output to the
 * buffer in form of NvDsMeta and push the buffer to downstream element.
 */
static gpointer
gst_nvinfer_output_loop (gpointer data)
{
...
    //get the NvDsInferContextPtr wchich is std::shared_ptr<INvDsInferContext>
    NvDsInferContextPtr nvdsinfer_ctx = impl->m_InferCtx;

    /* Create and initialize the object for managing the usage of batch_output. */
    auto tensor_deleter = [] (GstNvInferTensorOutputObject *o) {
      if (o)
        gst_mini_object_unref (GST_MINI_OBJECT (o));
    };
    std::unique_ptr<GstNvInferTensorOutputObject, decltype(tensor_deleter)>
        tensor_out_object (new GstNvInferTensorOutputObject, tensor_deleter);
    gst_mini_object_init (GST_MINI_OBJECT (tensor_out_object.get()), 0, G_TYPE_POINTER, NULL,
        NULL, gst_nvinfer_tensoroutput_free);
    tensor_out_object->infer_context = nvdsinfer_ctx;

    batch_output = &tensor_out_object->batch_output;
    /* Dequeue inferencing output from NvDsInferContext */
    status = nvdsinfer_ctx->dequeueOutputBatch (*batch_output);
    
    ...
    
        /* For each frame attach metadata output. */
    for (guint i = 0; i < batch->frames.size (); i++) {
      GstNvInferFrame & frame = batch->frames[i];
      NvDsInferFrameOutput &frame_output = batch_output->frames[i];
      auto obj_history = frame.history.lock ();

      /* If we have an object's history and the buffer PTS is same as last
       * inferred PTS mark the object as not being inferred. This check could be
       * useful if object is inferred multiple times before completion of an
       * existing inference. */
      if (obj_history != nullptr) {
        if (obj_history->last_inferred_frame_num == frame.frame_num)
          obj_history->under_inference = FALSE;
      }
        //after post process then start attach the meta data
      if (IS_DETECTOR_INSTANCE (nvinfer)) {
        attach_metadata_detector (nvinfer, GST_MINI_OBJECT (tensor_out_object.get()),
                frame, frame_output.detectionOutput);
      } else if (IS_CLASSIFIER_INSTANCE (nvinfer)) {
        NvDsInferClassificationOutput &classification_output = frame_output.classificationOutput;
        GstNvInferObjectInfo new_info;
        new_info.attributes.assign(classification_output.attributes,
            classification_output.attributes + classification_output.numAttributes);
        new_info.label.assign(classification_output.label);

        /* Object history is available merge the old and new classification
         * results. */
        if (obj_history != nullptr) {
          merge_classification_output (*obj_history, new_info);
        }

        /* Use the merged classification results if available otherwise use
         * the new results. */
        auto &  info = (obj_history) ? obj_history->cached_info : new_info;

        /* Attach metadata only if not operating in async mode. In async mode,
         * the GstBuffer and the associated metadata are not valid here, since
         * the buffer is already pushed downstream. The metadata will be updated
         * in the input thread. */
        if (nvinfer->classifier_async_mode == FALSE) {
          attach_metadata_classifier (nvinfer, GST_MINI_OBJECT (tensor_out_object.get()),
                  frame, info);
        }
      } else if (IS_SEGMENTATION_INSTANCE (nvinfer)) {
        attach_metadata_segmentation (nvinfer, GST_MINI_OBJECT (tensor_out_object.get()),
            frame, frame_output.segmentationOutput);
      }
    }
}
```

```cpp
nvdsinfer_context_impl.cpp reside in nvdsinfer lib
/**
 * Implementation of the INvDsInferContext interface.
 */
class NvDsInferContextImpl : public INvDsInferContext {
...
    std::unique_ptr<InferPreprocessor> m_Preprocessor;
    std::unique_ptr<InferPostprocessor> m_Postprocessor;
...
}

/* Dequeue batch output of the inference engine for each batch input. */
NvDsInferStatus
NvDsInferContextImpl::dequeueOutputBatch(NvDsInferContextBatchOutput &batchOutput)
{
  ...
    assert(m_Postprocessor);
    /* Fill the host buffers information in the output. */
    RETURN_NVINFER_ERROR(
        //
        m_Postprocessor->postProcessHost(*recyleBatch, batchOutput),
        "postprocessing host buffers failed.");
    ...
}

NvDsInferStatus
InferPostprocessor::postProcessHost(NvDsInferBatch& batch,
        NvDsInferContextBatchOutput& batchOutput)
{
    batchOutput.frames = new NvDsInferFrameOutput[batch.m_BatchSize];
    batchOutput.numFrames = batch.m_BatchSize;

    /* For each frame in the current batch, parse the output and add the frame
     * output to the batch output. The number of frames output in one batch
     * will be equal to the number of frames present in the batch during queuing
     * at the input.
     */
    for (unsigned int index = 0; index < batch.m_BatchSize; index++)
    {
        NvDsInferFrameOutput& frameOutput = batchOutput.frames[index];
        frameOutput.outputType = NvDsInferNetworkType_Other;

        /* Calculate the pointer to the output for each frame in the batch for
         * each output layer buffer. The NvDsInferLayerInfo vector for output
         * layers is passed to the output parsing function. */
        for (unsigned int i = 0; i < m_OutputLayerInfo.size(); i++)
        {
            NvDsInferLayerInfo& info = m_OutputLayerInfo[i];
            info.buffer =
                (void*)(batch.m_HostBuffers[info.bindingIndex]->ptr<uint8_t>() +
                        info.inferDims.numElements *
                            getElementSize(info.dataType) * index);
        }
        //call here
        RETURN_NVINFER_ERROR(parseEachBatch(m_OutputLayerInfo, frameOutput),
            "Infer context initialize inference info failed");
    }
    ...
}

//If this is a detect network, then initialize a subclass 
/** Implementation of post-processing class for object detection networks. */
class DetectPostprocessor : public InferPostprocessor

NvDsInferStatus
DetectPostprocessor::parseEachBatch(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsInferFrameOutput& result)
{
    result.outputType = NvDsInferNetworkType_Detector;
    fillDetectionOutput(outputLayers, result.detectionOutput);
    return NVDSINFER_SUCCESS;
}


```

```cpp
//nvdsinfer_context_impl_output_parsing.cpp reside in nvdsinfer lib
//Note: some implementation of DetectPostprocessor are in nvdsinfer_context_impl.cpp 
NvDsInferStatus
DetectPostprocessor::fillDetectionOutput(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsInferDetectionOutput& output)
{
    /* Clear the object lists. */
    m_ObjectList.clear();

    /* Call custom parsing function if specified otherwise use the one
     * written along with this implementation. */
    if (m_CustomBBoxParseFunc)
    {
        if (!m_CustomBBoxParseFunc(outputLayers, m_NetworkInfo,
                    m_DetectionParams, m_ObjectList))
        {
            printError("Failed to parse bboxes using custom parse function");
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
    }
    else
    {
        if (!parseBoundingBox(outputLayers, m_NetworkInfo,
                    m_DetectionParams, m_ObjectList))
        {
            printError("Failed to parse bboxes");
            return NVDSINFER_OUTPUT_PARSING_FAILED;
        }
    }

    filterDetectionOutput(m_DetectionParams, m_ObjectList);

    switch (m_ClusterMode)
    {
        case NVDSINFER_CLUSTER_NMS:
            clusterAndFillDetectionOutputNMS(output);
            break;

        case NVDSINFER_CLUSTER_DBSCAN:
            clusterAndFillDetectionOutputDBSCAN(output);
            break;

        case NVDSINFER_CLUSTER_GROUP_RECTANGLES:
            clusterAndFillDetectionOutputCV(output);
            break;

        case NVDSINFER_CLUSTER_NONE:
            fillUnclusteredOutput(output);
            break;

        default:
            break;
    }

    return NVDSINFER_SUCCESS;
}
```

```cpp
//gstnvinfer_meta_utils.cpp reside in nvinfer plugin
/**
 * Attach metadata for the detector. We will be adding a new metadata.
 */
void
attach_metadata_detector (GstNvInfer * nvinfer, GstMiniObject * tensor_out_object,
    GstNvInferFrame & frame, NvDsInferDetectionOutput & detection_output)
{
  static gchar font_name[] = "Serif";
  NvDsObjectMeta *obj_meta = NULL;
  NvDsObjectMeta *parent_obj_meta = frame.obj_meta; /* This will be  NULL in case of primary detector */
  NvDsFrameMeta *frame_meta = frame.frame_meta;
  NvDsBatchMeta *batch_meta = frame_meta->base_meta.batch_meta;
  nvds_acquire_meta_lock (batch_meta);

  frame_meta->bInferDone = TRUE;
  /* Iterate through the inference output for one frame and attach the detected
   * bnounding boxes. */
  for (guint i = 0; i < detection_output.numObjects; i++) {
    NvDsInferObject & obj = detection_output.objects[i];
    GstNvInferDetectionFilterParams & filter_params =
        (*nvinfer->perClassDetectionFilterParams)[obj.classIndex];

    /* Scale the bounding boxes proportionally based on how the object/frame was
     * scaled during input. */
    obj.left /= frame.scale_ratio_x;
    obj.top /= frame.scale_ratio_y;
    obj.width /= frame.scale_ratio_x;
    obj.height /= frame.scale_ratio_y;

    /* Check if the scaled box co-ordinates meet the detection filter criteria.
     * Skip the box if it does not. */
    if(nvinfer->filter_out_class_ids->find(obj.classIndex) != nvinfer->filter_out_class_ids->end())
        continue;
    if (obj.width < filter_params.detectionMinWidth)
      continue;
    if (obj.height < filter_params.detectionMinHeight)
      continue;
    if (filter_params.detectionMaxWidth > 0 &&
        obj.width > filter_params.detectionMaxWidth)
      continue;
    if (filter_params.detectionMaxHeight > 0 &&
        obj.height > filter_params.detectionMaxHeight)
      continue;
    if (obj.top < filter_params.roiTopOffset)
      continue;
    if (obj.top + obj.height >
        (frame.input_surf_params->height - filter_params.roiBottomOffset))
      continue;
    obj_meta = nvds_acquire_obj_meta_from_pool (batch_meta);

    obj_meta->unique_component_id = nvinfer->unique_id;
    obj_meta->confidence = obj.confidence;

    /* This is an untracked object. Set tracking_id to -1. */
    obj_meta->object_id = UNTRACKED_OBJECT_ID;
    obj_meta->class_id = obj.classIndex;

    NvOSD_RectParams & rect_params = obj_meta->rect_params;
    NvOSD_TextParams & text_params = obj_meta->text_params;

    /* Assign bounding box coordinates. */
    rect_params.left = obj.left;
    rect_params.top = obj.top;
    rect_params.width = obj.width;
    rect_params.height = obj.height;

    if(!nvinfer->process_full_frame) {
      rect_params.left += parent_obj_meta->rect_params.left;
      rect_params.top += parent_obj_meta->rect_params.top;
    }

    /* Border of width 3. */
    rect_params.border_width = 3;
    if (obj.classIndex > (gint) nvinfer->perClassColorParams->size()) {
      rect_params.has_bg_color = 0;
      rect_params.border_color = (NvOSD_ColorParams) {1, 0, 0, 1};
    } else {
      GstNvInferColorParams &color_params =
          (*nvinfer->perClassColorParams)[obj.classIndex];
      rect_params.has_bg_color = color_params.have_bg_color;
      rect_params.bg_color = color_params.bg_color;
      rect_params.border_color = color_params.border_color;
    }

    if (obj.label)
      strncpy (obj_meta->obj_label, obj.label, MAX_LABEL_SIZE);
    /* display_text requires heap allocated memory. */
    text_params.display_text = g_strdup (obj.label);
    /* Display text above the left top corner of the object. */
    text_params.x_offset = rect_params.left;
    text_params.y_offset = rect_params.top - 10;
    /* Set black background for the text. */
    text_params.set_bg_clr = 1;
    text_params.text_bg_clr = (NvOSD_ColorParams) {
    0, 0, 0, 1};
    /* Font face, size and color. */
    text_params.font_params.font_name = font_name;
    text_params.font_params.font_size = 11;
    text_params.font_params.font_color = (NvOSD_ColorParams) {
    1, 1, 1, 1};
    nvds_add_obj_meta_to_frame (frame_meta, obj_meta, parent_obj_meta);
  }
  nvds_release_meta_lock (batch_meta);
}
```
