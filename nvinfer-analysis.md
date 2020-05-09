## pre process call stack

```
gst_nvinfer_start -> createNvDsInferContext -> new NvDsInferContextImpl()
                                               -> ctx->initialize -> NvDsInferContextImpl::initInferenceInfo 
                                               //(initialize  m_OutputLayerInfo)
                                                                  -> NvDsInferContextImpl::preparePreprocess
                                                                  -> NvDsInferContextImpl::preparePostprocess
```

```cpp
//in gstnvinfer.cpp
/**
 * Initialize all resources and start the output thread
 */
static gboolean
gst_nvinfer_start (GstBaseTransform * btrans)
{
  GstNvInfer *nvinfer = GST_NVINFER (btrans);
  GstAllocationParams allocation_params;
  cudaError_t cudaReturn;
  NvBufSurfaceColorFormat color_format;
  NvDsInferStatus status;
  std::string nvtx_str;
  DsNvInferImpl *impl = DS_NVINFER_IMPL (nvinfer);
  NvDsInferContextHandle infer_context = nullptr;

 ...
  /* Create the NvDsInferContext instance. */
  status =
      createNvDsInferContext (&infer_context, *init_params,
      nvinfer, gst_nvinfer_logger);
  ...
}

//in nvdsinfer_context.h
/** An opaque pointer type to be used as a handle for a context instance. */
typedef struct INvDsInferContext * NvDsInferContextHandle;

//in nvdsinfer_context_impl.cpp
/*
 * Factory function to create an NvDsInferContext instance and initialize it with
 * supplied parameters.
 */
NvDsInferStatus
createNvDsInferContext(NvDsInferContextHandle *handle,
        NvDsInferContextInitParams &initParams, void *userCtx,
        NvDsInferContextLoggingFunc logFunc)
{
    NvDsInferStatus status;
    NvDsInferContextImpl *ctx = new NvDsInferContextImpl();

    status = ctx->initialize(initParams, userCtx, logFunc);
    if (status == NVDSINFER_SUCCESS)
    {
        *handle = ctx; // 
    }
    else
    {
        static_cast<INvDsInferContext *>(ctx)->destroy();
    }
    return status;
}

//in nvdsinfer_context_impl.h
/**
 * Implementation of the INvDsInferContext interface.
 */
class NvDsInferContextImpl : public INvDsInferContext
{
public:
    /**
     * Default constructor.
     */
    NvDsInferContextImpl();

    /**
     * Initializes the Infer engine, allocates layer buffers and other required
     * initialization steps.
     */
    NvDsInferStatus initialize(NvDsInferContextInitParams &initParams,
            void *userCtx, NvDsInferContextLoggingFunc logFunc);
...
}

//in gstnvinfer.cpp
/* Helper function to queue a batch for inferencing and push it to the element's
 * processing queue. */
static gpointer
gst_nvinfer_input_queue_loop (gpointer data) {
...
  DsNvInferImpl *impl = DS_NVINFER_IMPL (nvinfer);
...
  NvDsInferContextPtr nvdsinfer_ctx = impl->m_InferCtx;
...
  status = nvdsinfer_ctx->queueInputBatch (input_batch);
}

//in gstnvinfer_impl.h
using NvDsInferContextPtr = std::shared_ptr<INvDsInferContext>;

class DsNvInferImpl
{
public:
  using ContextReplacementPtr =
      std::unique_ptr<std::tuple<NvDsInferContextPtr, NvDsInferContextInitParamsPtr, std::string>>;

  DsNvInferImpl (GstNvInfer *infer);
  ~DsNvInferImpl ();
  /* Start the model load thread. */
  NvDsInferStatus start ();
  /* Stop the model load thread. Release the NvDsInferContext. */
  void stop ();

  bool isContextReady () const { return m_InferCtx.get(); }

  /** Load new model in separate thread */
  bool triggerNewModel (const std::string &modelPath, ModelLoadType loadType);

  /** replace context, action in submit_input_buffer */
  NvDsInferStatus ensureReplaceNextContext ();
  void notifyLoadModelStatus (const ModelStatus &res);

  /** NvDsInferContext to be used for inferencing. */
  NvDsInferContextPtr m_InferCtx;

  /** NvDsInferContext initialization params. */
  NvDsInferContextInitParamsPtr m_InitParams;
...
}

//in gstnvinfer.cpp
//Create a instance of DsNvInferImpl
static void
gst_nvinfer_init (GstNvInfer * nvinfer)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (nvinfer);

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  // little trick to use pointer and reinterpret_cast
  nvinfer->impl = reinterpret_cast<GstNvInferImpl*>(new DsNvInferImpl(nvinfer));
  ...
}

//in gstnvinfer_impl.cpp
//call NvDsInferContext_ResetInitParams
DsNvInferImpl::DsNvInferImpl (GstNvInfer * infer)
  : m_InitParams (new NvDsInferContextInitParams),
    m_GstInfer (infer)
{
  NvDsInferContext_ResetInitParams (m_InitParams.get ());
}

//in gstnvinfer_impl.cpp
/*
 * Reset the members inside the initParams structure to default values.
 */
void
NvDsInferContext_ResetInitParams (NvDsInferContextInitParams *initParams)
{
    if (initParams == nullptr)
    {
        fprintf(stderr, "Warning. NULL initParams passed to "
                "NvDsInferContext_ResetInitParams()\n");
        return;
    }

    memset(initParams, 0, sizeof (*initParams));

    initParams->networkMode = NvDsInferNetworkMode_FP32;
    initParams->networkInputFormat = NvDsInferFormat_Unknown;
    initParams->uffInputOrder = NvDsInferTensorOrder_kNCHW;
    initParams->maxBatchSize = 1;
    initParams->networkScaleFactor = 1.0;
    initParams->networkType = NvDsInferNetworkType_Detector;
    initParams->outputBufferPoolSize = NVDSINFER_MIN_OUTPUT_BUFFERPOOL_SIZE;
}

```
```cpp
// in nvdsinfer_context_impl.cpp
/* The function performs all the initialization steps required by the inference
 * engine. */
NvDsInferStatus
NvDsInferContextImpl::initialize(NvDsInferContextInitParams& initParams,
        void* userCtx, NvDsInferContextLoggingFunc logFunc)
{
    ...
    
    /* Load the custom library if specified. */
    if (!string_empty(initParams.customLibPath))
    {
        std::unique_ptr<DlLibHandle> dlHandle =
            std::make_unique<DlLibHandle>(initParams.customLibPath, RTLD_LAZY);
        if (!dlHandle->isValid())
        {
            printError("Could not open custom lib: %s", dlerror());
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
        m_CustomLibHandle = std::move(dlHandle); //dlHaddle is std::unique_ptr type
    }

    m_BackendContext = generateBackendContext(initParams); //create BackendContext
    if (!m_BackendContext)
    {
        printError("generate backend failed, check config file settings");
        return NVDSINFER_CONFIG_FAILED;
    }

    RETURN_NVINFER_ERROR(initInferenceInfo(initParams, *m_BackendContext),
        "Infer context initialize inference info failed");
    assert(m_AllLayerInfo.size());

    RETURN_NVINFER_ERROR(preparePreprocess(initParams),
        "Infer Context prepare preprocessing resource failed.");
    assert(m_Preprocessor);

    RETURN_NVINFER_ERROR(preparePostprocess(initParams),
        "Infer Context prepare postprocessing resource failed.");
    assert(m_Postprocessor);

    /* Allocate binding buffers on the device and the corresponding host
     * buffers. */
    NvDsInferStatus status = allocateBuffers();
    if (status != NVDSINFER_SUCCESS)
    {
        printError("Failed to allocate buffers");
        return status;
    }

    /* If there are more than one input layers (non-image input) and custom
     * library is specified, try to initialize these layers. */
    if (m_InputDeviceBuffers.size() > 1)
    {
        NvDsInferStatus status = initNonImageInputLayers();
        if (status != NVDSINFER_SUCCESS)
        {
            printError("Failed to initialize non-image input layers");
            return status;
        }
    }

    m_Initialized = true;
    return NVDSINFER_SUCCESS;
}

// in nvdsinfer_context_impl.cpp
/* Deserialize engine and create backend context for the model from the init
 * params (caffemodel & prototxt/uff/onnx/etlt&key/custom-parser, int8
 * calibration tables, etc) and return the backend */

std::unique_ptr<BackendContext>
NvDsInferContextImpl::generateBackendContext(NvDsInferContextInitParams& initParams)
{
    int dla = -1;
    if (initParams.useDLA && initParams.dlaCore >= 0)
        dla = initParams.dlaCore;

    std::shared_ptr<TrtEngine> engine;  // TODO why using shared_ptr
    std::unique_ptr<BackendContext> backend;
    if (!string_empty(initParams.modelEngineFilePath))
    {
        if (!deserializeEngineAndBackend(
                initParams.modelEngineFilePath, dla, engine, backend))
        {
            printWarning(
                "deserialize backend context from engine from file :%s failed, "
                "try rebuild",
                safeStr(initParams.modelEngineFilePath));
        }
    }

    if (backend &&
        checkBackendParams(*backend, initParams) == NVDSINFER_SUCCESS)
    {
        printInfo("Use deserialized engine model: %s",
            safeStr(initParams.modelEngineFilePath));
        return backend;
    }
    else if (backend)
    {
        printWarning(
            "deserialized backend context :%s failed to match config params, "
            "trying rebuild",
            safeStr(initParams.modelEngineFilePath));
        backend.reset();
        engine.reset();
    }

    backend = buildModel(initParams);  // build the model if deserized fail
...
    return backend;
}

/* Create engine and backend context for the model from the init params
 * (caffemodel & prototxt/uff/onnx, int8 calibration tables, etc) and return the
 * backend */
std::unique_ptr<BackendContext>
NvDsInferContextImpl::buildModel(NvDsInferContextInitParams& initParams)
{
    printInfo("Trying to create engine from model files");

    std::unique_ptr<TrtModelBuilder> builder =
        std::make_unique<TrtModelBuilder>(
            initParams.gpuID, *gTrtLogger, m_CustomLibHandle);
    assert(builder);

    if (!string_empty(initParams.int8CalibrationFilePath) &&
        file_accessible(initParams.int8CalibrationFilePath))
    {
        auto calibrator = std::make_unique<NvDsInferInt8Calibrator>(
            initParams.int8CalibrationFilePath);
        builder->setInt8Calibrator(std::move(calibrator));
    }

    std::string enginePath;
    std::shared_ptr<TrtEngine> engine =
        builder->buildModel(initParams, enginePath);
    if (!engine)
    {
        printError("build engine file failed");
        return nullptr;
    }

    if (builder->serializeEngine(enginePath, engine->engine()) !=
        NVDSINFER_SUCCESS)
    {
        printWarning(
            "failed to serialize cude engine to file: %s", safeStr(enginePath));
    }
    else
    {
        printInfo("serialize cuda engine to file: %s successfully",
            safeStr(enginePath));
    }

    std::unique_ptr<BackendContext> backend;
    auto newBackend = createBackendContext(engine); // TODO why not assign the return value to backend
    if (!newBackend)
    {
        printWarning("create backend context from engine failed");
        return nullptr;
    }

    engine->printEngineInfo();

    backend = std::move(newBackend);

    if (checkBackendParams(*backend, initParams) != NVDSINFER_SUCCESS)
    {
        printError(
            "deserialized backend context :%s failed to match config params",
            safeStr(enginePath));
        return nullptr;
    }

    builder.reset();

    return backend;
}


```
```cpp
// in nvdsinfer_backend.h
/**
 * Helper class for managing Cuda Streams.
 */
class CudaStream
{
public:
    explicit CudaStream(uint flag = cudaStreamDefault, int priority = 0); //TODO explicit
    ~CudaStream();
    operator cudaStream_t() { return m_Stream; }
    cudaStream_t& ptr() { return m_Stream; }
    SIMPLE_MOVE_COPY(CudaStream)

private:
    void move_copy(CudaStream&& o) // move implementaton for rvalue 
    {
        m_Stream = o.m_Stream;
        o.m_Stream = nullptr;
    }
    DISABLE_CLASS_COPY(CudaStream);

    cudaStream_t m_Stream = nullptr;
};

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
