```cpp
//strcut _GstNvInferImpl is not defined so the member impl is the GstNvInferImpl* and we use reinterpret_cast to cast the type to DsNvInferImpl*
typedef struct _GstNvInferImpl GstNvInferImpl;
#define DS_NVINFER_IMPL(gst_nvinfer) reinterpret_cast<DsNvInferImpl*>((gst_nvinfer)->impl)
* GstNvInfer element structure.
 */
struct _GstNvInfer {
...
  GstNvInferImpl *impl;
...
};
```

```cpp
//Use unique ptr to manage the memory, and also sepcify the deleter to unref the memory then the lifetime end
  auto pool_deleter = [](GstBufferPool *p) { if (p) gst_object_unref (p); };
  std::unique_ptr<GstBufferPool, decltype(pool_deleter)> pool_ptr (
      gst_buffer_pool_new (), pool_deleter);
```
