- [GST BASIC](#gst-basic)
  * [Gst Event](#gst-event)
  * [Multithread in gstreamer](#Multithread-in-gstreamer)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

# GST BASIC

## Gst Event
* refer [llink](https://blog.csdn.net/knowledgebao/article/details/84621238)
>Gstreamer-GstEvent：https://blog.csdn.net/knowledgebao/article/details/86065064

>gst_element_get_request_pad与gst_element_request_pad的区别：https://blog.csdn.net/knowledgebao/article/details/85793298

## Multithread in gstreamer

refer [gst tutorial](https://gstreamer.freedesktop.org/documentation/tutorials/basic/multithreading-and-pad-availability.html?gi-language=c#)

## GstStructrue

proxy pattern
```cpp
 GstStructureField
 typedef struct
{
  GstStructure s;

  /* owned by parent structure, NULL if no parent */
  gint *parent_refcount;

  GArray *fields;
} GstStructureImpl;
struct _GstStructureField
{
  GQuark name;
  GValue value;
};
struct _GstStructure {
  GType type;

  /*< private >*/
  GQuark name;
};
static GstStructure *
gst_structure_new_id_empty_with_size (GQuark quark, guint prealloc)
{
  GstStructureImpl *structure;

  structure = g_slice_new (GstStructureImpl);
  ((GstStructure *) structure)->type = _gst_structure_type;
  ((GstStructure *) structure)->name = quark;
  GST_STRUCTURE_REFCOUNT (structure) = NULL;
  GST_STRUCTURE_FIELDS (structure) =
      g_array_sized_new (FALSE, FALSE, sizeof (GstStructureField), prealloc);

  GST_TRACE ("created structure %p", structure);

  return GST_STRUCTURE_CAST (structure);
}
```
