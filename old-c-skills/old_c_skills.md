## macro

* to check function defination
    * `An identifier can be declared as often as you want.However, it must be defined exactly once.`
    below macro can be used to check if the type of callback funciton the same as NvDsInferParseCustomFunc
    ```cpp
            /**
         * Type definition for the custom bounding box parsing function.
         *
         * @param[in]  outputLayersInfo A vector containing information on the output
         *                              layers of the model.
         * @param[in]  networkInfo      Network information.
         * @param[in]  detectionParams  Detection parameters required for parsing
         *                              objects.
         * @param[out] objectList       A reference to a vector in which the function
         *                              is to add parsed objects.
         */
        typedef bool (* NvDsInferParseCustomFunc) (
                std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                NvDsInferNetworkInfo  const &networkInfo,
                NvDsInferParseDetectionParams const &detectionParams,
                std::vector<NvDsInferObjectDetectionInfo> &objectList);

        /**
         * Validates a custom parser function definition. Must be called
         * after defining the function.
         */
        #define CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(customParseFunc) \
            static void checkFunc_ ## customParseFunc (NvDsInferParseCustomFunc func = customParseFunc) \
                { checkFunc_ ## customParseFunc (); }; \
            extern "C" bool customParseFunc (std::vector<NvDsInferLayerInfo> const &outputLayersInfo, \
                   NvDsInferNetworkInfo  const &networkInfo, \
                   NvDsInferParseDetectionParams const &detectionParams, \
                   std::vector<NvDsInferObjectDetectionInfo> &objectList);
    ```
    in callback function file 
    ```cpp
    /* Check that the custom function has been defined correctly */
    CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomSSDTLT);

    ```
    expand the macro
    ```cpp
    static void checkFunc_NvDsInferParseCustomSSDTLT (NvDsInferParseCustomFunc func = NvDsInferParseCustomSSDTLT) { checkFunc_NvDsInferParseCustomSSDTLT (); }; 
    extern "C" bool NvDsInferParseCustomSSDTLT (std::vector<NvDsInferLayerInfo> const &outputLayersInfo, NvDsInferNetworkInfo const &networkInfo, NvDsInferParseDetectionParams const &detectionParams, std::vector<NvDsInferObjectDetectionInfo> &objectList);;

    ```
