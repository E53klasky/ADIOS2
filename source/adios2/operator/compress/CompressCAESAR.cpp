#include "CompressCAESAR.h"
#include "adios2/helper/adiosFunctions.h"
#include "models/compress/compressor.h"
#include "dataset/dataset.h"

namespace adios2
{
    namespace core
    {
        namespace compress
        {
            // Helper function to detect device
            std::string DetectDevice()
            {
                if (torch::cuda::is_available()) {
                    return "cuda";
                }
                return "cpu";
            }

            CompressCAESAR::CompressCAESAR(const Params& parameters)
                : Operator("caesar" , COMPRESS_CAESAR , "compress" , parameters)
            {
            }

            size_t CompressCAESAR::Operate(const char* dataIn , const Dims& blockStart , const Dims& blockCount ,
                const DataType type , char* bufferOut)
            {
                const uint8_t bufferVersion = 1;
                size_t bufferOutOffset = 0;

                // Write common header
                MakeCommonHeader(bufferOut , bufferOutOffset , bufferVersion);

                // Store dimension information
                const size_t ndims = blockCount.size();

                // CAESAR metadata section
                PutParameter(bufferOut , bufferOutOffset , ndims);
                for (const auto& d : blockCount)
                {
                    PutParameter(bufferOut , bufferOutOffset , d);
                }
                PutParameter(bufferOut , bufferOutOffset , type);

                // TODO: Add CAESAR version info if available
                // PutParameter(bufferOut, bufferOutOffset, static_cast<uint8_t>(CAESAR_VERSION_MAJOR));

                // Validate dimensions - CAESAR only supports 3D and 4D data
                if (ndims < 3 || ndims > 4)
                {
                    helper::Throw<std::invalid_argument>("Operator" , "CompressCAESAR" , "Operate" ,
                        "CAESAR only supports 3D and 4D data, got " +
                        std::to_string(ndims) + " dimensions");
                }

                PutParameter(bufferOut , bufferOutOffset , true);

                std::string device_str = DetectDevice();
                // fix logic later 
                auto device = (device_str == "cuda") ? torch::kCUDA : torch::kCPU;
                Compressor compressor(device);

                DatasetConfig config;
                // memory_data needs to be converted to a 5d tensor (n1,n2,n3,n4,n5)
                // n1 = variable should always be one
                // n2 = data must be at least 1
                // n3 = data must be at least 8
                // n4 = data must be at least 256
                // n5 = data must be at least 256

                // TODO: Figure out how to pass dataIn pointer to config.memory_data
                // config.memory_data is std::optional<torch::Tensor>, not void*
                // May need to create torch::Tensor from dataIn first

                config.n_frame = 8;
                config.dataset_name = "ADIOS2_dataset";
                config.variable_idx = 0;

                // batch size maybe should be determined by the cpu, gpu, like 32 cpu, 64 gpu
                CompressionResult result = compressor.compress(config , 32);

                // Serialize CompressionResult into bufferOut
                // TODO: Need to serialize:
                // - result.num_samples
                // - result.num_batches
                // - result.latents (vector of tensors)
                // - result.hyper_latents (vector of tensors)
                // - result.offsets (vector of tensors)
                // - result.scales (vector of tensors)
                // - result.indices (vector of tensor arrays)

                PutParameter(bufferOut , bufferOutOffset , result.num_samples);
                PutParameter(bufferOut , bufferOutOffset , result.num_batches);

                return static_cast<size_t>(-1);
            }

            size_t CompressCAESAR::InverseOperate(const char* bufferIn , const size_t sizeIn , char* dataOut)
            {
                size_t bufferInOffset = 1; // skip operator type
                const uint8_t bufferVersion = GetParameter<uint8_t>(bufferIn , bufferInOffset);
                bufferInOffset += 2; // skip two reserved bytes

                if (bufferVersion == 1)
                {
                    return DecompressV1(bufferIn + bufferInOffset , sizeIn - bufferInOffset , dataOut);
                }
                else
                {
                    helper::Throw<std::runtime_error>("Operator" , "CompressCAESAR" , "InverseOperate" ,
                        "invalid CAESAR buffer version");
                }

                return 0;
            }

            bool CompressCAESAR::IsDataTypeValid(const DataType type) const
            {
                // NOTE: Float and Double are confirmed to work but see if others can be added
                if (type == DataType::Double || type == DataType::Float)
                {
                    return true;
                }
                return false;
            }

            size_t CompressCAESAR::DecompressV1(const char* bufferIn , const size_t sizeIn , char* dataOut)
            {
                size_t bufferInOffset = 0;

                // Read metadata
                const size_t ndims = GetParameter<size_t , size_t>(bufferIn , bufferInOffset);
                Dims blockCount(ndims);
                for (size_t i = 0; i < ndims; ++i)
                {
                    blockCount[i] = GetParameter<size_t , size_t>(bufferIn , bufferInOffset);
                }
                const DataType type = GetParameter<DataType>(bufferIn , bufferInOffset);

                // HARDCODED for now - adjust as needed
                m_VersionInfo = "CAESAR_V";

                size_t sizeOut = helper::GetTotalSize(blockCount , helper::GetDataTypeSize(type));

                // TODO: Deserialize CompressionResult from bufferIn
                // TODO: Call CAESAR decompression
                // TODO: Write decompressed data to dataOut

                return static_cast<size_t>(-1);
            }

        } // end namespace compress
    } // end namespace core
} // end namespace adios2