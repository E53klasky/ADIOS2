#include "CompressCAESAR.h"
#include "adios2/helper/adiosFunctions.h"
#include "models/caesar_compress.h"
#include "models/caesar_decompress.h"
#include "dataset/dataset.h"
#include <cstring>

namespace adios2
{
    namespace core
    {
        namespace compress
        {
            // add amd support later --- ONCE GAE HAS IT
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

                MakeCommonHeader(bufferOut , bufferOutOffset , bufferVersion);

                const size_t ndims = blockCount.size();
                PutParameter(bufferOut , bufferOutOffset , ndims);
                for (const auto& d : blockCount) {
                    PutParameter(bufferOut , bufferOutOffset , d);
                }
                PutParameter(bufferOut , bufferOutOffset , type);

                if (ndims != 3 && ndims != 4) {
                    helper::Throw<std::invalid_argument>("Operator" , "CompressCAESAR" , "Operate" ,
                        "CAESAR only supports 3D and 4D data, got " + std::to_string(ndims) + " dimensions");
                }

                // MAYBE this should be within CAESAR in the dataset??????
                if (blockCount[0] < 8) {
                    helper::Throw<std::invalid_argument>("Operator" , "CompressCAESAR" , "Operate" ,
                        "First dimension must be >= 8 for CAESAR compression, got " + std::to_string(blockCount[0]));
                }

                // Check if compression idk also need to add the 128x128 for the spatial dims to not compress
                size_t thresholdSize = 1 * 1024 * 1024; // 1MB threshold
                size_t totalSize = helper::GetTotalSize(blockCount , helper::GetDataTypeSize(type));
                if (totalSize < thresholdSize) {
                    PutParameter(bufferOut , bufferOutOffset , false);
                    return bufferOutOffset;
                }


                torch::Tensor data_tensor;
                std::vector<int64_t> sizes;
                for (const auto& d : blockCount) {
                    sizes.push_back(static_cast<int64_t>(d));
                }

                if (type == DataType::Float) {
                    data_tensor = torch::from_blob(const_cast<char*>(dataIn) , sizes , torch::kFloat32).clone();
                }
                else if (type == DataType::Double) {
                    data_tensor = torch::from_blob(const_cast<char*>(dataIn) , sizes , torch::kFloat64)
                        .to(torch::kFloat32).clone();
                }
                else {
                    helper::Throw<std::invalid_argument>("Operator" , "CompressCAESAR" , "Operate" ,
                        "Unsupported data type");
                }

                torch::Tensor data_5d;
                if (ndims == 3) {
                    data_5d = data_tensor.unsqueeze(0).unsqueeze(0);
                }
                else {
                    data_5d = data_tensor.unsqueeze(0);
                }

                DatasetConfig config;
                config.memory_data = data_5d;
                config.n_frame = static_cast<int>(blockCount[0]);
                config.dataset_name = "ADIOS2_Block";
                config.variable_idx = 0;
                config.train_mode = false;
                config.inst_norm = true;
                config.norm_type = "mean_range";
                config.n_overlap = 0;

                // Get batch size from parameters or use default default to 32
                int batch_size = 32;
                auto itBatchSize = m_Parameters.find("batch_size");
                if (itBatchSize != m_Parameters.end()) {
                    batch_size = std::stoi(itBatchSize->second);
                }

                // Add better logic for amd once we get it 
                std::string device_str = DetectDevice();
                auto device = (device_str == "cuda") ? torch::kCUDA : torch::kCPU;
                Compressor compressor(device);
                CompressionResult result = compressor.compress(config , batch_size);

                PutParameter(bufferOut , bufferOutOffset , true);

                // Write compression metadata
                PutParameter(bufferOut , bufferOutOffset , static_cast<uint64_t>(result.num_samples));
                PutParameter(bufferOut , bufferOutOffset , static_cast<uint64_t>(result.num_batches));

                // Write encoded streams sizes first
                PutParameter(bufferOut , bufferOutOffset , static_cast<uint64_t>(result.encoded_latents.size()));
                for (const auto& stream : result.encoded_latents) {
                    PutParameter(bufferOut , bufferOutOffset , static_cast<uint64_t>(stream.size()));
                }

                PutParameter(bufferOut , bufferOutOffset , static_cast<uint64_t>(result.encoded_hyper_latents.size()));
                for (const auto& stream : result.encoded_hyper_latents) {
                    PutParameter(bufferOut , bufferOutOffset , static_cast<uint64_t>(stream.size()));
                }

                // Write encoded latent streams
                for (const auto& stream : result.encoded_latents) {
                    std::memcpy(bufferOut + bufferOutOffset , stream.data() , stream.size());
                    bufferOutOffset += stream.size();
                }

                // Write encoded hyper-latent streams
                for (const auto& stream : result.encoded_hyper_latents) {
                    std::memcpy(bufferOut + bufferOutOffset , stream.data() , stream.size());
                    bufferOutOffset += stream.size();
                }

                return bufferOutOffset;
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

                const bool isCompressed = GetParameter<bool>(bufferIn , bufferInOffset);

                size_t sizeOut = helper::GetTotalSize(blockCount , helper::GetDataTypeSize(type));

                if (!isCompressed)
                {
                    return 0;
                }

                // Read compression metadata
                uint64_t num_samples = GetParameter<uint64_t>(bufferIn , bufferInOffset);
                uint64_t num_batches = GetParameter<uint64_t>(bufferIn , bufferInOffset);

                // Read encoded stream sizes
                uint64_t num_latent_streams = GetParameter<uint64_t>(bufferIn , bufferInOffset);
                std::vector<uint64_t> latent_sizes(num_latent_streams);
                for (auto& size : latent_sizes) {
                    size = GetParameter<uint64_t>(bufferIn , bufferInOffset);
                }

                uint64_t num_hyper_latent_streams = GetParameter<uint64_t>(bufferIn , bufferInOffset);
                std::vector<uint64_t> hyper_latent_sizes(num_hyper_latent_streams);
                for (auto& size : hyper_latent_sizes) {
                    size = GetParameter<uint64_t>(bufferIn , bufferInOffset);
                }

                // Read encoded latent streams
                std::vector<std::string> encoded_latents;
                for (auto size : latent_sizes) {
                    encoded_latents.emplace_back(bufferIn + bufferInOffset , size);
                    bufferInOffset += size;
                }

                // Read encoded hyper-latent streams
                std::vector<std::string> encoded_hyper_latents;
                for (auto size : hyper_latent_sizes) {
                    encoded_hyper_latents.emplace_back(bufferIn + bufferInOffset , size);
                    bufferInOffset += size;
                }

                std::string device_str = DetectDevice();
                auto device = (device_str == "cuda") ? torch::kCUDA : torch::kCPU;

                int batch_size = 32;
                auto itBatchSize = m_Parameters.find("batch_size");
                if (itBatchSize != m_Parameters.end()) {
                    batch_size = std::stoi(itBatchSize->second);
                }

                Decompressor decompressor(device);
                int n_frame = static_cast<int>(blockCount[0]);

                DecompressionResult result = decompressor.decompress(
                    encoded_latents ,
                    encoded_hyper_latents ,
                    batch_size ,
                    n_frame
                );

                // Reassemble the decompressed data
                // The result contains multiple samples that need to be concatenated
                if (result.reconstructed_data.empty()) {
                    helper::Throw<std::runtime_error>("Operator" , "CompressCAESAR" , "DecompressV1" ,
                        "Decompression returned no data");
                }

                // Concatenate all reconstructed samples
                torch::Tensor reconstructed;
                if (result.reconstructed_data.size() == 1) {
                    reconstructed = result.reconstructed_data[0];
                }
                else {
                    reconstructed = torch::cat(result.reconstructed_data , 0);
                }

                // Remove dummy dimensions to match original shape
                while (reconstructed.dim() > static_cast<int64_t>(ndims)) {
                    reconstructed = reconstructed.squeeze(0);
                }

                // Convert to CPU and contiguous need better logic for gpu also ask question it
                reconstructed = reconstructed.to(torch::kCPU).contiguous();

                // Copy to output buffer
                if (type == DataType::Float) {
                    std::memcpy(dataOut , reconstructed.data_ptr<float>() , sizeOut);
                }
                else if (type == DataType::Double) {
                    torch::Tensor double_tensor = reconstructed.to(torch::kFloat64);
                    std::memcpy(dataOut , double_tensor.data_ptr<double>() , sizeOut);
                }

                return sizeOut;
            }

        } // end namespace compress
    } // end namespace core
} // end namespace adios2