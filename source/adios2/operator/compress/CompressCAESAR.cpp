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
            std::string DetectDevice()
            {   // TODO: add ROCm support
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

                // doing and saving magic above idk what is goind on hope it works aove

                // for now ----------------------
                if (ndims != 3 && ndims != 4) {
                    helper::Throw<std::invalid_argument>("Operator" , "CompressCAESAR" , "Operate" ,
                        "CAESAR only supports 3D and 4D data, got " + std::to_string(ndims) + " dimensions");
                }

                // Validate n_frame requirement (first dimension must be >= 8) for now -----------------
                if (blockCount[0] < 8) {
                    std::cerr << "ERROR: CAESAR requires first dimension >= 8, got " << blockCount[0] << std::endl;
                    std::cerr << "Block dimensions: [";
                    for (size_t i = 0; i < ndims; i++) {
                        std::cerr << blockCount[i];
                        if (i < ndims - 1) std::cerr << ", ";
                    }
                    std::cerr << "]" << std::endl;
                    helper::Throw<std::invalid_argument>("Operator" , "CompressCAESAR" , "Operate" ,
                        "First dimension must be >= 8 for CAESAR compression");
                }

                PutParameter(bufferOut , bufferOutOffset , true); // idk --------------

                torch::Tensor data_tensor;

                if (type == DataType::Float) {
                    std::vector<int64_t> sizes;
                    for (const auto& d : blockCount) {
                        sizes.push_back(static_cast<int64_t>(d));
                    }
                    data_tensor = torch::from_blob(const_cast<char*>(dataIn) , sizes , torch::kFloat32);
                }
                else if (type == DataType::Double) {
                    std::vector<int64_t> sizes;
                    for (const auto& d : blockCount) {
                        sizes.push_back(static_cast<int64_t>(d));
                    }
                    data_tensor = torch::from_blob(const_cast<char*>(dataIn) , sizes , torch::kFloat64).to(torch::kFloat32);
                }
                else {
                    helper::Throw<std::invalid_argument>("Operator" , "CompressCAESAR" , "Operate" ,
                        "Unsupported data type");
                }

                data_tensor = data_tensor.clone(); // Make a copy to own the memory

                // Reshape to 5D by adding dummy dimensions
                torch::Tensor data_5d;
                if (ndims == 3) {
                    data_5d = data_tensor.unsqueeze(0).unsqueeze(0);
                }
                else { // ndims == 4
                    data_5d = data_tensor.unsqueeze(0);
                }

                // Setup DatasetConfig
                DatasetConfig config;
                config.memory_data = data_5d;
                config.n_frame = static_cast<int>(blockCount[0]);
                config.dataset_name = "ADIOS2_Block";
                config.variable_idx = 0;
                config.train_mode = false;
                config.inst_norm = true;
                config.norm_type = "mean_range";
                config.n_overlap = 0;


                std::string device_str = DetectDevice();
                auto device = (device_str == "cuda") ? torch::kCUDA : torch::kCPU;
                Compressor compressor(device);

                int batch_size = 32; // TODO: Make configurable (32 CPU, 64 GPU) or something for better performance
                CompressionResult result = compressor.compress(config , batch_size);

                // IDK magic below hope it works ----------------------
                PutParameter(bufferOut , bufferOutOffset , static_cast<uint64_t>(result.num_samples));
                PutParameter(bufferOut , bufferOutOffset , static_cast<uint64_t>(result.num_batches));


                size_t metadata_offset_pos = bufferOutOffset;
                PutParameter(bufferOut , bufferOutOffset , static_cast<uint64_t>(0)); // Placeholder

                // Write tensor data first (like data.bin)
                size_t tensor_data_start = bufferOutOffset;
                uint64_t current_offset = 0;

                std::vector<std::tuple<std::string , uint64_t , uint64_t , std::vector<int64_t>>> metadata_records;

                auto append_tensor = [&](const std::string& name , const torch::Tensor& tensor) {
                    torch::Tensor cpu_tensor = tensor.to(torch::kCPU).contiguous();
                    size_t num_bytes = cpu_tensor.numel() * sizeof(float);

                    // Write tensor data
                    std::memcpy(bufferOut + bufferOutOffset ,
                        cpu_tensor.data_ptr<float>() ,
                        num_bytes);

                    // Store metadata record below
                    std::vector<int64_t> shape = cpu_tensor.sizes().vec();
                    metadata_records.push_back({ name, current_offset, num_bytes, shape });

                    bufferOutOffset += num_bytes;
                    current_offset += num_bytes;
                    };

                for (size_t i = 0; i < result.num_samples; i++) {
                    append_tensor("latent_" + std::to_string(i) , result.latents[i]);
                    append_tensor("hyper_latent_" + std::to_string(i) , result.hyper_latents[i]);
                    append_tensor("offset_" + std::to_string(i) , result.offsets[i]);
                    append_tensor("scale_" + std::to_string(i) , result.scales[i]);
                }

                // Write indices
                for (size_t i = 0; i < result.num_samples; i++) {
                    auto& idx = result.indices[i];
                    for (int j = 0; j < 4; j++) {
                        int64_t val = idx[j].item<int64_t>();
                        PutParameter(bufferOut , bufferOutOffset , val);
                    }
                }

                // Now write metadata section (like meta.bin)
                size_t metadata_section_start = bufferOutOffset;

                // Update metadata offset placeholder
                uint64_t metadata_offset_value = metadata_section_start - tensor_data_start;
                std::memcpy(bufferOut + metadata_offset_pos , &metadata_offset_value , sizeof(uint64_t));

                // Write metadata
                uint32_t num_tensors = result.num_samples * 4; // latent, hyper_latent, offset, scale per sample
                PutParameter(bufferOut , bufferOutOffset , num_tensors);

                for (const auto& [name , offset , num_bytes , shape] : metadata_records) {
                    uint32_t name_len = name.size();
                    PutParameter(bufferOut , bufferOutOffset , name_len);
                    std::memcpy(bufferOut + bufferOutOffset , name.c_str() , name_len);
                    bufferOutOffset += name_len;

                    PutParameter(bufferOut , bufferOutOffset , offset);
                    PutParameter(bufferOut , bufferOutOffset , num_bytes);

                    uint32_t ndim = shape.size();
                    PutParameter(bufferOut , bufferOutOffset , ndim);
                    for (auto d : shape) {
                        PutParameter(bufferOut , bufferOutOffset , d);
                    }
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

                m_VersionInfo = "CAESAR decompression is currently under development. "
                    "Compressed data can be stored but decompression is not yet available. "
                    "Please check back in future releases for decompression support.";

                helper::Throw<std::runtime_error>("Operator" , "CompressCAESAR" , "DecompressV1" ,
                    m_VersionInfo);

                return 0;
            }

        } // end namespace compress
    } // end namespace core
} // end namespace adios2