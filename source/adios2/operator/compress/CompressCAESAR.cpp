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

            std::string DetectDevice()
            {
                if (torch::cuda::is_available()) {
                    return "cuda";
                }
                return "cpu";
            }


// =============================================================
// Basic Put/Get for primitive data types
// =============================================================

            template <typename T>
            void PutParameter(std::vector<uint8_t>& buffer , size_t& pos , const T& parameter)
            {
                size_t newSize = pos + sizeof(T);
                if (buffer.size() < newSize) buffer.resize(newSize);
                std::memcpy(buffer.data() + pos , &parameter , sizeof(T));
                pos += sizeof(T);
            }

            template <typename T>
            T GetParameter(const std::vector<uint8_t>& buffer , size_t& pos)
            {
                T ret;
                std::memcpy(&ret , buffer.data() + pos , sizeof(T));
                pos += sizeof(T);
                return ret;
            }

            // =============================================================
            // std::string
            // =============================================================

            inline void PutString(std::vector<uint8_t>& buffer , size_t& pos , const std::string& s)
            {
                uint64_t size = s.size();
                PutParameter(buffer , pos , size);
                size_t newSize = pos + size;
                if (buffer.size() < newSize) buffer.resize(newSize);
                std::memcpy(buffer.data() + pos , s.data() , size);
                pos += size;
            }

            inline std::string GetString(const std::vector<uint8_t>& buffer , size_t& pos)
            {
                uint64_t size = GetParameter<uint64_t>(buffer , pos);
                std::string s(size , '\0');
                std::memcpy(s.data() , buffer.data() + pos , size);
                pos += size;
                return s;
            }

            // =============================================================
            // std::vector<T>
            // =============================================================

            template <typename T>
            void PutVector(std::vector<uint8_t>& buffer , size_t& pos , const std::vector<T>& v)
            {
                uint64_t size = v.size();
                PutParameter(buffer , pos , size);
                for (const auto& val : v)
                    PutParameter(buffer , pos , val);
            }

            template <typename T>
            std::vector<T> GetVector(const std::vector<uint8_t>& buffer , size_t& pos)
            {
                uint64_t size = GetParameter<uint64_t>(buffer , pos);
                std::vector<T> v(size);
                for (auto& val : v)
                    val = GetParameter<T>(buffer , pos);
                return v;
            }

            // =============================================================
            // std::vector<std::vector<T>>
            // =============================================================

            template <typename T>
            void PutVector2D(std::vector<uint8_t>& buffer , size_t& pos , const std::vector<std::vector<T>>& vv)
            {
                uint64_t outer = vv.size();
                PutParameter(buffer , pos , outer);
                for (const auto& inner : vv)
                    PutVector(buffer , pos , inner);
            }

            template <typename T>
            std::vector<std::vector<T>> GetVector2D(const std::vector<uint8_t>& buffer , size_t& pos)
            {
                uint64_t outer = GetParameter<uint64_t>(buffer , pos);
                std::vector<std::vector<T>> vv(outer);
                for (auto& inner : vv)
                    inner = GetVector<T>(buffer , pos);
                return vv;
            }

            // =============================================================
            // std::pair<T1, T2>
            // =============================================================

            template <typename T1 , typename T2>
            void PutPair(std::vector<uint8_t>& buffer , size_t& pos , const std::pair<T1 , T2>& p)
            {
                PutParameter(buffer , pos , p.first);
                PutParameter(buffer , pos , p.second);
            }

            template <typename T1 , typename T2>
            std::pair<T1 , T2> GetPair(const std::vector<uint8_t>& buffer , size_t& pos)
            {
                T1 first = GetParameter<T1>(buffer , pos);
                T2 second = GetParameter<T2>(buffer , pos);
                return { first, second };
            }

            // =============================================================
            // std::tuple<int32_t, int32_t, std::vector<int32_t>>
            // =============================================================

            inline void PutTuple(std::vector<uint8_t>& buffer , size_t& pos , const std::tuple<int32_t , int32_t , std::vector<int32_t>>& t)
            {
                PutParameter(buffer , pos , std::get<0>(t));
                PutParameter(buffer , pos , std::get<1>(t));
                PutVector(buffer , pos , std::get<2>(t));
            }

            inline std::tuple<int32_t , int32_t , std::vector<int32_t>> GetTuple(const std::vector<uint8_t>& buffer , size_t& pos)
            {
                int32_t a = GetParameter<int32_t>(buffer , pos);
                int32_t b = GetParameter<int32_t>(buffer , pos);
                auto v = GetVector<int32_t>(buffer , pos);
                return { a, b, v };
            }

            // =============================================================
            // GAEMetaData
            // =============================================================

            struct GAEMetaData {
                bool GAE_correction_occur;
                std::vector<int> padding_recon_info;
                std::vector<std::vector<float>> pcaBasis;
                std::vector<float> uniqueVals;
                double quanBin;
                int64_t nVec;
                int64_t prefixLength;
                int64_t dataBytes;
                size_t coeffIntBytes;
            };

            inline void PutGAEMetaData(std::vector<uint8_t>& buffer , size_t& pos , const GAEMetaData& m)
            {
                PutParameter(buffer , pos , m.GAE_correction_occur);
                PutVector(buffer , pos , m.padding_recon_info);
                PutVector2D(buffer , pos , m.pcaBasis);
                PutVector(buffer , pos , m.uniqueVals);
                PutParameter(buffer , pos , m.quanBin);
                PutParameter(buffer , pos , m.nVec);
                PutParameter(buffer , pos , m.prefixLength);
                PutParameter(buffer , pos , m.dataBytes);
                PutParameter(buffer , pos , m.coeffIntBytes);
            }

            inline GAEMetaData GetGAEMetaData(const std::vector<uint8_t>& buffer , size_t& pos)
            {
                GAEMetaData m;
                m.GAE_correction_occur = GetParameter<bool>(buffer , pos);
                m.padding_recon_info = GetVector<int>(buffer , pos);
                m.pcaBasis = GetVector2D<float>(buffer , pos);
                m.uniqueVals = GetVector<float>(buffer , pos);
                m.quanBin = GetParameter<double>(buffer , pos);
                m.nVec = GetParameter<int64_t>(buffer , pos);
                m.prefixLength = GetParameter<int64_t>(buffer , pos);
                m.dataBytes = GetParameter<int64_t>(buffer , pos);
                m.coeffIntBytes = GetParameter<size_t>(buffer , pos);
                return m;
            }

            // =============================================================
            // CompressionMetaData
            // =============================================================

            struct CompressionMetaData {
                std::vector<float> offsets;
                std::vector<float> scales;
                std::vector<std::vector<int32_t>> indexes;
                std::tuple<int32_t , int32_t , std::vector<int32_t>> block_info;
                std::vector<int32_t> data_input_shape;
                std::vector<std::pair<int32_t , float>> filtered_blocks;
                float global_scale;
                float global_offset;
                int64_t pad_T;
            };

            inline void PutCompressionMetaData(std::vector<uint8_t>& buffer , size_t& pos , const CompressionMetaData& m)
            {
                PutVector(buffer , pos , m.offsets);
                PutVector(buffer , pos , m.scales);
                PutVector2D(buffer , pos , m.indexes);
                PutTuple(buffer , pos , m.block_info);
                PutVector(buffer , pos , m.data_input_shape);

                uint64_t fb_size = m.filtered_blocks.size();
                PutParameter(buffer , pos , fb_size);
                for (auto& p : m.filtered_blocks)
                    PutPair(buffer , pos , p);

                PutParameter(buffer , pos , m.global_scale);
                PutParameter(buffer , pos , m.global_offset);
                PutParameter(buffer , pos , m.pad_T);
            }

            inline CompressionMetaData GetCompressionMetaData(const std::vector<uint8_t>& buffer , size_t& pos)
            {
                CompressionMetaData m;
                m.offsets = GetVector<float>(buffer , pos);
                m.scales = GetVector<float>(buffer , pos);
                m.indexes = GetVector2D<int32_t>(buffer , pos);
                m.block_info = GetTuple(buffer , pos);
                m.data_input_shape = GetVector<int32_t>(buffer , pos);

                uint64_t fb_size = GetParameter<uint64_t>(buffer , pos);
                m.filtered_blocks.resize(fb_size);
                for (auto& p : m.filtered_blocks)
                    p = GetPair<int32_t , float>(buffer , pos);

                m.global_scale = GetParameter<float>(buffer , pos);
                m.global_offset = GetParameter<float>(buffer , pos);
                m.pad_T = GetParameter<int64_t>(buffer , pos);
                return m;
            }

            inline void PutVectorOfStrings(std::vector<uint8_t>& buffer , size_t& pos ,
                const std::vector<std::string>& v)
            {
                uint64_t count = v.size();
                PutParameter(buffer , pos , count);
                for (const auto& s : v)
                    PutString(buffer , pos , s);
            }

            inline std::vector<std::string> GetVectorOfStrings(const std::vector<uint8_t>& buffer , size_t& pos)
            {
                uint64_t count = GetParameter<uint64_t>(buffer , pos);
                std::vector<std::string> v(count);
                for (auto& s : v)
                    s = GetString(buffer , pos);
                return v;
            }


            size_t CompressCAESAR::Operate(
                const char* dataIn ,
                const Dims& blockStart ,
                const Dims& blockCount ,
                const DataType type ,
                char* bufferOut)
            {
                const uint8_t bufferVersion = 1;
                size_t bufferOutOffset = 0;

                MakeCommonHeader(bufferOut , bufferOutOffset , bufferVersion);

                const size_t ndims = blockCount.size();
                PutParameter(bufferOut , bufferOutOffset , ndims);
                for (const auto& d : blockCount)
                    PutParameter(bufferOut , bufferOutOffset , d);
                PutParameter(bufferOut , bufferOutOffset , type);

                if (ndims != 3 && ndims != 4)
                    helper::Throw<std::invalid_argument>(
                        "Operator" , "CompressCAESAR" , "Operate" ,
                        "CAESAR only supports 3D and 4D data, got " + std::to_string(ndims) + " dimensions");

                if (blockCount[0] < 8)
                    helper::Throw<std::invalid_argument>(
                        "Operator" , "CompressCAESAR" , "Operate" ,
                        "First dimension must be >= 8 for CAESAR compression, got " + std::to_string(blockCount[0]));

                size_t thresholdSize = 1 * 1024 * 1024;
                size_t totalSize = helper::GetTotalSize(blockCount , helper::GetDataTypeSize(type));
                if (totalSize < thresholdSize)
                {
                    PutParameter(bufferOut , bufferOutOffset , false);
                    return bufferOutOffset;
                }

                // Prepare input tensor
                torch::Tensor data_tensor;
                std::vector<int64_t> sizes(blockCount.begin() , blockCount.end());

                if (type == DataType::Float)
                    data_tensor = torch::from_blob(const_cast<char*>(dataIn) , sizes , torch::kFloat32).clone();
                else if (type == DataType::Double)
                    data_tensor = torch::from_blob(const_cast<char*>(dataIn) , sizes , torch::kFloat64)
                    .to(torch::kFloat32)
                    .clone();
                else
                    helper::Throw<std::invalid_argument>("Operator" , "CompressCAESAR" , "Operate" , "Unsupported data type");

                torch::Tensor data_5d = (ndims == 3) ? data_tensor.unsqueeze(0).unsqueeze(0)
                    : data_tensor.unsqueeze(0);

                DatasetConfig config;
                config.memory_data = data_5d;
                config.n_frame = 8;
                config.dataset_name = "ADIOS2_Block";
                config.variable_idx = 0;
                config.train_mode = false;
                config.inst_norm = true;
                config.norm_type = "mean_range";
                config.n_overlap = 0;

                int batch_size = 32;
                auto itBatchSize = m_Parameters.find("batch_size");
                if (itBatchSize != m_Parameters.end())
                    batch_size = std::stoi(itBatchSize->second);

                float rel_eb = 0.001f;
                auto itRelEB = m_Parameters.find("rel_eb");
                if (itRelEB != m_Parameters.end())
                    rel_eb = std::stof(itRelEB->second);

                std::string device_str = DetectDevice();
                auto device = (device_str == "cuda") ? torch::kCUDA : torch::kCPU;
                Compressor compressor(device);

                CompressionResult comp = compressor.compress(config , batch_size , rel_eb);

                // === Start writing output buffer ===
                PutParameter(bufferOut , bufferOutOffset , true); // mark as compressed

                // encoded streams (directly to char*)
                WriteVectorOfStrings(bufferOut , bufferOutOffset , comp.encoded_latents);
                WriteVectorOfStrings(bufferOut , bufferOutOffset , comp.encoded_hyper_latents);

                WriteVector(bufferOut , bufferOutOffset , comp.gae_comp_data);

                // CompressionMetaData
                WriteCompressionMetaData(bufferOut , bufferOutOffset , comp.compressionMetaData);

                // GAEMetaData
                WriteGAEMetaData(bufferOut , bufferOutOffset , comp.gaeMetaData);

                // Other scalar fields
                PutParameter(bufferOut , bufferOutOffset , comp.final_nrmse);
                PutParameter(bufferOut , bufferOutOffset , comp.num_samples);
                PutParameter(bufferOut , bufferOutOffset , comp.num_batches);

                // Decompression parameters
                PutParameter(bufferOut , bufferOutOffset , batch_size);
                PutParameter(bufferOut , bufferOutOffset , config.n_frame);

                return bufferOutOffset;
            }

            size_t CompressCAESAR::DecompressV1(
                const char* bufferIn ,
                const size_t sizeIn ,
                char* dataOut)
            {
                size_t bufferInOffset = 0;

                const size_t ndims = GetParameter<size_t , size_t>(bufferIn , bufferInOffset);
                Dims blockCount(ndims);
                for (size_t i = 0; i < ndims; ++i)
                    blockCount[i] = GetParameter<size_t , size_t>(bufferIn , bufferInOffset);

                const DataType type = GetParameter<DataType>(bufferIn , bufferInOffset);
                const bool isCompressed = GetParameter<bool>(bufferIn , bufferInOffset);
                if (!isCompressed)
                    return 0;

                size_t sizeOut = helper::GetTotalSize(blockCount , helper::GetDataTypeSize(type));

                // Read compressed data
                std::vector<std::string> encoded_latents = ReadVectorOfStrings(bufferIn , bufferInOffset);
                std::vector<std::string> encoded_hyper_latents = ReadVectorOfStrings(bufferIn , bufferInOffset);
                std::vector<uint8_t> gae_comp_data = ReadVectorPrimitive<uint8_t>(bufferIn , bufferInOffset);

                // CompressionMetaData
                CompressionMetaData meta = ReadCompressionMetaData(bufferIn , bufferInOffset);

                // GAEMetaData
                GAEMetaData gaeMeta = ReadGAEMetaData(bufferIn , bufferInOffset);

                // Other fields
                double final_nrmse = GetParameter<double>(bufferIn , bufferInOffset);
                int num_samples = GetParameter<int>(bufferIn , bufferInOffset);
                int num_batches = GetParameter<int>(bufferIn , bufferInOffset);
                int batch_size = GetParameter<int>(bufferIn , bufferInOffset);
                int n_frame = GetParameter<int>(bufferIn , bufferInOffset);

                // Rebuild CompressionResult
                CompressionResult comp;
                comp.encoded_latents = std::move(encoded_latents);
                comp.encoded_hyper_latents = std::move(encoded_hyper_latents);
                comp.gae_comp_data = std::move(gae_comp_data);
                comp.compressionMetaData = std::move(meta);
                comp.gaeMetaData = std::move(gaeMeta);
                comp.final_nrmse = final_nrmse;
                comp.num_samples = num_samples;
                comp.num_batches = num_batches;

                // Decompress
                std::string device_str = DetectDevice();
                torch::Device device = (device_str == "cuda") ? torch::Device(torch::kCUDA)
                    : torch::Device(torch::kCPU);
                Decompressor decompressor(device);

                torch::Tensor reconstructed = decompressor.decompress(
                    comp.encoded_latents ,
                    comp.encoded_hyper_latents ,
                    batch_size ,
                    n_frame ,
                    comp);

                // Post-process
                if (ndims == 3)
                {
                    reconstructed = reconstructed.squeeze(1);
                    reconstructed = reconstructed.reshape({ -1, reconstructed.size(-2), reconstructed.size(-1) });
                    reconstructed = reconstructed.slice(0 , 0 , blockCount[0]);
                }
                else if (ndims == 4)
                {
                    reconstructed = reconstructed.permute({ 1, 0, 2, 3, 4 });
                    reconstructed = reconstructed.reshape({ reconstructed.size(0), -1, reconstructed.size(-2), reconstructed.size(-1) });
                    reconstructed = reconstructed.slice(1 , 0 , blockCount[1]);

                    if (blockCount[0] > 1)
                        reconstructed = reconstructed.expand({ static_cast<int64_t>(blockCount[0]), -1, -1, -1 });
                }

                reconstructed = reconstructed.to(torch::kCPU).contiguous();

                // Copy to dataOut
                if (type == DataType::Float)
                    std::memcpy(dataOut , reconstructed.data_ptr<float>() , sizeOut);
                else if (type == DataType::Double)
                {
                    torch::Tensor double_tensor = reconstructed.to(torch::kFloat64);
                    std::memcpy(dataOut , double_tensor.data_ptr<double>() , sizeOut);
                }

                return sizeOut;
            }



        } // end namespace compress
    } // end namespace core
} // end namespace adios2