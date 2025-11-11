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

                if (blockCount[0] < 8) {
                    helper::Throw<std::invalid_argument>("Operator" , "CompressCAESAR" , "Operate" ,
                        "First dimension must be >= 8 for CAESAR compression, got " + std::to_string(blockCount[0]));
                }

                // Check if compression threshold
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
                config.n_frame = 8;
                config.dataset_name = "ADIOS2_Block";
                config.variable_idx = 0;
                config.train_mode = false;
                config.inst_norm = true;
                config.norm_type = "mean_range";
                config.n_overlap = 0;

                int batch_size = 32;
                auto itBatchSize = m_Parameters.find("batch_size");
                if (itBatchSize != m_Parameters.end()) {
                    batch_size = std::stoi(itBatchSize->second);
                }

                float rel_eb = 0.001f;
                auto itRelEB = m_Parameters.find("rel_eb");
                if (itRelEB != m_Parameters.end()) {
                    rel_eb = std::stof(itRelEB->second);
                }

                // Device setup
                std::string device_str = DetectDevice();
                auto device = (device_str == "cuda") ? torch::kCUDA : torch::kCPU;
                Compressor compressor(device);
                CompressionResult comp = compressor.compress(config , batch_size , rel_eb);

                // Mark as compressed
                PutParameter(bufferOut , bufferOutOffset , true);

                // Store encoded streams
                PutParameter(bufferOut , bufferOutOffset , comp.encoded_latents);
                PutParameter(bufferOut , bufferOutOffset , comp.encoded_hyper_latents);

                // Store GAE compressed data
                PutParameter(bufferOut , bufferOutOffset , comp.gae_comp_data);

                // Store CompressionMetaData fields individually
                const auto& meta = comp.compressionMetaData;
                PutParameter(bufferOut , bufferOutOffset , meta.offsets);
                PutParameter(bufferOut , bufferOutOffset , meta.scales);
                PutParameter(bufferOut , bufferOutOffset , meta.indexes);

                // Store block_info tuple components
                PutParameter(bufferOut , bufferOutOffset , std::get<0>(meta.block_info));
                PutParameter(bufferOut , bufferOutOffset , std::get<1>(meta.block_info));
                PutParameter(bufferOut , bufferOutOffset , std::get<2>(meta.block_info));

                PutParameter(bufferOut , bufferOutOffset , meta.data_input_shape);
                PutParameter(bufferOut , bufferOutOffset , meta.filtered_blocks);
                PutParameter(bufferOut , bufferOutOffset , meta.global_scale);
                PutParameter(bufferOut , bufferOutOffset , meta.global_offset);
                PutParameter(bufferOut , bufferOutOffset , meta.padding_recon_info);
                PutParameter(bufferOut , bufferOutOffset , meta.pad_T);

                // Store GAEMetaData fields individually
                const auto& gaeMeta = comp.gaeMetaData;
                PutParameter(bufferOut , bufferOutOffset , gaeMeta.pcaBasis);
                PutParameter(bufferOut , bufferOutOffset , gaeMeta.uniqueVals);
                PutParameter(bufferOut , bufferOutOffset , gaeMeta.quanBin);
                PutParameter(bufferOut , bufferOutOffset , gaeMeta.nVec);
                PutParameter(bufferOut , bufferOutOffset , gaeMeta.prefixLength);
                PutParameter(bufferOut , bufferOutOffset , gaeMeta.dataBytes);
                PutParameter(bufferOut , bufferOutOffset , gaeMeta.coeffIntBytes);

                // Store other CompressionResult fields
                PutParameter(bufferOut , bufferOutOffset , comp.final_nrmse);
                PutParameter(bufferOut , bufferOutOffset , comp.num_samples);
                PutParameter(bufferOut , bufferOutOffset , comp.num_batches);

                // Store batch_size and n_frame for decompression
                PutParameter(bufferOut , bufferOutOffset , batch_size);
                PutParameter(bufferOut , bufferOutOffset , 8); // n_frame

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
                std::cerr << "[CompressCAESAR][DecompressV1] Entered. sizeIn=" << sizeIn << std::endl;

                size_t bufferInOffset = 0;
                std::cerr << "[CompressCAESAR][DecompressV1] bufferInOffset=" << bufferInOffset << std::endl;

                // Read metadata
                const size_t ndims = GetParameter<size_t , size_t>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] ndims=" << ndims << " offset=" << bufferInOffset << std::endl;
                Dims blockCount(ndims);
                for (size_t i = 0; i < ndims; ++i)
                {
                    blockCount[i] = GetParameter<size_t , size_t>(bufferIn , bufferInOffset);
                    std::cerr << "[CompressCAESAR][DecompressV1] blockCount[" << i << "]=" << blockCount[i] << " offset=" << bufferInOffset << std::endl;
                }
                const DataType type = GetParameter<DataType>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] DataType=" << static_cast<int>(type) << " offset=" << bufferInOffset << std::endl;

                const bool isCompressed = GetParameter<bool>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] isCompressed=" << isCompressed << " offset=" << bufferInOffset << std::endl;

                size_t sizeOut = helper::GetTotalSize(blockCount , helper::GetDataTypeSize(type));
                std::cerr << "[CompressCAESAR][DecompressV1] Computed sizeOut=" << sizeOut << std::endl;

                if (!isCompressed)
                {
                    std::cerr << "[CompressCAESAR][DecompressV1] Data not compressed. Exiting and leaving data unchanged." << std::endl;
                    return 0;
                }

                // Read encoded streams
                std::cerr << "[CompressCAESAR][DecompressV1] Reading encoded_latents at offset=" << bufferInOffset << std::endl;
                std::vector<std::string> encoded_latents = GetParameter<std::vector<std::string>>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] encoded_latents count=" << encoded_latents.size() << std::endl;
                for (size_t i = 0; i < encoded_latents.size(); ++i) {
                    std::cerr << "[CompressCAESAR][DecompressV1] encoded_latents[" << i << "].size=" << encoded_latents[i].size() << std::endl;
                }

                std::cerr << "[CompressCAESAR][DecompressV1] Reading encoded_hyper_latents at offset=" << bufferInOffset << std::endl;
                std::vector<std::string> encoded_hyper_latents = GetParameter<std::vector<std::string>>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] encoded_hyper_latents count=" << encoded_hyper_latents.size() << std::endl;
                for (size_t i = 0; i < encoded_hyper_latents.size(); ++i) {
                    std::cerr << "[CompressCAESAR][DecompressV1] encoded_hyper_latents[" << i << "].size=" << encoded_hyper_latents[i].size() << std::endl;
                }

                // Read GAE compressed data
                std::cerr << "[CompressCAESAR][DecompressV1] Reading gae_comp_data at offset=" << bufferInOffset << std::endl;
                std::vector<uint8_t> gae_comp_data = GetParameter<std::vector<uint8_t>>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] gae_comp_data.size=" << gae_comp_data.size() << std::endl;

                // Reconstruct CompressionMetaData
                std::cerr << "[CompressCAESAR][DecompressV1] Reconstructing CompressionMetaData at offset=" << bufferInOffset << std::endl;
                CompressionMetaData meta;
                meta.offsets = GetParameter<std::vector<float>>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] meta.offsets.size=" << meta.offsets.size() << std::endl;
                meta.scales = GetParameter<std::vector<float>>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] meta.scales.size=" << meta.scales.size() << std::endl;
                meta.indexes = GetParameter<std::vector<std::vector<int32_t>>>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] meta.indexes.size=" << meta.indexes.size() << std::endl;

                // Reconstruct block_info tuple
                std::cerr << "[CompressCAESAR][DecompressV1] Reconstructing block_info at offset=" << bufferInOffset << std::endl;
                int32_t block_info_0 = GetParameter<int32_t>(bufferIn , bufferInOffset);
                int32_t block_info_1 = GetParameter<int32_t>(bufferIn , bufferInOffset);
                std::vector<int32_t> block_info_2 = GetParameter<std::vector<int32_t>>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] block_info_0=" << block_info_0 << " block_info_1=" << block_info_1 << " block_info_2.size=" << block_info_2.size() << std::endl;
                meta.block_info = std::make_tuple(block_info_0 , block_info_1 , block_info_2);

                meta.data_input_shape = GetParameter<std::vector<int32_t>>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] meta.data_input_shape.size=" << meta.data_input_shape.size() << std::endl;
                meta.filtered_blocks = GetParameter<std::vector<std::pair<int32_t , float>>>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] meta.filtered_blocks.size=" << meta.filtered_blocks.size() << std::endl;
                meta.global_scale = GetParameter<float>(bufferIn , bufferInOffset);
                meta.global_offset = GetParameter<float>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] meta.global_scale=" << meta.global_scale << " meta.global_offset=" << meta.global_offset << std::endl;
                meta.padding_recon_info = GetParameter<std::vector<int>>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] meta.padding_recon_info.size=" << meta.padding_recon_info.size() << std::endl;
                meta.pad_T = GetParameter<int64_t>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] meta.pad_T=" << meta.pad_T << std::endl;

                // Reconstruct GAEMetaData
                std::cerr << "[CompressCAESAR][DecompressV1] Reconstructing GAEMetaData at offset=" << bufferInOffset << std::endl;
                GAEMetaData gaeMeta;
                gaeMeta.pcaBasis = GetParameter<std::vector<std::vector<float>>>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] gaeMeta.pcaBasis.size=" << gaeMeta.pcaBasis.size() << std::endl;
                gaeMeta.uniqueVals = GetParameter<std::vector<float>>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] gaeMeta.uniqueVals.size=" << gaeMeta.uniqueVals.size() << std::endl;
                gaeMeta.quanBin = GetParameter<double>(bufferIn , bufferInOffset);
                gaeMeta.nVec = GetParameter<int64_t>(bufferIn , bufferInOffset);
                gaeMeta.prefixLength = GetParameter<int64_t>(bufferIn , bufferInOffset);
                gaeMeta.dataBytes = GetParameter<int64_t>(bufferIn , bufferInOffset);
                gaeMeta.coeffIntBytes = GetParameter<size_t>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] gaeMeta.quanBin=" << gaeMeta.quanBin
                    << " nVec=" << gaeMeta.nVec << " prefixLength=" << gaeMeta.prefixLength
                    << " dataBytes=" << gaeMeta.dataBytes << " coeffIntBytes=" << gaeMeta.coeffIntBytes << std::endl;

          // Reconstruct other CompressionResult fields
                double final_nrmse = GetParameter<double>(bufferIn , bufferInOffset);
                int num_samples = GetParameter<int>(bufferIn , bufferInOffset);
                int num_batches = GetParameter<int>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] final_nrmse=" << final_nrmse << " num_samples=" << num_samples << " num_batches=" << num_batches << std::endl;

                // Read batch_size and n_frame
                int batch_size = GetParameter<int>(bufferIn , bufferInOffset);
                int n_frame = GetParameter<int>(bufferIn , bufferInOffset);
                std::cerr << "[CompressCAESAR][DecompressV1] batch_size=" << batch_size << " n_frame=" << n_frame << std::endl;

                // Reconstruct full CompressionResult
                std::cerr << "[CompressCAESAR][DecompressV1] Reconstructing CompressionResult structure" << std::endl;
                CompressionResult comp;
                comp.encoded_latents = encoded_latents;
                comp.encoded_hyper_latents = encoded_hyper_latents;
                comp.gae_comp_data = gae_comp_data;
                comp.compressionMetaData = meta;
                comp.gaeMetaData = gaeMeta;
                comp.final_nrmse = final_nrmse;
                comp.num_samples = num_samples;
                comp.num_batches = num_batches;

                // Setup device
                std::string device_str = DetectDevice();
                std::cerr << "[CompressCAESAR][DecompressV1] Detected device: " << device_str << std::endl;
                torch::Device device = (device_str == "cuda") ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

                // Decompress
                std::cerr << "[CompressCAESAR][DecompressV1] Creating Decompressor and calling decompress()" << std::endl;
                Decompressor decompressor(device);

                torch::Tensor reconstructed;
                try {
                    reconstructed = decompressor.decompress(
                        encoded_latents ,
                        encoded_hyper_latents ,
                        batch_size ,
                        n_frame ,
                        comp
                    );
                }
                catch (const std::exception& e) {
                    std::cerr << "[CompressCAESAR][DecompressV1] Exception during decompressor.decompress(): " << e.what() << std::endl;
                    throw;
                }
                std::cerr << "[CompressCAESAR][DecompressV1] decompress() returned. reconstructed.sizes=" << reconstructed.sizes() << std::endl;

                // Reshape and crop based on original dimensions
                try {
                    if (ndims == 3) {
                        std::cerr << "[CompressCAESAR][DecompressV1] ndims==3. Before squeeze/reshape sizes=" << reconstructed.sizes() << std::endl;
                        // Original was [T, H, W], reconstructed is [N, 1, 8, H, W]
                        reconstructed = reconstructed.squeeze(1);  // [N, 8, H, W]
                        std::cerr << "[CompressCAESAR][DecompressV1] After squeeze sizes=" << reconstructed.sizes() << std::endl;
                        reconstructed = reconstructed.reshape({ -1, reconstructed.size(-2), reconstructed.size(-1) });  // [N*8, H, W]
                        std::cerr << "[CompressCAESAR][DecompressV1] After reshape sizes=" << reconstructed.sizes() << std::endl;
                        reconstructed = reconstructed.slice(0 , 0 , blockCount[0]);  // [T, H, W]
                        std::cerr << "[CompressCAESAR][DecompressV1] After slice sizes=" << reconstructed.sizes() << std::endl;
                    }
                    else if (ndims == 4) {
                        std::cerr << "[CompressCAESAR][DecompressV1] ndims==4. Before permute sizes=" << reconstructed.sizes() << std::endl;
                        // Original was [C, T, H, W], reconstructed is [N, 1, 8, H, W]
                        reconstructed = reconstructed.permute({ 1, 0, 2, 3, 4 });  // [1, N, 8, H, W]
                        std::cerr << "[CompressCAESAR][DecompressV1] After permute sizes=" << reconstructed.sizes() << std::endl;
                        reconstructed = reconstructed.reshape({ reconstructed.size(0), -1, reconstructed.size(-2), reconstructed.size(-1) });  // [1, N*8, H, W]
                        std::cerr << "[CompressCAESAR][DecompressV1] After reshape sizes=" << reconstructed.sizes() << std::endl;
                        reconstructed = reconstructed.slice(1 , 0 , blockCount[1]);  // [1, T, H, W]
                        std::cerr << "[CompressCAESAR][DecompressV1] After slice sizes=" << reconstructed.sizes() << std::endl;
                        // If original had more channels, we need to handle that
                        if (blockCount[0] > 1) {
                            std::cerr << "[CompressCAESAR][DecompressV1] Expanding channels from 1 to " << blockCount[0] << std::endl;
                            reconstructed = reconstructed.expand({ static_cast<int64_t>(blockCount[0]), -1, -1, -1 });
                            std::cerr << "[CompressCAESAR][DecompressV1] After expand sizes=" << reconstructed.sizes() << std::endl;
                        }
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "[CompressCAESAR][DecompressV1] Exception during reshaping/cropping: " << e.what() << std::endl;
                    throw;
                }

                // Verify final shape
                std::vector<int64_t> expected_shape;
                for (const auto& d : blockCount)
                    expected_shape.push_back(static_cast<int64_t>(d));

                std::cerr << "[CompressCAESAR][DecompressV1] expected_shape=[";
                for (size_t i = 0; i < expected_shape.size(); ++i) {
                    std::cerr << expected_shape[i];
                    if (i < expected_shape.size() - 1) std::cerr << ", ";
                }
                std::cerr << "] reconstructed.sizes=" << reconstructed.sizes() << std::endl;

                if (reconstructed.sizes().vec() != expected_shape) {
                    std::cerr << "[CompressCAESAR][DecompressV1] Shape mismatch. Expected: [";
                    for (size_t i = 0; i < expected_shape.size(); ++i) {
                        std::cerr << expected_shape[i];
                        if (i < expected_shape.size() - 1) std::cerr << ", ";
                    }
                    std::cerr << "], Got: " << reconstructed.sizes() << std::endl;

                    helper::Throw<std::runtime_error>("Operator" , "CompressCAESAR" , "DecompressV1" ,
                        "Final shape mismatch after cropping");
                }

                reconstructed = reconstructed.to(torch::kCPU).contiguous();
                std::cerr << "[CompressCAESAR][DecompressV1] Converted reconstructed to CPU and contiguous. data_ptr=" << reconstructed.data_ptr() << std::endl;

                // Copy to output buffer
                if (type == DataType::Float) {
                    std::cerr << "[CompressCAESAR][DecompressV1] Copying float data to output buffer, sizeOut=" << sizeOut << std::endl;
                    std::memcpy(dataOut , reconstructed.data_ptr<float>() , sizeOut);
                }
                else if (type == DataType::Double) {
                    std::cerr << "[CompressCAESAR][DecompressV1] Converting to double and copying to output buffer, sizeOut=" << sizeOut << std::endl;
                    torch::Tensor double_tensor = reconstructed.to(torch::kFloat64);
                    std::memcpy(dataOut , double_tensor.data_ptr<double>() , sizeOut);
                }
                else {
                    std::cerr << "[CompressCAESAR][DecompressV1] Unsupported DataType encountered during copy: " << static_cast<int>(type) << std::endl;
                    helper::Throw<std::runtime_error>("Operator" , "CompressCAESAR" , "DecompressV1" , "Unsupported DataType in DecompressV1");
                }

                std::cerr << "[CompressCAESAR][DecompressV1] Finished decompression and data copy. Returning sizeOut=" << sizeOut << std::endl;
                return sizeOut;
            }
        } // end namespace compress
    } // end namespace core
} // end namespace adios2