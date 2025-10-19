#ifndef ADIOS2_OPERATOR_COMPRESS_COMPRESS_CAESAR_H_
#define ADIOS2_OPERATOR_COMPRESS_COMPRESS_CAESAR_H_
#include "adios2/core/Operator.h"
// TODO: find out what other methdos and variables are needed this is what both MGARD and SZ have
namespace adios2
{
    namespace core
    {
        namespace compress
        {

            class CompressCAESAR : public Operator
            {
            public:
                 /**
     * @param dataIn
     * @param blockStart
     * @param blockCount
     * @param type
     * @param bufferOut
     * @return size of compressed buffer
     */
                CompressCAESAR(const Params& parameters);
                ~CompressCAESAR() override = default;

                size_t Operate(const char* dataIn , const Dims& blockStart , const Dims& blockCount ,
                    const DataType type , char* bufferOut) final;

                size_t InverseOperate(const char* bufferIn , const size_t sizeIn , char* dataOut) final;
                bool IsDataTypeValid(const DataType type) const final;

            private:
                /**
     * Decompress function for V1 buffer. Do NOT remove even if the buffer
     * version is updated. Data might be still in lagacy formats. This function
     * must be kept for backward compatibility
     * @param bufferIn : compressed data buffer (V1 only)
     * @param sizeIn : number of bytes in bufferIn
     * @param dataOut : decompressed data buffer
     * @return : number of bytes in dataOut
     */
                size_t DecompressV1(const char* bufferIn , const size_t sizeIn , char* dataOut); // idk what this is
                std::string m_VersionInfo; // this is the model
            };

        } // namespace compress
    } // namespace core
} // namespace adios2

#endif // ADIOS2_OPERATOR_COMPRESS_COMPRESS_CAESAR_H_
