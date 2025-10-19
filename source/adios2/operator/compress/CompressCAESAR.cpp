#include "CompressCAESAR.h"
#include "adios2/helper/adiosFunctions.h"
#include "caesar/models/compress/compressor.h"

namespace adios2
{
    namespace core
    {
        namespace compress
        {
            CompressCAESAR::CompressCAESAR(const Params& parameters)
                : Operator("caesar" , COMPRESS_CAESAR , "compress" , parameters)
            {
            }
            size_t CompressCAESAR::Operate(const char* dataIn , const Dims& blockStart , const Dims& blockCount ,
                const DataType type , char* bufferOut)
            {
                return 0;
            }
            size_t CompressCAESAR::InverseOperate(const char* bufferIn , const size_t sizeIn , char* dataOut)
            {
                return 0;
            }

            // TODO: check what other values work i have tried double and float thus far 
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
                return 0;
            }

        } // end namespace compress
    } // end namespace core
} // end namespace adios2
