#ifndef ADIOS2_OPERATOR_COMPRESS_COMPRESS_CAESAR_H_
#define ADIOS2_OPERATOR_COMPRESS_COMPRESS_CAESAR_H_

#include "adios2/core/Operator.h"

namespace adios2
{
    namespace core
    {
        namespace compress
        {

            class CompressCAESAR : public Operator
            {
            public:
                CompressCAESAR(const Params& parameters);
                ~CompressCAESAR() override = default;
            };

        } // namespace compress
    } // namespace core
} // namespace adios2

#endif // ADIOS2_OPERATOR_COMPRESS_COMPRESS_CAESAR_H_
