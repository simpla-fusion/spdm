#include "DataBlock.h"
#include "../utility/Logger.h"
namespace sp::db
{

DataBlock::DataBlock(int nd, const size_t* dimensions, void* data, int element_size)
{
    NOT_IMPLEMENTED;
}
DataBlock::DataBlock(DataBlock const&)
{
    NOT_IMPLEMENTED;
}
DataBlock::DataBlock(DataBlock&&)
{
    NOT_IMPLEMENTED;
}

void DataBlock::swap(DataBlock&) { NOT_IMPLEMENTED; }

} // namespace sp::db