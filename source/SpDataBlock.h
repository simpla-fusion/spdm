#ifndef SP_DATABLOCK_
#define SP_DATABLOCK_
namespace sp
{

    struct DataBlock
    {
        char *data;
        int nd;
        size_t *dimensions;
        size_t *elementsize;
        char _[];
    };
    class SpDataBlock : public DataBlock
    {
    };

} // namespace sp

#endif // SP_DATABLOCK_