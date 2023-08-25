#include "test_helpers.h"
#include "../east_plugin.h"
#include <clientserver/printStructs.h>
#include <clientserver/udaTypes.h>

#define DEMO_PRINT(TAG, fmts, ...) printf(fmts, ##__VA_ARGS__)

void printDataBlock(DATA_BLOCK str)
{
    DEMO_PRINT(UDA_LOG_DEBUG, "\nData Block Contents\n\n");
    DEMO_PRINT(UDA_LOG_DEBUG, "handle       : %d\n", str.handle);
    DEMO_PRINT(UDA_LOG_DEBUG, "error code   : %d\n", str.errcode);
    DEMO_PRINT(UDA_LOG_DEBUG, "error msg    : %s\n", str.error_msg);
    DEMO_PRINT(UDA_LOG_DEBUG, "source status: %d\n", str.source_status);
    DEMO_PRINT(UDA_LOG_DEBUG, "signal status: %d\n", str.signal_status);
    DEMO_PRINT(UDA_LOG_DEBUG, "data_number  : %d\n", str.data_n);
    DEMO_PRINT(UDA_LOG_DEBUG, "rank         : %d\n", str.rank);
    DEMO_PRINT(UDA_LOG_DEBUG, "order        : %d\n", str.order);
    DEMO_PRINT(UDA_LOG_DEBUG, "data_type    : %d\n", str.data_type);
    DEMO_PRINT(UDA_LOG_DEBUG, "error_type   : %d\n", str.error_type);
    DEMO_PRINT(UDA_LOG_DEBUG, "errhi != NULL: %d\n", str.errhi != NULL);
    DEMO_PRINT(UDA_LOG_DEBUG, "errlo != NULL: %d\n", str.errlo != NULL);

    DEMO_PRINT(UDA_LOG_DEBUG, "opaque_type : %d\n", str.opaque_type);
    DEMO_PRINT(UDA_LOG_DEBUG, "opaque_count: %d\n", str.opaque_count);

    switch (str.opaque_type)
    {
    case (UDA_OPAQUE_TYPE_XML_DOCUMENT):
        if (str.opaque_block != NULL)
            DEMO_PRINT(UDA_LOG_DEBUG, "\nXML: %s\n\n", (char *)str.opaque_block);
        break;
    default:
        break;
    }

    int k = 10;
    if (str.data_n < 10)
    {
        k = str.data_n;
    }

    if (str.data_type == UDA_TYPE_FLOAT)
    {
        int j;
        for (j = 0; j < k; j++)
        {
            DEMO_PRINT(UDA_LOG_DEBUG, "data[%d]: %f\n", j, *((float *)str.data + j));
        }
    }
    if (str.data_type == UDA_TYPE_DOUBLE)
    {
        int j;
        for (j = 0; j < k; j++)
        {
            DEMO_PRINT(UDA_LOG_DEBUG, "data[%d]: %f\n", j, *((double *)str.data + j));
        }
    }

    if (str.error_type == UDA_TYPE_FLOAT && str.errhi != NULL)
    {
        int j;
        for (j = 0; j < k; j++)
        {
            DEMO_PRINT(UDA_LOG_DEBUG, "errhi[%d]: %f\n", j, *((float *)str.errhi + j));
        }
    }

    if (str.error_type == UDA_TYPE_FLOAT && str.errlo != NULL && str.errasymmetry)
    {
        int j;
        for (j = 0; j < k; j++)
        {
            DEMO_PRINT(UDA_LOG_DEBUG, "errlo[%d]: %f\n", j, *((float *)str.errlo + j));
        }
    }

    DEMO_PRINT(UDA_LOG_DEBUG, "error model : %d\n", str.error_model);
    DEMO_PRINT(UDA_LOG_DEBUG, "asymmetry   : %d\n", str.errasymmetry);
    DEMO_PRINT(UDA_LOG_DEBUG, "error model no. params : %d\n", str.error_param_n);

    int i;
    for (i = 0; i < str.error_param_n; i++)
    {
        DEMO_PRINT(UDA_LOG_DEBUG, "param[%d] = %f \n", i, str.errparams[i]);
    }

    DEMO_PRINT(UDA_LOG_DEBUG, "data_units  : %s\n", str.data_units);
    DEMO_PRINT(UDA_LOG_DEBUG, "data_label  : %s\n", str.data_label);
    DEMO_PRINT(UDA_LOG_DEBUG, "data_desc   : %s\n", str.data_desc);

    for (i = 0; i < (int)str.rank; i++)
    {
        DEMO_PRINT(UDA_LOG_DEBUG, "\nDimension #%d Contents\n\n", i);
        DEMO_PRINT(UDA_LOG_DEBUG, "data_type    : %d\n", str.dims[i].data_type);
        DEMO_PRINT(UDA_LOG_DEBUG, "error_type   : %d\n", str.dims[i].error_type);
        DEMO_PRINT(UDA_LOG_DEBUG, "errhi != NULL: %d\n", str.dims[i].errhi != NULL);
        DEMO_PRINT(UDA_LOG_DEBUG, "errlo != NULL: %d\n", str.dims[i].errlo != NULL);
        DEMO_PRINT(UDA_LOG_DEBUG, "error model  : %d\n", str.dims[i].error_model);
        DEMO_PRINT(UDA_LOG_DEBUG, "asymmetry    : %d\n", str.dims[i].errasymmetry);
        DEMO_PRINT(UDA_LOG_DEBUG, "error model no. params : %d\n", str.dims[i].error_param_n);

        int j;
        for (j = 0; j < str.dims[i].error_param_n; j++)
        {
            DEMO_PRINT(UDA_LOG_DEBUG, "param[%d] = %f \n", j, str.dims[i].errparams[j]);
        }

        DEMO_PRINT(UDA_LOG_DEBUG, "data_number : %d\n", str.dims[i].dim_n);
        DEMO_PRINT(UDA_LOG_DEBUG, "compressed? : %d\n", str.dims[i].compressed);
        DEMO_PRINT(UDA_LOG_DEBUG, "method      : %d\n", str.dims[i].method);

        if (str.dims[i].method == 0)
        {
            if (str.dims[i].compressed)
            {
                DEMO_PRINT(UDA_LOG_DEBUG, "starting val: %f\n", str.dims[i].dim0);
                DEMO_PRINT(UDA_LOG_DEBUG, "stepping val: %f\n", str.dims[i].diff);
            }
            else
            {
                if (str.dims[i].data_type == UDA_TYPE_FLOAT)
                {
                    k = 10;
                    if (str.dims[i].dim_n < 10)
                        k = str.dims[i].dim_n;
                    if (str.dims[i].dim != NULL)
                        for (j = 0; j < k; j++)
                            DEMO_PRINT(UDA_LOG_DEBUG, "val[%d] = %f\n", j, *((float *)str.dims[i].dim + j));
                }
                if (str.dims[i].data_type == UDA_TYPE_DOUBLE)
                {
                    k = 10;
                    if (str.dims[i].dim_n < 10)
                        k = str.dims[i].dim_n;
                    if (str.dims[i].dim != NULL)
                        for (j = 0; j < k; j++)
                            DEMO_PRINT(UDA_LOG_DEBUG, "val[%d] = %f\n", j, *((double *)str.dims[i].dim + j));
                }
            }
        }
        else
        {
            DEMO_PRINT(UDA_LOG_DEBUG, "udoms: %d\n", str.dims[i].udoms);
            switch (str.dims[i].method)
            {
            case 1:
                if (str.dims[i].data_type == UDA_TYPE_FLOAT)
                {
                    k = 10;
                    if (str.dims[i].udoms < 10)
                        k = str.dims[i].udoms;
                    for (j = 0; j < k; j++)
                    {
                        DEMO_PRINT(UDA_LOG_DEBUG, "sams[%d]: %d\n", j, (int)*(str.dims[i].sams + j));
                        DEMO_PRINT(UDA_LOG_DEBUG, "offs[%d]: %f\n", j, *((float *)str.dims[i].offs + j));
                        DEMO_PRINT(UDA_LOG_DEBUG, "ints[%d]: %f\n", j, *((float *)str.dims[i].ints + j));
                    }
                }
                if (str.dims[i].data_type == UDA_TYPE_DOUBLE)
                {
                    k = 10;
                    if (str.dims[i].udoms < 10)
                        k = str.dims[i].udoms;
                    for (j = 0; j < k; j++)
                    {
                        DEMO_PRINT(UDA_LOG_DEBUG, "sams[%d]: %d\n", j, (int)*(str.dims[i].sams + j));
                        DEMO_PRINT(UDA_LOG_DEBUG, "offs[%d]: %f\n", j, *((double *)str.dims[i].offs + j));
                        DEMO_PRINT(UDA_LOG_DEBUG, "ints[%d]: %f\n", j, *((double *)str.dims[i].ints + j));
                    }
                }
                break;
            case 2:
                if (str.dims[i].data_type == UDA_TYPE_FLOAT)
                {
                    k = 10;
                    if (str.dims[i].udoms < 10)
                        k = str.dims[i].udoms;
                    for (j = 0; j < k; j++)
                        DEMO_PRINT(UDA_LOG_DEBUG, "offs[%d]: %f\n", j, *((float *)str.dims[i].offs + j));
                }
                if (str.dims[i].data_type == UDA_TYPE_DOUBLE)
                {
                    k = 10;
                    if (str.dims[i].udoms < 10)
                        k = str.dims[i].udoms;
                    for (j = 0; j < k; j++)
                        DEMO_PRINT(UDA_LOG_DEBUG, "offs[%d]: %f\n", j, *((double *)str.dims[i].offs + j));
                }
                break;
            case 3:
                if (str.dims[i].data_type == UDA_TYPE_FLOAT)
                {
                    DEMO_PRINT(UDA_LOG_DEBUG, "offs[0] val: %f\n", *((float *)str.dims[i].offs));
                    DEMO_PRINT(UDA_LOG_DEBUG, "ints[0] val: %f\n", *((float *)str.dims[i].ints));
                }
                if (str.dims[i].data_type == UDA_TYPE_DOUBLE)
                {
                    DEMO_PRINT(UDA_LOG_DEBUG, "offs[0] val: %f\n", *((double *)str.dims[i].offs));
                    DEMO_PRINT(UDA_LOG_DEBUG, "ints[0] val: %f\n", *((double *)str.dims[i].ints));
                }
                break;
            default:
                DEMO_PRINT(UDA_LOG_WARN, "unknown method (%d) for dim (%d)", str.dims[i].method, i);
            }
        }
        if (str.dims[i].error_type == UDA_TYPE_FLOAT)
        {
            k = 10;
            if (str.dims[i].dim_n < 10)
                k = str.dims[i].dim_n;
            if (str.dims[i].errhi != NULL)
                for (j = 0; j < k; j++)
                    DEMO_PRINT(UDA_LOG_DEBUG, "errhi[%d] = %f\n", j, *((float *)str.dims[i].errhi + j));
            if (str.dims[i].errlo != NULL && str.dims[i].errasymmetry)
                for (j = 0; j < k; j++)
                    DEMO_PRINT(UDA_LOG_DEBUG, "errlo[%d] = %f\n", j, *((float *)str.dims[i].errlo + j));
        }
        DEMO_PRINT(UDA_LOG_DEBUG, "data_units  : %s\n", str.dims[i].dim_units);
        DEMO_PRINT(UDA_LOG_DEBUG, "data_label  : %s\n", str.dims[i].dim_label);
    }
}

int main(int argc, char const *argv[])
{
#ifdef FATCLIENT
#include "setup.inc"
#endif

    IDAM_PLUGIN_INTERFACE plugin_interface = uda::test::generate_plugin_interface(argv[1]);
    // IDAM_PLUGIN_INTERFACE plugin_interface = uda::test::generate_plugin_interface("east::read(element='wall/description_2d/#/type/index', indices='1', experiment='EAST', dtype=3, shot=" SHOT_NUM ", IDS_version='')");

    east_plugin(&plugin_interface);
    printDataBlock(*plugin_interface.data_block);
    east_plugin(&plugin_interface);
    printDataBlock(*plugin_interface.data_block);

    IDAM_PLUGIN_INTERFACE plugin_interface_close = uda::test::generate_plugin_interface("east::close()");
    east_plugin(&plugin_interface_close);
    return 0;
}
