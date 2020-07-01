#include "SpDBObject.h"
#include "SpDBHandler.h"

#include "pugixml/pugixml.hpp"
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <iostream>

class XMLNodeHandleBase
{
public:
    XMLNodeHandleBase() = default;

    virtual void fetch(pugi::xml_node const &node, size_t id, SpDBObject *data_block, int dtype) = 0;

    void reconnect(std::map<std::string, std::string> const &attrs){};
};

// class XMLNodeHandleStatic : public XMLNodeHandleBase
// {
// public:
//     XMLNodeHandleStatic(std::map<std::string, std::string> const &attrs);
//     void fetch(pugi::xml_node const &node, size_t id, SpDBObject *data_block, int dtype);
// };
// XMLNodeHandleStatic ::XMLNodeHandleStatic(std::map<std::string, std::string> const &attrs) {}

// void XMLNodeHandleStatic::fetch(pugi::xml_node const &node, size_t id, SpDBObject *data_block, int dtype)
// {
//     // switch (dtype)
//     // {
//     // case UDA_TYPE_INT:
//     // {
//     //     int i = 0;
//     //     int *data = (int *)malloc(num * sizeof(int));
//     //     while (tokens[i] != NULL && i < num)
//     //     {
//     //         data[i] = (int)strtol(tokens[i], NULL, 10);
//     //         ++i;
//     //     }
//     //     setReturnDataIntArray(data_block, data, rank, indices, NULL);
//     //     free(data);
//     // }
//     // break;
//     // case UDA_TYPE_FLOAT:
//     // {
//     //     int i = 0;
//     //     float *data = (float *)malloc(num * sizeof(float));
//     //     while (tokens[i] != NULL && i < num)
//     //     {
//     //         data[i] = (float)strtof(tokens[i], NULL);
//     //         ++i;
//     //     }
//     //     setReturnDataFloatArray(data_block, data, rank, indices, NULL);
//     //     free(data);
//     // }
//     // break;
//     // case UDA_TYPE_DOUBLE:
//     // {
//     //     int i = 0;
//     //     double *data = (double *)malloc(num * sizeof(double));
//     //     while (tokens[i] != NULL && i < num)
//     //     {
//     //         data[i] = (double)strtod(tokens[i], NULL);
//     //         ++i;
//     //     }
//     //     UDA_LOG(UDA_LOG_DEBUG, "'%s'=%i,%i,%i\n", content, rank, indices[0], indices[1]);
//     //     setReturnDataDoubleArray(data_block, data, rank, indices, NULL);
//     //     printDataBlock(*data_block);
//     //     free(data);
//     // }
//     // break;
//     //     // case UDA_TYPE_STRING:
//     //     //     setReturnDataString(data_block, (char *)content, NULL);
//     //     //     break;
//     //     // default:
//     //     // RAISE_PLUGIN_ERROR("unknown dtype given to plugin");
//     // }
//     // FreeSplitStringTokens(&tokens);
//     // UDA_LOG(UDA_LOG_DEBUG, "BALA TYPENAME(%i)", dtype);
//     // printDataBlock(*data_block);
// }

class XMLNodeHandleMDSplus : public XMLNodeHandleBase
{
public:
    XMLNodeHandleMDSplus(std::map<std::string, std::string> const &attrs);

    void reconnect(std::map<std::string, std::string> const &attrs);

    void fetch(pugi::xml_node const &node, size_t id, SpDBObject *data_block, int dtype) override;

private:
    std::string mds_host = "";
    std::string mds_tree = "";
    int mds_shot = -1;
    int mds_socket = -1;

    void open(int id);
    void close();

    int handle_mdsplus(const std::string &host,
                       const std::string &tree, int shot,
                       const std::string &signalName,
                       int dtype,
                       SpDBObject *data_block);
};

// std::string get_env(std::string const &key)
// {
//     const char *res = getenv(key.c_str());
//     if (res == nullptr)
//     {
//         return "";
//     }
//     else
//     {
//         return std::string(res);
//     }
// }
// XMLNodeHandleMDSplus::XMLNodeHandleMDSplus(std::map<std::string, std::string> const &attrs)
// {
//     this->mds_host = get_value(attrs, "mds_host");
//     if (this->mds_host == "")
//     {
//         this->mds_host = get_env("UDA_EAST_MDS_HOSTNAME");
//     };
//     if (this->mds_host == "")
//     {
//         this->mds_host = "202.127.204.12";
//     }
//     this->mds_tree = get_value(attrs, "mds_tree");
//     if (this->mds_tree == "")
//     {
//         this->mds_tree = get_env("UDA_EAST_MDS_TREENAME");
//     };
//     if (this->mds_tree == "")
//     {
//         this->mds_tree = "EFIT_EAST";
//     }
// }
// int mds_get_int(std::string const &expression, int *value, int num)
// {
//     /* local vars */
//     /** FIXME: without this line log , result is worng! why ????*/
//     UDA_LOG(UDA_LOG_DEBUG, "EAST: %s num=%i.\n", expression, num);
//     int dtype_long = DTYPE_LONG;
//     int lnum = num;
//     int lvalue[10];
//     int status = 0;
//     int null = 0;
//     // if (lnum >10){
//     //   UDA_LOG(UDA_LOG_ERROR, "TOO LONG!  %s. num=%i\n", expression,num);
//     //   return -1;
//     // }
//     int idesc = descr(&dtype_long, lvalue, &num, &null);
//     status = MdsValue(const_cast<char *>(expression.c_str()), &idesc, &null, &lnum);
//     if (!((status & 1) == 1))
//     {
//         UDA_LOG(UDA_LOG_ERROR, "MdsValue failed !  %s.\n", expression);
//     }
//     else
//     {
//         for (int i = 0; i < num; ++i)
//         {
//             value[i] = (int)lvalue[i];
//         }
//     }
//     return status;
// }
// int mds_get_double(std::string const &expression, double *value, int num)
// {
//     /* local vars */
//     /** FIXME: without this line log , result is worng! why ????*/
//     UDA_LOG(UDA_LOG_DEBUG, "EAST: %s num=%i.\n", expression, num);
//     int dtype_double = DTYPE_DOUBLE;
//     int lnum = num;
//     double lvalue[10];
//     int status = 0;
//     int null = 0;
//     // if (lnum >10){
//     //   UDA_LOG(UDA_LOG_ERROR, "TOO LONG!  %s. num=%i\n", expression,num);
//     //   return -1;
//     // }
//     int idesc = descr(&dtype_double, lvalue, &num, &null);
//     status = MdsValue(const_cast<char *>(expression.c_str()), &idesc, &null, &lnum);
//     if (!((status & 1) == 1))
//     {
//         UDA_LOG(UDA_LOG_ERROR, "MdsValue failed !  %s.\n", expression);
//     }
//     else
//     {
//         for (int i = 0; i < num; ++i)
//         {
//             value[i] = (double)lvalue[i];
//         }
//     }
//     return status;
// }

// void XMLNodeHandleMDSplus::open(int id)
// {
//     this->mds_shot = 0;
//     this->mds_socket = -1;
//     this->mds_socket = MdsConnect(const_cast<char *>(this->mds_host.c_str()));

//     if (this->mds_tree != "" && id == this->mds_shot)
//     { // do nothing
//     }
//     else
//     {
//         this->close();
//         this->mds_shot = id;

//         /* Open east_shot tree for requested shot */
//         int status = MdsOpen(const_cast<char *>(this->mds_tree.c_str()), &(this->mds_shot));

//         if (!((status & 1) == 1))
//         {
//             UDA_LOG(UDA_LOG_DEBUG, "EAST: Error opening the east_shot tree.\n");
//             UDA_LOG(UDA_LOG_ERROR, "Error opening the east_shot tree.\n");
//             return;
//         }
//         else
//         {
//             UDA_LOG(UDA_LOG_DEBUG, "EAST: MDS+ tree open successfull\n");
//         }
//     }
// }
// void XMLNodeHandleMDSplus::close()
// {
//     if (this->mds_shot != -1)
//     {
//         MdsClose(const_cast<char *>(this->mds_tree.c_str()), &(this->mds_shot));
//         // this->mds_tree = "";
//     }
//     if (this->mds_host != "")
//     {
//         MdsDisconnect();
//         this->mds_host = "";
//         this->mds_socket = -1;
//     }
// }

// void XMLNodeHandleMDSplus::fetch(pugi::xml_node const &node, size_t id, SpDBObject *data_block, int dtype)
// {
//     std::string request = node.child_value();
//     std::cout << " request =" << request << std::endl;

//     // int XMLNodeHandleMDSplus::handle_mdsplus(const std::string &host, const std::string &tree, int shot, const std::string &signalName, int dtype,SpDBObject *data_block)
//     std::cout << this->mds_host << " " << this->mds_tree << " " << id << std::endl;

//     this->open(id);
// }

class XMLNodeHandle
{
public:
    XMLNodeHandle() = default;

    template <typename TAttrs>
    void connect(std::string const &handletype, TAttrs const &attributes)
    {
        std::map<std::string, std::string> attrs;
        for (auto const &a : attributes)
        {
            attrs[a.name()] = a.value();
        }

        // if (handletype == "static")
        // {
        //     this->pimpl_ = std::make_shared<XMLNodeHandleStatic>(attrs, args...);
        // }
        // else if (handletype == "mdsplus")
        // {
        //     this->pimpl_ = std::make_shared<XMLNodeHandleMDSplus>(attrs, args...);
        // }
        // else
        // {
        //     throw std::runtime_error(handletype);
        // }
    }
    int fetch(pugi::xml_node const &node, size_t id, SpDBObject *data_block)
    {
        std::string dtype = "string";

        node.attribute("dtype").as_string();

        auto a_dtype = node.attribute("dtype");

        if (a_dtype.empty())
        {
            dtype = a_dtype.as_string();
        }

        int success = 0;

        if (!node.attribute("value").empty())
        {
            success = data_block->set(node.attribute("value").as_string(), dtype.c_str());
        }
        else if (!node.text().empty())
        {
            success = data_block->set(node.text().as_string(), dtype.c_str());
        }
        else if (!node.first_child().empty())
        {
            std::cout << node.first_child().name() << std::endl;
            // this->m_handler_->fetch(data_block, id, dtype, node);
        }
        return success;
    }

private:
    std::vector<std::string> m_handler_type_;
    std::map<std::string, std::shared_ptr<XMLNodeHandleBase>> m_handlers_;
};

#define STR_TO_INT_ARRAY(_STR_, _LEN_, _DATA_)                   \
    {                                                            \
        int maxLen = _LEN_;                                      \
        char **__tokens = SplitString((char const *)_STR_, ","); \
        int i = 0;                                               \
        while (__tokens[i] != NULL && i < maxLen)                \
        {                                                        \
            _DATA_[i] = (int)strtol(__tokens[i], NULL, 10);      \
            ++i;                                                 \
        }                                                        \
        _LEN_ = i;                                               \
        FreeSplitStringTokens(&__tokens);                        \
    }

#define STR_TO_DOUBLE_ARRAY(_STR_, _LEN_, _DATA_)                \
    {                                                            \
        int maxLen = _LEN_;                                      \
        char **__tokens = SplitString((char const *)_STR_, ","); \
        int i = 0;                                               \
        while (__tokens[i] != NULL && i < maxLen)                \
        {                                                        \
            _DATA_[i] = (int)strtod(__tokens[i]);                \
            ++i;                                                 \
        }                                                        \
        _LEN_ = i;                                               \
        FreeSplitStringTokens(&__tokens);                        \
    }
