#include <mdslib.h>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <stack>
#include <vector>
#include <iterator>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <regex>
#include "SpDBProxy.h"
#include "SpXMLUtil.h"

struct SpDBProxy::pimpl_s
{
    pugi::xml_document doc;
    pugi::xml_node root;
    std::string m_node_path_prefix_ = "/";
};

SpDBProxy::SpDBProxy() : m_pimpl_(nullptr){};

SpDBProxy::~SpDBProxy() { this->close(); }

int SpDBProxy::init(std::string const &prefix)
{
    if (this->m_pimpl_ == nullptr)
    {
        this->m_pimpl_ = new SpDBProxy::pimpl_s;
    }

    const char *mapping_file_directory = getenv("UDA_EAST_MAPPING_FILE_DIRECTORY");

    std::string path_prefix = mapping_file_directory == nullptr ? prefix : std::string(mapping_file_directory) + "/" + prefix;
    if (load_preprocess(this->m_pimpl_->doc, "mappings.xml", path_prefix))
    {
        this->m_pimpl_->root = this->m_pimpl_->doc.first_child();
        this->m_pimpl_->m_node_path_prefix_ += this->m_pimpl_->root.name();
    }
    else
    {
        throw std::runtime_error("Can not load mapping files from " + path_prefix + "!");
    }
};

int SpDBProxy::reset()
{
    if (this->m_pimpl_ == nullptr)
    {
        this->m_pimpl_ = new SpDBProxy::pimpl_s;
    }

    this->m_pimpl_->doc.reset();
    this->m_pimpl_->root = this->m_pimpl_->doc.first_child();
    this->m_pimpl_->m_node_path_prefix_ += this->m_pimpl_->root.name();
};

int SpDBProxy::close()
{
    delete this->m_pimpl_;
}


int SpDBProxy::fetch(std::string const &request, SpDBObject *data_block) const
{
    std::cout << "Request:" << request;

    std::smatch m;
    if (!std::regex_match(request, m, url_pattern))
    {
        throw std::runtime_error("illegal request! " + request);
    }

    std::string scheme = m[2].length() == 0 ? "mdsplus" : m[2].str();
    std::string authority = m[4].str();
    std::string path = m[5].str();
    std::string query = m[7].str();
    std::string fragment = m[9];

    pugi::xpath_node_set nodes = this->m_pimpl_->doc.select_nodes((this->m_pimpl_->m_node_path_prefix_ + path).c_str());

    std::cout << " selected " << nodes.size() << " nodes." << std::endl;

    if (query == "count")
    {
        int num = nodes.size();
        data_block->set(num);
    }
    else
    {
        for (auto const &node : nodes)
        {
            std::cout << "name=" << node.node().name() << "   type=" << node.node().type() << std::endl;
            for (auto const &child : node.node().children())
            {
                std::cout << "\t child=" << child.name() << "   type=" << child.type() << std::endl;
            }
        }
    }
    return 0;
}
