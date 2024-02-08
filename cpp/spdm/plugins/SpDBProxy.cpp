#include "SpDBProxy.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <mdslib.h>
#include <regex>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <vector>

bool load_preprocess(pugi::xml_document &doc, const std::string &path, std::string const &prefix = "");

bool preprocess(pugi::xml_node node, std::string const &prefix)
{
    for (pugi::xml_node child = node.first_child(); child;)
    {
        if (child.type() == pugi::node_element && strcmp(child.name(), "xi:include") == 0)
        {
            pugi::xml_node include = child;

            // load new preprocessed document (note: ideally this should handle relative paths)
            std::string path = include.attribute("href").as_string();

            pugi::xml_document doc;

            if (load_preprocess(doc, path, prefix))
            {
                // insert the comment marker above include directive
                node.insert_child_before(pugi::node_comment, include).set_value(path.c_str());

                // copy the document above the include directive (this retains the original order!)
                for (pugi::xml_node ic = doc.first_child(); ic; ic = ic.next_sibling())
                {
                    node.insert_copy_before(ic, include);
                }
                // remove the include node and move to the next child
                child = child.next_sibling();

                node.remove_child(include);
            }
            else
            {
                child = child.next_sibling();
            }
        }
        else
        {
            if (!preprocess(child, prefix))
                return false;

            child = child.next_sibling();
        }
    }

    return true;
}

bool load_preprocess(pugi::xml_document &doc, std::string const &path, std::string const &prefix)
{
    std::string abs_path = urljoin(prefix, path);

    pugi::xml_parse_result result = doc.load_file(abs_path.c_str(), pugi::parse_default | pugi::parse_pi); // for <?include?>

    std::cout << "Load file [" << abs_path << "] (" << result.description() << ")" << std::endl;

    return result ? preprocess(doc, abs_path) : false;
}

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
