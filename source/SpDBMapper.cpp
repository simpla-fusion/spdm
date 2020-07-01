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
#include "pugixml/pugixml.hpp"
#include "SpDBMapper.h"

SpDBMapper const &SpDBMapper::load(std::string const &prefix)
{
    static std::map<std::string, std::shared_ptr<SpDBMapper>> mappers;

    auto it = mappers.find(prefix);
    if (it == mappers.end())
    {
        return *mappers.emplace(prefix, std::make_shared<SpDBMapper>(prefix)).first->second;
    }
    else
    {
        return *it->second;
    }
}

struct SpDBMapper::pimpl_s
{
    pugi::xml_document doc;
    pugi::xml_node root;
    std::string m_node_path_prefix_ = "/";
};

std::string path_append(std::string const &prefix, std::string const &path)
{
    std::string res;
    if (path[0] == '/')
    {
        res = path;
    }
    else
    {
        auto pos = prefix.rfind('/');
        if (pos != std::string::npos)
        {
            res = prefix.substr(0, pos) + "/" + path;
        }
        else
        {
            res = path;
        }
    }
    return res;
}

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
    std::string abs_path = path_append(prefix, path);

    pugi::xml_parse_result result = doc.load_file(abs_path.c_str(), pugi::parse_default | pugi::parse_pi); // for <?include?>

    std::cout << "Load file [" << abs_path << "] (" << result.description() << ")" << std::endl;

    return result ? preprocess(doc, abs_path) : false;
}

SpDBMapper::SpDBMapper(std::string const &prefix) : m_pimpl_(new pimpl_s)
{
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

SpDBMapper::~SpDBMapper() { delete this->m_pimpl_; }

int SpDBMapper::init() { return 0; };

int SpDBMapper::reset() { return 0; };

int SpDBMapper::close() { return 0; }

/**
 * https://www.ietf.org/rfc/rfc3986.txt
 * 
 *    scheme    = $2
 *    authority = $4
 *    path      = $5
 *    query     = $7
 *    fragment  = $9
 * 
 * 
*/
static const std::regex url_pattern("(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\\?([^#]*))?(#(.*))?");

static const std::regex path_pattern("([a-zA-Z_\\$][^/#\\[\\]]*)(\\[([^\\[\\]]*)\\])?");

int SpDBMapper::fetch(std::string const &request, SpDBObject *data_block) const
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
        dobj_set_int(data_block, reinterpret_cast<char *>(&num), 0, nullptr);
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
