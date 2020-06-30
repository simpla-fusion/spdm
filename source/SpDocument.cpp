#include "SpDocument.h"
#include <cstdlib>
#include <cstring>
#include "SpUtil.h"
#include "pugixml/pugixml.hpp"
SpOID::SpOID() : m_id_(0)
{
    // TODO:  random  init m_id_
}

SpXPath::SpXPath(std::string const &path) : m_path_(path) {}
SpXPath::SpXPath(const char *path) : m_path_(path) {}
// SpXPath::~SpXPath() = default;
// SpXPath::SpXPath(SpXPath &&) = default;
// SpXPath::SpXPath(SpXPath const &) = default;
// SpXPath &SpXPath::operator=(SpXPath const &) = default;

std::string const &SpXPath::value() const { return m_path_; }

SpXPath SpXPath::operator/(std::string const &suffix) const
{
    return SpXPath(urljoin(m_path_, suffix));
}
SpXPath::operator std::string() const { return m_path_; }

SpAttribute::SpAttribute() { ; }
SpAttribute::~SpAttribute() { ; }
SpAttribute::SpAttribute(SpAttribute &&other) { ; }

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

SpNode::SpNode() {}
SpNode::~SpNode() {}
SpNode::SpNode(SpNode &&) {}
SpNode::range SpNode::select(SpXPath const &path)
{
    SpNode::range nodes;
    return std::move(nodes);
}
const SpNode::range SpNode::select(SpXPath const &path) const
{
    SpNode::range nodes;
    return std::move(nodes);
}
SpAttribute SpNode::attribute(std::string const &)
{
    SpAttribute attr;
    return std::move(attr);
}
SpNode SpNode::child()
{
    SpNode node;
    return std::move(node);
}

SpDocument::SpDocument()
{
}

SpDocument::~SpDocument()
{
}

SpDocument::SpDocument(SpDocument &&){};
int SpDocument::load(std::string const &) { return 0; }
int SpDocument::save(std::string const &) { return 0; }
int SpDocument::load(std::istream const &) { return 0; }
int SpDocument::save(std::ostream const &) { return 0; }

SpNode SpDocument::root()
{
    SpNode node;
    return std::move(node);
}
