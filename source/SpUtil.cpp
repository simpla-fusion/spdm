#include "SpDBUtil.h"
#include "pugixml/pugixml.hpp"

#include <string>

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
